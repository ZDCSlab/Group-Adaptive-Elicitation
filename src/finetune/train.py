from .training_utils import *
import torch 
import os 
import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import wandb
from dataset.common import TextDataClass
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from transformers import set_seed  # works with recent transformers
from tqdm import trange
from .adapter import *
from accelerate import Accelerator
from accelerate.utils import set_seed  
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import trange

def train_accelerate(args):
    mp = (
        "bf16" if getattr(args, "dtype", "").lower() in ["bfloat16", "bf16"]
        else "fp16" if getattr(args, "dtype", "").lower() in ["float16", "fp16"]
        else "no"
    )
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))

    accelerator = Accelerator(mixed_precision=mp)
    set_seed(args.seed, device_specific=True)  
    logger_print = accelerator.print

    local_bs = args.batch_size
    world_size = accelerator.num_processes
    global_bs =  world_size *  local_bs

    # 2) Tokenizer & Data
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<EOS>", "<EOP>", "[PAD]", "<Answer>"]})

    logger_print("loading data")
    text_data_dict = load_dataset(args.data_dir)
    
    dataset = TextDataClass(text_data_dict, tokenizer, mode=args.mode)
    best_val_loss = 1e9
    logger_print(list(dataset.data_dict.keys()))

    # 3) token ids
    cand_ids_tensor = dataset.build_cand_ids_tensor(
        tokenizer,
        candidate_strs=["A", "B"],
        device=accelerator.device,
        strict_single_token=True
    )
    logger_print(f'cand_ids_tensor shape={tuple(cand_ids_tensor.shape)}')

    # 4) Model & Optimizer
    torch_dtype = (
        torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    )
    logger_print("loading model")
    logger_print(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))

    if getattr(args, "peft", False):
        lora_config = LoraConfig(
            r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.target_modules
        )
        model = get_peft_model(model, lora_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()


    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), betas=args.betas, weight_decay=args.weight_decay)
    total_steps = args.epochs           
    warmup_steps = max(100, int(0.06 * total_steps))  

    scheduler = get_cosine_schedule_with_min_lr(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    min_lr_ratio=0.1   # lr 10%
    )

    # 5) 
    model, optimizer = accelerator.prepare(model, optimizer)

    # 6) 
    if getattr(args, "wandb", False) and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # 7) Training Loop
    model.train()
    pbar = trange(args.epochs, desc="Iteration", disable=not accelerator.is_main_process)

    for iter_num in pbar:
        seed_for_this_iter = args.seed + iter_num * 100003 + accelerator.process_index
        set_seed(seed_for_this_iter, device_specific=True)

        # —— get one batch
        X, Y, gradient_mask, nmask = dataset.get_batch('train', args.batch_size, args.block_size)
        X = X.to(accelerator.device)
        Y = Y.to(accelerator.device)
        gradient_mask = gradient_mask.to(accelerator.device)
        nmask = nmask.to(accelerator.device)

        # with accelerator.accumulate(model):
        loss = get_loss_with_candidates(model, X, Y, gradient_mask, nmask, return_scalar=True, mask=True, cand_ids_tensor=cand_ids_tensor, device=accelerator.device)
        accelerator.backward(loss)

        # —— Grad Clip
        if getattr(args, "grad_clip", 0.0):
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
            # —— Logging
            if accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
              
                if grad_accum_steps > 1:
                    eff_step = (iter_num + 1) // grad_accum_steps
                else:
                    eff_step = iter_num + 1

                pbar.set_postfix({"loss": mean_loss, "eff_step": eff_step})

                log_obj = {
                    "iter": iter_num,
                    "eff_step": eff_step,
                    "lr": lr,
                    "train/loss_step": float(mean_loss),
                }
                print(log_obj)
                if getattr(args, "wandb", False):
                    wandb.log(log_obj)

        # —— 
        if getattr(args, "debug", False) and accelerator.is_main_process:
            with torch.no_grad():
                active = gradient_mask.sum().item()
                print(f"[iter {iter_num}] masked_positions={active}")
                if active > 0:
                    b, t = (gradient_mask > 0).nonzero(as_tuple=False)[0].tolist()
                    target_id = Y[b, t].item()
                    print("target(after <Answer>):", tokenizer.decode([target_id]))
                    logits_bt = model(input_ids=X[b:b+1]).logits[0, t]
                    topk = torch.topk(logits_bt, 5)
                    ids, vals = topk.indices.tolist(), topk.values.softmax(dim=0).tolist()
                    print("top-5 preds:", [(tokenizer.decode([i]), p) for i, p in zip(ids, vals)])

        # —— Evaluation
        if (iter_num % args.eval_interval == 0) or (iter_num == args.epochs - 1):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"[iter {iter_num}] Evaluation")
                model.eval()
                with torch.no_grad():
                    core_model = accelerator.unwrap_model(model)

                    train_loss, train_acc = estimate_loss_and_acc_with_candidates(
                        core_model, dataset, split='train',
                        device=accelerator.device, batch_size=args.batch_size,
                        block_size=args.block_size, cand_ids_tensor=cand_ids_tensor
                    )
                    val_loss, val_acc = estimate_loss_and_acc_with_candidates(
                        core_model, dataset, split='val',
                        device=accelerator.device, batch_size=args.batch_size,
                        block_size=args.block_size, cand_ids_tensor=cand_ids_tensor
                    )
                    test_loss, test_acc = estimate_loss_and_acc_with_candidates(
                        core_model, dataset, split='test',
                        device=accelerator.device, batch_size=args.batch_size,
                        block_size=args.block_size, cand_ids_tensor=cand_ids_tensor
                    )

                print(f"[iter {iter_num}] lr={lr:.3e}  train={train_loss:.4f} val={val_loss:.4f} test={test_loss:.4f}")
                print(f"[iter {iter_num}] lr={lr:.3e}  train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}")

                if getattr(args, "wandb", False):
                    wandb.log({
                        "iter": iter_num, "lr": lr,
                        "train/eval_loss": train_loss, "val/eval_loss": val_loss, "test/eval_loss": test_loss,
                        "train/accuracy": train_acc, "val/accuracy": val_acc, "test/accuracy": test_acc
                    })

                # —— 
                if val_loss < best_val_loss and getattr(args, "save_dir", None):
                    best_val_loss = val_loss
                    core_model.save_pretrained(args.save_dir)
                    accelerator.save_state(args.save_dir)
                    print(f"✓ Saved best checkpoint to {args.save_dir} (val={val_loss:.4f})")

                model.train()
            accelerator.wait_for_everyone()
