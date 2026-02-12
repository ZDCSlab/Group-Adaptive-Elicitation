
import torch 
import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from transformers import set_seed  
from tqdm import trange

from .dataset import QTextDataset
from .training_utils import *


# =========================
# Helpers
# =========================
def _to_device(batch, device):
    """Move a tuple/list of tensors to device, keep None as None."""
    out = []
    for x in batch:
        if x is None:
            out.append(None)
        else:
            out.append(x.to(device))
    return out

def _compute_total_steps(iters: int, grad_accum_steps: int) -> int:
    """
    User-defined convention:
      - args.epochs == iters
      - optimizer steps == iters // grad_accum_steps  (floor, no tail flush)
    """
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps}")
    return iters // grad_accum_steps

def _compute_warmup_steps(total_steps: int, warmup_ratio: float = 0.06) -> int:
    """Pure ratio warmup; clamp to [0, total_steps]."""
    if total_steps <= 0:
        return 0
    ws = int(warmup_ratio * total_steps)
    ws = max(0, min(ws, total_steps))
    return ws

def _log_lr_sanity(
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float,
):
    """Print a brief schedule summary on main process."""
    if not accelerator.is_main_process:
        return
    min_lr = base_lr * float(min_lr_ratio)
    accelerator.print(
        f"[LR] base_lr={base_lr:.3e}  min_lr={min_lr:.3e} (ratio={min_lr_ratio})  "
        f"total_steps={total_steps}  warmup_steps={warmup_steps}"
    )
    # Print first few LRs by simulating scheduler stepping on a copy
    # (Non-invasive: we won't mutate current optimizer/scheduler)
    # If your scheduler is stateful and can't be trivially copied, skip this.
    try:
        import copy
        opt2 = copy.deepcopy(optimizer)
        sch2 = copy.deepcopy(scheduler)
        lrs = []
        for s in range(min(12, max(0, total_steps))):
            # mimic: optimizer.step(); scheduler.step()
            sch2.step()
            lrs.append(opt2.param_groups[0]["lr"])
        accelerator.print(f"[LR] first_steps(lr after scheduler.step): {['%.3e'%x for x in lrs]}")
    except Exception as e:
        accelerator.print(f"[LR] (skip preview) {type(e).__name__}: {e}")


# =========================
# Main training
# =========================

def train_accelerate(args):
    mp = (
        "bf16" if getattr(args, "dtype", "").lower() in ["bfloat16", "bf16"]
        else "fp16" if getattr(args, "dtype", "").lower() in ["float16", "fp16"]
        else "no"
    )
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    accelerator = Accelerator(mixed_precision=mp, gradient_accumulation_steps=grad_accum_steps)
    logger_print = accelerator.print

    # ---- Reproducibility
    set_seed(args.seed, device_specific=True)

    # ---- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<Question>", "<Answer>"]})

    if accelerator.is_main_process and getattr(args, "save_dir", None):
        os.makedirs(args.save_dir, exist_ok=True)
        tokenizer.save_pretrained(args.save_dir)

    # ---- Load option dict (qid -> tokenized options)
    qid_to_options = None
    if getattr(args, "option_dict_path", None):
        qid_to_options = load_qid_to_tokenized_options(
            args.option_dict_path,
            tokenizer,
            id_key=getattr(args, "option_id_key", "id"),
        )

    # ---- Load data + dataset
    text_data_dict = load_dataset(args.data_dir)
    dataset = QTextDataset(
        text_data_dict,
        tokenizer,
        args.block_size,
        option_dict=qid_to_options,
    )
    logger_print(f"[data] splits={list(getattr(dataset, 'splits', []))}")

    # ---- Model
    torch_dtype = (
        torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    )
    logger_print("[model] loading:", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation=getattr(args, "attn_impl", "flash_attention_2"),
    )
    model.resize_token_embeddings(len(tokenizer))

    # ---- PEFT / LoRA
    if getattr(args, "peft", False):
        lora_config = LoraConfig(
            r=int(args.r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=getattr(args, "target_modules", ["q_proj", "v_proj"]),
        )
        model = get_peft_model(model, lora_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # ---- Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=getattr(args, "betas", (0.9, 0.999)),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )

    # ---- Scheduler steps: per your convention (floor, no tail flush)
    iters = int(args.epochs)
    total_steps = _compute_total_steps(iters=iters, grad_accum_steps=grad_accum_steps)
    warmup_steps = _compute_warmup_steps(
        total_steps=total_steps,
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.06)),
    )
    min_lr_ratio = float(getattr(args, "min_lr_ratio", 0.1))
    print(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}, min_lr_ratio: {min_lr_ratio}")
    
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # ---- Prepare (DDP / fp16/bf16)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # ---- LR sanity info
    _log_lr_sanity(
        accelerator=accelerator,
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=float(args.lr),
        min_lr_ratio=min_lr_ratio,
    )

    # ---- W&B
    if getattr(args, "wandb", False) and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # =========================
    # Training Loop
    # =========================
    model.train()
    pbar = trange(iters, desc="Iteration", disable=not accelerator.is_main_process)

    best_val_loss = float("inf")
    opt_step = 0  # true optimizer steps (only increments when sync_gradients=True)

    for iter_num in pbar:
        # Deterministic per-iter seed across ranks
        seed_for_this_iter = int(args.seed) + iter_num * 100003 + accelerator.process_index
        set_seed(seed_for_this_iter, device_specific=True)

        # ---- batch
        batch = dataset.get_batch("train", args.batch_size)
        X, Y, gradient_mask, attention_mask, batch_cand_ids = _to_device(batch, accelerator.device)

        with accelerator.accumulate(model):
            loss = get_loss_variable_candidates(
                model,
                X,
                Y,
                gradient_mask,
                attention_mask,
                batch_cand_ids=batch_cand_ids,
                device=accelerator.device,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # grad clip
                grad_clip = float(getattr(args, "grad_clip", 0.0))
                if grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                opt_step += 1

                mean_loss = accelerator.reduce(loss.detach(), reduction="mean").item()

                # logging
                if accelerator.is_main_process:
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({"loss": mean_loss, "opt_step": opt_step, "lr": f"{lr:.2e}"})

                    log_obj = {
                        "iter": iter_num,
                        "opt_step": opt_step,
                        "lr": float(lr),
                        "train/loss_step": float(mean_loss),
                    }
                    print(log_obj)
                    if getattr(args, "wandb", False):
                        wandb.log(log_obj)

        # ---- Debug (optional)
        if getattr(args, "debug", False) and accelerator.is_main_process:
            with torch.no_grad():
                active = gradient_mask.sum().item()
                print(f"[iter {iter_num}] masked_positions={active}")
                if active > 0:
                    b, t = (gradient_mask > 0).nonzero(as_tuple=False)[0].tolist()
                    target_id = Y[b, t].item()
                    print("target(after <Answer>):", tokenizer.decode([target_id]))
                    logits_bt = model(input_ids=X[b:b+1], attention_mask=attention_mask[b:b+1]).logits[0, t]
                    topk = torch.topk(logits_bt, 5)
                    ids = topk.indices.tolist()
                    probs = torch.softmax(topk.values, dim=0).tolist()
                    print("top-5 preds:", [(tokenizer.decode([i]), p) for i, p in zip(ids, probs)])

        # =========================
        # Evaluation (main process only)
        # =========================
        if (iter_num % int(args.eval_interval) == 0) or (iter_num == iters - 1):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"[iter {iter_num}] Evaluation")
                model.eval()
                with torch.no_grad():
                    core_model = accelerator.unwrap_model(model)

                    train_loss, train_gen_acc, train_mc_acc = estimate_loss_and_acc_variable(
                        core_model,
                        dataset,
                        split="train",
                        device=accelerator.device,
                        batch_size=args.batch_size,
                        tokenizer=tokenizer,
                    )
                    val_loss, val_gen_acc, val_mc_acc = estimate_loss_and_acc_variable(
                        core_model,
                        dataset,
                        split="val",
                        device=accelerator.device,
                        batch_size=args.batch_size,
                        tokenizer=tokenizer,
                    )
                    test_loss, test_gen_acc, test_mc_acc = estimate_loss_and_acc_variable(
                        core_model,
                        dataset,
                        split="test",
                        device=accelerator.device,
                        batch_size=args.batch_size,
                        tokenizer=tokenizer,
                    )

                print(f"[iter {iter_num}] train={train_loss:.4f} val={val_loss:.4f} test={test_loss:.4f}")
                print(f"[iter {iter_num}] train_gen_acc={train_gen_acc:.4f} val_gen_acc={val_gen_acc:.4f} test_gen_acc={test_gen_acc:.4f}")
                print(f"[iter {iter_num}] train_mc_acc={train_mc_acc:.4f} val_mc_acc={val_mc_acc:.4f} test_mc_acc={test_mc_acc:.4f}")

                if getattr(args, "wandb", False):
                    # pull current lr safely (defined even if no step yet)
                    cur_lr = float(optimizer.param_groups[0]["lr"])
                    wandb.log({
                        "iter": iter_num,
                        "opt_step": opt_step,
                        "lr": cur_lr,

                        "train/eval_loss": float(train_loss),
                        "val/eval_loss": float(val_loss),
                        "test/eval_loss": float(test_loss),

                        "train/accuracy": float(train_mc_acc),
                        "val/accuracy": float(val_mc_acc),
                        "test/accuracy": float(test_mc_acc),

                        "train/gen_accuracy": float(train_gen_acc),
                        "val/gen_accuracy": float(val_gen_acc),
                        "test/gen_accuracy": float(test_gen_acc),
                    })

                # Save best
                if val_loss < best_val_loss and getattr(args, "save_dir", None):
                    best_val_loss = float(val_loss)
                    save_dir = args.save_dir
                    os.makedirs(save_dir, exist_ok=True)

                    core_model = accelerator.unwrap_model(model)
                    core_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"✓ Saved best checkpoint to {save_dir} (val={val_loss:.4f})")

                model.train()
            accelerator.wait_for_everyone()

    # Final sanity check
    if accelerator.is_main_process:
        logger_print(f"[done] opt_step={opt_step}  expected_total_steps={total_steps}  (iters={iters}, accum={grad_accum_steps})")
