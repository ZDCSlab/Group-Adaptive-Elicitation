import torch
import os
from dataset import build_loaders_for_epoch



def run_training_loop(model, graph, masker, cfg, device, log_f):
    """Run optimizer setup and full train/val/test loop; save best checkpoint by val loss."""
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"])
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])

    for epoch in range(1, epochs + 1):
        spec = masker.epoch_spec(epoch)
        packs = build_loaders_for_epoch(graph, spec, batch_size=batch_size)

        # Train Split
        model.train()
        train_mp = packs["train"]["data_mp"].to(device)
        tot_loss = tot_acc = 0.0
        n_batches = 0
        for batch in packs["train"]["loader"]:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad()
            loss, acc, _ = model(train_mp, batch)
            loss.backward()
            opt.step()
            tot_loss += float(loss)
            tot_acc += float(acc)
            n_batches += 1
        tr_loss = tot_loss / max(n_batches, 1)
        tr_acc = tot_acc / max(n_batches, 1)

        # Val Split
        model.eval()
        val_mp = packs["val"]["data_mp"].to(device)
        v_tot_loss = v_tot_acc = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in packs["val"]["loader"]:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                v_loss, v_acc, _ = model(val_mp, batch)
                v_tot_loss += float(v_loss)
                v_tot_acc += float(v_acc)
                v_n += 1
        va_loss = v_tot_loss / max(v_n, 1)
        va_acc = v_tot_acc / max(v_n, 1)

        # Test Split
        model.eval()
        test_mp = packs["test"]["data_mp"].to(device)
        t_tot_loss = t_tot_acc = 0.0
        t_n = 0
        with torch.no_grad():
            for batch in packs["test"]["loader"]:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                t_loss, t_acc, _ = model(test_mp, batch)
                t_tot_loss += float(t_loss)
                t_tot_acc += float(t_acc)
                t_n += 1
        te_loss = t_tot_loss / max(t_n, 1)
        te_acc = t_tot_acc / max(t_n, 1)

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt_path = os.path.join(cfg["checkpoint"]["ckpt_dir"], f'{cfg["checkpoint"]["ckpt_prefix"]}_best.pt')
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": opt.state_dict(),
                    "best_val_loss": best_val,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"[CKPT] Saved best checkpoint at epoch {epoch} -> {ckpt_path}")

        msg = (f"Epoch {epoch:02d} | "
               f"train_sup={packs['train']['num_sup']} mp={packs['train']['num_mp']} "
               f"| train CE={tr_loss:.4f} acc={tr_acc:.3f} "
               f"| val CE={va_loss:.4f} acc={va_acc:.3f} "
               f"| test CE={te_loss:.4f} acc={te_acc:.3f}")
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    return best_epoch, best_val, best_state

