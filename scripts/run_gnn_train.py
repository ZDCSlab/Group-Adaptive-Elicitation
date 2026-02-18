import argparse
import torch
import os

from src.gnn.model import HGNNModel
from src.gnn.dataset import QAGraph, EpochMasker, load_split_from_json, load_qa_graph
from src.gnn.utils import load_config, parse_overrides, deep_update
from src.gnn.train import run_training_loop


# --- Utils ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument(
        "--set", nargs="*", default=[],
        help='Optional overrides like: --set train.epochs=100 optim.lr=0.001 data.demo_cols="[a,b]"'
    )
    return p.parse_args()

def pick_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    overrides = parse_overrides(args.set)
    cfg = deep_update(cfg, overrides)

    checkpoint_dir = cfg["checkpoint"]["ckpt_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = cfg["logs"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{cfg["checkpoint"]["ckpt_prefix"]}.txt')
    log_f = open(log_path, "w", encoding="utf-8")

    # Seed
    torch.manual_seed(int(cfg["train"]["seed"]))

    # Load Graph
    graph = load_qa_graph(cfg)

    # Split + Epoch Masking
    split_path = cfg["split"].get("path", "") if isinstance(cfg.get("split"), dict) else ""
    split = load_split_from_json(split_path, graph.edges_u_to_qopt, graph.uid2idx)
    masker = EpochMasker(split, half_ratio=float(cfg["split"]["mask_ratio"]))  


    # Model/Optimizer
    device = pick_device(cfg["train"].get("device", "auto"))
    model = HGNNModel(
        graph.data,
        d_in=int(cfg["model"]["hidden"]),
        d_h=int(cfg["model"]["hidden"]),
        layers=int(cfg["model"]["layers"]),
        dropout=float(cfg["model"]["dropout"])
    ).to(device)

    run_training_loop(model, graph, masker, cfg, device, log_f)


if __name__ == "__main__":
    main()
