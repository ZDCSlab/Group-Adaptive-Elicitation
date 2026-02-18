import argparse 
from pathlib import Path 
import yaml 
from datetime import datetime
from src.meta_train.train import train_accelerate
from src.meta_train.args import parse_arguments


def main():
    args = parse_arguments()
    model_config_path = Path(f"scripts/args_llm/{args.model_name}.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    for key, value in model_config.items():
        setattr(args, key, value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.wandb_project = f"Graph-Adaptive-Elicitation"
    args.wandb_run_name = f"{args.model_name}_{args.dataset}_finetuned_{timestamp}"
    print("wandb project:", args.wandb_project)
    print("wandb run name:", args.wandb_run_name)
    if args.save_dir is not None:
        args.save_dir = Path(args.save_dir) / args.dataset / args.model_name / timestamp
        args.save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.root_data_dir) / args.dataset / "processed_data" 
    args.data_dir = data_dir

    print("data_dir:", args.data_dir)
    print("save_dir:", args.save_dir)
   
    train_accelerate(args)
   

if __name__ == "__main__":
    main()