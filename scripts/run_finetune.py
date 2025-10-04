import argparse 
from pathlib import Path 
import yaml 
from datetime import datetime
from finetune import train_accelerate
from finetune.args import parse_arguments

def main():
    args = parse_arguments()
    model_config_path = Path(f"./scripts/model_args/{args.model_name}.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    for key, value in model_config.items():
        setattr(args, key, value)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.wandb_project = f"{args.dataset}"
    args.wandb_run_name = f"{args.model_name}_{args.split}_neighbor{args.neighbor}_{timestamp}"
    print("wandb project:", args.wandb_project)
    print("wandb run name:", args.wandb_run_name)
    if args.save_dir is not None:
        
        args.save_dir = Path(args.save_dir) / args.dataset / args.split / f'neighbor_{args.neighbor}' / args.model_name / timestamp
        args.save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.root_data_dir) / args.dataset / "processed" 
    args.data_dir = data_dir / f"dataset_{args.split}_neighbor{args.neighbor}.json"

    print("data_dir:", args.data_dir)
    print("save_dir:", args.save_dir)
    # input()
    train_accelerate(args)
    # train_accelerate(args)
    

if __name__ == "__main__":
    main()