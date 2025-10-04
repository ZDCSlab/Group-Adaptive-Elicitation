import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default='data',
        help="data directory",
    )  
    parser.add_argument(
        "--identity",
        type=str,
        default='identity',
        help="identity directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ces", "ces_golden",  "ces_golden_demo"],
        default='ces',
        help="dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["20-24", "22-24", "20-22", "20", "22", "24"],
        default='20',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gpt2", "Llama-3.2-1B", "Llama-3.1-8B"],
        default='gpt2',
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        "--neighbor",
        type=str,
        choices=['0', '1'],
        default='0',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['all', 'no_self_neighbor'],
        default='all',
    )
    return parser.parse_args()