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
        "--option_dict_path",
        type=str,
        default='codebook.jsonl',
        help="codebook file path",
    )  
  
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ces", "opinionqa", "twin"],
        default='ces',
        help="dataset name",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Llama-3.2-1B", "Llama-3.1-8B"],
        default='Llama-3.1-8B',
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
    return parser.parse_args()