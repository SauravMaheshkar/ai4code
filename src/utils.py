"""Utility Functions"""
import argparse
import os
import random

import numpy as np
import torch
from rich import print


def set_seed(seed: int = 42) -> None:
    """
    Sets Random Seed
    :param seed: Seed, defaults to 42
    :type seed: int, optional
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def parse_args():
    """Argument Parser"""
    parser = argparse.ArgumentParser(description="Process some arguments")
    parser.add_argument(
        "--model_name_or_path", type=str, default="microsoft/codebert-base"
    )
    parser.add_argument(
        "--train_mark_path", type=str, default="data/processed/train_mark.csv"
    )
    parser.add_argument(
        "--train_features_path", type=str, default="data/processed/train_fts.json"
    )
    parser.add_argument(
        "--val_mark_path", type=str, default="data/processed/val_mark.csv"
    )
    parser.add_argument(
        "--val_features_path", type=str, default="data/processed/val_fts.json"
    )
    parser.add_argument("--val_path", type=str, default="data/processed/val.csv")

    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--md_max_len", type=int, default=64)
    parser.add_argument("--total_max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=8)

    args = parser.parse_args()

    return args
