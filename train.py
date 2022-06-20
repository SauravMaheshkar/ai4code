"""Training Code"""
import argparse
import gc
import logging
import os
from pathlib import Path

import torch
import transformers
from rich import print
from rich.logging import RichHandler
from transformers import get_linear_schedule_with_warmup

import wandb
from src.io.data import preprocess_fn, read_processed_data, serialize_dataframes
from src.io.dataset import get_dataloader
from src.nn.engine import train_fn, validation_fn
from src.nn.model import MarkdownModel
from src.nn.optimizers import fetch_optimizer
from src.utils import set_seed

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
transformers.logging.set_verbosity_error()

if not os.path.exists("data/processed"):
    os.mkdir("data/processed")
if not os.path.exists("models"):
    os.mkdir("models")

parser = argparse.ArgumentParser(description="Process some arguments")
parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
parser.add_argument(
    "--train_mark_path", type=str, default="data/processed/train_mark.csv"
)
parser.add_argument(
    "--train_features_path", type=str, default="data/processed/train_fts.json"
)
parser.add_argument("--val_mark_path", type=str, default="data/processed/val_mark.csv")
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
if not os.path.exists("models"):
    os.mkdir("models")
data_dir = Path("data/")

if __name__ == "__main__":

    # Miscellaneous
    set_seed(seed=42)

    # Initialize a Weights & Biases Run
    wandb.init(
        project="ai4code",
        job_type="train",
        group=str(args.model_name_or_path),
    )

    # Pre-Process Data
    logger.info("Pre-Process Data")
    train_df, val_df = preprocess_fn(
        data_dir=Path("data/"), test_size=0.1, random_seed=42, logger=logger
    )

    # Serialize Processed Data to disk
    logger.info("Serializing Processed Data to disk")
    serialize_dataframes(train_df, val_df, logger=logger)

    # Read Processed Data
    logger.info("Reading Processed Data")
    (
        train_df_mark,
        train_fts,
        val_df_mark,
        val_fts,
        val_df,
        df_orders,
    ) = read_processed_data(args, data_dir)

    # Get Dataloaders
    logger.info("Fetching Dataloaders")
    train_loader, val_loader = get_dataloader(
        train_df_mark, val_df_mark, train_fts, val_fts, args, logger=logger
    )

    # Create Model
    logger.info("Creating Model")
    model = MarkdownModel(args.model_name_or_path)
    model = model.cuda()

    # Get Number of Parameters
    trainable_model_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model Parameters:{trainable_model_params}")
    wandb.run.summary["Model Parameters"] = trainable_model_params  # type: ignore

    # Create Optimizer
    logger.info("Creating Optimizer")
    optimizer = fetch_optimizer(model=model, weight_decay=args.weight_decay)

    num_train_optimization_steps = int(
        args.epochs * len(train_loader) / args.accumulation_steps  #  type: ignore
    )

    # Create Scheduler
    logger.info("Creating Scheduler")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )

    criterion = torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_fn(
            model,
            train_loader,
            criterion,
            scaler,
            optimizer,
            accumulation_steps=args.accumulation_steps,
        )
        validation_fn(model, val_loader, val_df, df_orders)
        wandb.log({"Learning Rate": scheduler.get_last_lr()})
        scheduler.step()
        _ = gc.collect()

    wandb.finish()
