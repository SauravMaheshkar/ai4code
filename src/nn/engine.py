"""Training Utilites"""
import gc
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from rich import print
from rich.progress import track
from torch import nn

import wandb
from src.nn.metrics import kendall_tau


def read_data(data) -> Tuple:
    """Read and move batch of Data"""
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validation_fn(
    model: nn.Module,
    val_loader: Iterable,
    val_df: pd.DataFrame,
    df_orders: pd.DataFrame,
):
    """
    Run Validation

    :param model: Model
    :type model: nn.Module
    :param val_loader: Validation Dataloader
    :type val_loader: Iterable
    :param val_df: Validation dataframe
    :type val_df: pd.DataFrame
    :param df_orders: Orders dataframe
    :type df_orders: pd.DataFrame
    """

    # Set Model into Evaluation Mode
    model.eval()

    # Miscellaneous Variables
    preds: List = []
    labels: List = []

    with torch.no_grad():
        for _, data in track(
            enumerate(val_loader),
            description="Running Validation ...",
            total=len(val_loader),  # type: ignore
        ):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    # y_val = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    val_df["pred"] = val_df.groupby(["id", "cell_type"])["rank"].rank(pct=True)
    val_df.loc[val_df["cell_type"] == "markdown", "pred"] = y_pred
    y_dummy = val_df.sort_values("pred").groupby("id")["cell_id"].apply(list)
    kendall_tau_score = kendall_tau(df_orders.loc[y_dummy.index], y_dummy)
    print("Preds score", kendall_tau_score)

    # Log Metrics
    wandb.log({"Valid/Kendall Tau Score": kendall_tau_score})

    torch.save(model.state_dict(), "./models/model.bin")


def train_fn(
    model: nn.Module,
    train_loader: Iterable,
    criterion: Callable,
    scaler,
    optimizer,
    accumulation_steps: int,
):
    """
    Run Training

    :param model: Model
    :type model: nn.Module
    :param train_loader: Training Dataloader
    :type train_loader: Iterable
    :param criterion: Loss Fn
    :type criterion: Callable
    :param scaler: torch amp Scaler
    :param optimizer: Pytorch Optimizer
    :param accumulation_steps: Gradient accumulation steps
    :type accumulation_steps: int
    """

    # Set Model into Training Mode
    model.train()

    # Miscellaneous Variables
    loss_list: List = []
    preds: List = []
    labels: List = []

    for idx, data in track(
        enumerate(train_loader),
        description="Running Training ...",
        total=len(train_loader),  # type: ignore
    ):
        inputs, target = read_data(data)

        with torch.cuda.amp.autocast():
            pred = model(*inputs)
            loss = criterion(pred, target)
        scaler.scale(loss).backward()
        if idx % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_list.append(loss.detach().cpu().item())
        preds.append(pred.detach().cpu().numpy().ravel())
        labels.append(target.detach().cpu().numpy().ravel())
        avg_loss = np.round(np.mean(loss_list), 4)

        # Log Metrics
        wandb.log({"Train/Loss": loss.detach().cpu().item()})
        wandb.log({"Train/Avg Loss": avg_loss})

        torch.cuda.empty_cache()
        _ = gc.collect()
