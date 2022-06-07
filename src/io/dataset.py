"""Custom Dataset Class and Dataloader utilities """
from logging import Logger
from typing import Any, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):
    """Custom Dataset Class"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name_or_path: str,
        total_max_len: int,
        md_max_len: int,
        fts: Any,
    ):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_md"]
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs["input_ids"]
        for temp in code_inputs["input_ids"]:
            ids.extend(temp[:-1])
        ids = ids[: self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [
                self.tokenizer.pad_token_id,
            ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs["attention_mask"]
        for temp in code_inputs["attention_mask"]:
            mask.extend(temp[:-1])
        mask = mask[: self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [
                self.tokenizer.pad_token_id,
            ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self) -> int:
        return self.dataframe.shape[0]


def get_dataloader(
    train_df_mark: pd.DataFrame,
    val_df_mark: pd.DataFrame,
    train_fts,
    val_fts,
    args,
    logger: Logger,
) -> Tuple[Iterable, Iterable]:
    """
    Fetch Dataloader

    :param train_df_mark: Training MD Dataframe
    :type train_df_mark: pd.DataFrame
    :param val_df_mark: Validation MD Dataframe
    :type val_df_mark: pd.DataFrame
    :param train_fts: Training Features
    :type train_fts: Any
    :param val_fts: Validation Features
    :type val_fts: Any
    :param args: Arguments (from argparse)
    :param logger: Logger Instance
    :type logger: Logger
    :return: Training and Validation Dataloader
    :rtype: Tuple[Iterable, Iterable]
    """
    logger.info("Creating Train Dataset")
    train_ds = MarkdownDataset(
        train_df_mark,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len,
        fts=train_fts,
    )

    logger.info("Creating Validation Dataset")
    val_ds = MarkdownDataset(
        val_df_mark,
        model_name_or_path=args.model_name_or_path,
        md_max_len=args.md_max_len,
        total_max_len=args.total_max_len,
        fts=val_fts,
    )

    logger.info("Creating Train Dataloader")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=True,
    )

    logger.info("Creating Validation Dataloader")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader
