"""Data Utilities"""
import json
from logging import Logger
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from rich.progress import track
from sklearn.model_selection import GroupShuffleSplit

from src.io.data_utils import get_features, get_ranks, read_notebook

ranks: Dict = {}


def preprocess_fn(
    data_dir: Path, logger: Logger, test_size: float = 0.1, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """preprocess_fn PreProcess Data

    :param data_dir: Path to the raw data
    :type data_dir: Path
    :param logger: Logger Instance
    :type logger: Logger
    :param test_size: Split ratio, defaults to 0.1
    :type test_size: float, optional
    :param random_seed: Random Seed for Splitting, defaults to 42
    :type random_seed: int, optional
    :return: Training and Validation dataframe
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    paths_train = list((data_dir / "train").glob("*.json"))

    # Read Notebooks
    logger.info("Reading Notebooks")
    notebooks_train = [
        read_notebook(path)
        for path in track(paths_train, description="Reading Notebooks ...")
    ]

    # Create a single Dataframe
    logger.info("Concatenating into a single dataframe")
    dataframe = (
        pd.concat(notebooks_train)
        .set_index("id", append=True)
        .swaplevel()
        .sort_index(level="id", sort_remaining=False)
    )

    logger.info("Reading Orders dataframe")
    df_orders = pd.read_csv(
        data_dir / "train_orders.csv",
        index_col="id",
        squeeze=True,
    ).str.split()  # Split the string representation of cell_ids into a list

    logger.info("Process Orders dataframe")
    df_orders_ = df_orders.to_frame().join(
        dataframe.reset_index("cell_id").groupby("id")["cell_id"].apply(list),
        how="right",
    )

    logger.info("Get Ranks")
    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {"cell_id": cell_id, "rank": get_ranks(cell_order, cell_id)}

    logger.info("Create Ranks dataframe")
    df_ranks = (
        pd.DataFrame.from_dict(ranks, orient="index")
        .rename_axis("id")
        .apply(pd.Series.explode)
        .set_index("cell_id", append=True)
    )

    logger.info("Reading Ancestors dataframe")
    df_ancestors = pd.read_csv(data_dir / "train_ancestors.csv", index_col="id")
    dataframe = (
        dataframe.reset_index()
        .merge(df_ranks, on=["id", "cell_id"])
        .merge(df_ancestors, on=["id"])
    )
    dataframe["pct_rank"] = dataframe["rank"] / dataframe.groupby("id")[
        "cell_id"
    ].transform("count")

    # Split into Train and Validation
    logger.info("Split into Train and Validation")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    train_ind, val_ind = next(
        splitter.split(dataframe, groups=dataframe["ancestor_id"])
    )
    train_df = dataframe.loc[train_ind].reset_index(drop=True)
    val_df = dataframe.loc[val_ind].reset_index(drop=True)

    return train_df, val_df


def serialize_dataframes(train_df: pd.DataFrame, val_df: pd.DataFrame, logger: Logger):
    """
    Serialize and save dataframes to disk

    :param train_df: Training Dataframe
    :type train_df: pd.DataFrame
    :param val_df: Validation Dataframe
    :type val_df: pd.DataFrame
    :param logger: Logger Instance
    :type logger: Logger
    """
    # Training Dataframes
    logger.info("Saving Train DataFrame")
    train_df.to_csv("data/processed/train.csv", index=False)

    train_df_mark: pd.DataFrame = train_df[
        train_df["cell_type"] == "markdown"
    ].reset_index(drop=True)
    logger.info("Saving Markdown Train DataFrame")
    train_df_mark.to_csv("data/processed/train_mark.csv", index=False)

    logger.info("Getting Training Features")
    train_fts = get_features(train_df)
    json.dump(train_fts, open("data/processed/train_fts.json", "wt"))

    # Validation Dataframes
    logger.info("Saving Validation DataFrame")
    val_df.to_csv("data/processed/val.csv", index=False)

    val_df_mark: pd.DataFrame = val_df[val_df["cell_type"] == "markdown"].reset_index(
        drop=True
    )
    logger.info("Saving Markdown Validation DataFrame")
    val_df_mark.to_csv("data/processed/val_mark.csv", index=False)

    logger.info("Getting Validation Features")
    val_fts = get_features(val_df)
    json.dump(val_fts, open("data/processed/val_fts.json", "wt"))


def read_processed_data(args, data_dir):
    """read_processed_data Read Pre-Processed Data from disk

    :param args: Parsed arguments
    :param data_dir: Raw Data Path
    :type data_dir: str
    :return: Training and Validation Data
    :rtype: Tuple
    """
    train_df_mark = (
        pd.read_csv(args.train_mark_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    train_fts = json.load(open(args.train_features_path))
    val_df_mark = (
        pd.read_csv(args.val_mark_path)
        .drop("parent_id", axis=1)
        .dropna()
        .reset_index(drop=True)
    )
    val_fts = json.load(open(args.val_features_path))
    val_df = pd.read_csv(args.val_path)

    df_orders = pd.read_csv(
        data_dir / "train_orders.csv",
        index_col="id",
        squeeze=True,
    ).str.split()

    return train_df_mark, train_fts, val_df_mark, val_fts, val_df, df_orders
