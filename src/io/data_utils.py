"""Data Processing Utilities"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from rich.progress import track


def clean_code(cell) -> str:
    """
    Cleans Code Snippet with correct newline character

    :return: Cleaned Code Cell
    :rtype: str
    """
    return str(cell).replace("\\n", "\n")


def get_ranks(base, derived):
    """Utility Function to get ranks"""
    return [base.index(d) for d in derived]


def read_notebook(path: Path) -> pd.DataFrame:
    """
    Read Notebook from path

    :param path: Path to the Notebook
    :type path: str
    :return: pandas dataframe of the notebook
    :rtype: pd.DataFrame
    """
    return (
        pd.read_json(path, dtype={"cell_type": "category", "source": "str"})
        .assign(id=path.stem)
        .rename_axis("cell_id")
    )


def sample_cells(cells, num: int) -> List:
    """
    Samples the given list with some global information

    :param cells: Cells Column of the dataframe
    :type cells: pd.Series
    :return: Cells sampled with global information
    :rtype: List
    """
    cells = [clean_code(cell) for cell in cells]
    if num >= len(cells):
        return [cell[:200] for cell in cells]
    else:
        results = []
        step = len(cells) / num
        idx = 0.0
        while int(np.round(idx)) < len(cells):
            results.append(cells[int(np.round(idx))])
            idx += step
        assert cells[0] in results
        if cells[-1] not in results:
            results[-1] = cells[-1]
        return results


def get_features(dataframe: pd.DataFrame) -> Dict:
    """
    Extract meta features such as total_code and total_md

    :param df: dataframe to extract from
    :type df: pd.DataFrame
    :return: a features dictionary
    :rtype: Dict
    """
    features: Dict = {}
    dataframe = dataframe.sort_values("rank").reset_index(drop=True)
    for idx, sub_df in track(
        dataframe.groupby("id"), description="Extracting features ..."
    ):
        features[idx] = {}
        total_md = sub_df[sub_df.cell_type == "markdown"].shape[0]
        code_sub_df = sub_df[sub_df.cell_type == "code"]
        total_code = code_sub_df.shape[0]
        codes = sample_cells(code_sub_df.source.values, 20)
        features[idx]["total_code"] = total_code
        features[idx]["total_md"] = total_md
        features[idx]["codes"] = codes
    return features
