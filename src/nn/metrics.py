"""Metric Utilities"""
from bisect import bisect
from typing import Iterable, List

import pandas as pd


def count_inversions(ranks: Iterable) -> int:
    """Count Inversions in ranks"""
    inversions = 0
    sorted_so_far: List = []
    for i, temp in enumerate(ranks):
        j = bisect(sorted_so_far, temp)
        inversions += i - j
        sorted_so_far.insert(j, temp)
    return inversions


def kendall_tau(ground_truth: pd.Series, predictions: pd.Series) -> float:
    """Compute Kenall Tau Score"""
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for truth, pred in zip(ground_truth, predictions):
        ranks = [
            truth.index(x) for x in pred
        ]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        quantity = len(truth)
        total_2max += quantity * (quantity - 1)
    return 1 - 4 * total_inversions / total_2max
