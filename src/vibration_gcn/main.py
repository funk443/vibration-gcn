# vibration-gcn: Classify vibration signals with GCN.
# Copyright (C) 2024  CToID <funk443@yahoo.com.tw>
#
# This file is part of vibration-gcn.
#
# vibration-gcn is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# vibration-gcn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vibration-gcn.  If not, see <https://www.gnu.org/licenses/>.

from math import ceil
from random import shuffle

from numpy import apply_along_axis, concatenate, double
from numpy.typing import NDArray
from matplotlib.pyplot import show
from torch import Tensor, cat, logical_not, tensor
import torch

from . import plot
from . import adj_matrix
from .input import read_file_to_array
from .preprocess import find_clean_indexes, group_signals, calc_feature


def make_mask(n: int, percentage: float, do_shuffle: bool = True) -> Tensor:
    assert percentage <= 1.0

    true_amount: int = ceil(n * percentage)
    temp: list[bool] = [True] * true_amount + [False] * (n - true_amount)
    if do_shuffle:
        shuffle(temp)

    return tensor(temp, dtype=torch.bool)


def main(normal_file_path: str, abnormal_file_path: str) -> None:
    normal_raw: NDArray = read_file_to_array(normal_file_path)
    abnormal_raw: NDArray = read_file_to_array(abnormal_file_path)

    # plot.linechart(normal_raw, "Raw signal, normal")
    # plot.linechart(abnormal_raw, "Raw signal, abnormal")

    normal_splited: NDArray = group_signals(normal_raw)
    abnormal_splited: NDArray = group_signals(abnormal_raw)
    dirty_features: NDArray = apply_along_axis(
        func1d=calc_feature,
        axis=1,
        arr=concatenate((normal_splited, abnormal_splited), dtype=double),
    )

    # Find all non-outliers indexes.
    clean_indexes: list[int] = find_clean_indexes(dirty_features, contamination=0.02)

    # Keep only non-outliers in the data.
    features: NDArray = dirty_features[clean_indexes]
    ground_truth_label: Tensor = tensor(
        [0] * normal_splited.shape[0] + [1] * abnormal_splited.shape[0],
        dtype=torch.uint8,
    )[clean_indexes]
    train_mask: Tensor = cat(
        (
            make_mask(n=normal_splited.shape[0], percentage=0.75),
            make_mask(n=abnormal_splited.shape[0], percentage=0.75),
        )
    )[clean_indexes]
    test_mask: Tensor = logical_not(train_mask)

    # plot.heatmap(
    #     features,
    #     title="Feature matrix",
    #     x_labels=[
    #         "Standard deviation",
    #         "Peak",
    #         "Skewness",
    #         "Kurtosis",
    #         "Root mean square",
    #         "Crest factor",
    #         "Square root amplitude",
    #         "Shape factor",
    #         "Impulse factor",
    #     ],
    # )

    adj_matrix_knn: NDArray = adj_matrix.knn(features)

    # plot.heatmap(adj_matrix_knn, "Adjacency matrix, KNN")
    # show()
