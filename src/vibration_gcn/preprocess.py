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
from numpy import absolute, double, fromiter, pad, std, amax, sqrt, ndenumerate
from numpy.typing import NDArray
from scipy.stats import skew, kurtosis
from sklearn.ensemble import IsolationForest


def group_signals(raw: NDArray, n: int = 500) -> NDArray:
    num_rows: int = ceil(raw.shape[0] / n)
    return pad(raw, (0, num_rows * n - raw.shape[0]), mode="edge").reshape(
        (num_rows, n)
    )


def calc_feature(raw: NDArray) -> NDArray:
    n: int = raw.shape[0]

    features: dict[str, double] = {}
    features["standard deviation"] = std(raw)
    features["peak"] = amax(absolute(raw))
    features["skewness"] = skew(raw)
    features["kurtosis"] = kurtosis(raw)
    features["root mean square"] = sqrt((raw**2).sum() / n)
    features["crest factor"] = features["peak"] / features["root mean square"]
    features["square root amplitude"] = (sqrt(absolute(raw)).sum() / n) ** 2
    features["shape factor"] = features["root mean square"] * n / absolute(raw).sum()
    features["impulse factor"] = features["peak"] * n / absolute(raw).sum()

    return fromiter(features.values(), dtype=double)


def find_clean_indexes(datas: NDArray, **kwargs) -> list[int]:
    forest: IsolationForest = IsolationForest(**kwargs)
    outliers: NDArray = forest.fit_predict(datas)
    return [i[0] for i, x in ndenumerate(outliers) if x != -1]
