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

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.pyplot import subplots
from numpy import arange, array, nditer
from numpy.typing import NDArray


def linechart(y: NDArray, title: str) -> None:
    fig: Figure
    ax: Axes
    fig, ax = subplots()

    ax.plot(y)
    ax.set_title(title)
    fig.tight_layout()


def heatmap(
    data: NDArray,
    title: str,
    x_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
) -> None:
    fig: Figure
    ax: Axes
    fig, ax = subplots()

    im: AxesImage = ax.imshow(
        data,
        cmap="hot",
        interpolation="nearest",
        aspect="auto",
    )
    fig.colorbar(im)

    if x_labels is not None:
        ax.set_xticks(arange(len(x_labels)), labels=x_labels)
        for t in ax.get_xticklabels():
            t.set(rotation=45, horizontalalignment="right", rotation_mode="anchor")
    if y_labels is not None:
        ax.set_yticks(arange(len(y_labels)), labels=y_labels)

    ax.set_title(title)
    fig.tight_layout()


def confusion_matrix(confusion_data: dict[str, float], title: str) -> None:
    fig: Figure
    ax: Axes
    fig, ax = subplots()

    matrix: NDArray = array(
        [
            [confusion_data["tp"], confusion_data["fn"]],
            [confusion_data["fp"], confusion_data["tn"]],
        ]
    )

    im: AxesImage = ax.imshow(matrix, interpolation="nearest", cmap="hot")
    fig.colorbar(im)
    ax.set(
        xticks=arange(matrix.shape[1]),
        yticks=arange(matrix.shape[0]),
        xticklabels=["Positive", "Negative"],
        yticklabels=["Positive", "Negative"],
        title=title,
        xlabel="Predicted",
        ylabel="Actual",
    )

    for t in ax.get_xticklabels():
        t.set(rotation=45, horizontalalignment="right", rotation_mode="anchor")
    with nditer(matrix, flags=["multi_index"]) as it:
        for x in it:
            i: int
            j: int
            i, j = it.multi_index
            ax.text(
                j,
                i,
                format(x, "d"),
                ha="center",
                va="center",
                color="white",
                backgroundcolor="black",
            )

    fig.tight_layout()
