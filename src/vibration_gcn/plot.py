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
from numpy import arange
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
        cmap="Blues",
        interpolation="none",
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
