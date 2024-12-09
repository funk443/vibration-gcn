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
from numpy import pad
from numpy.typing import NDArray


def group_signals(raw: NDArray, n: int = 500) -> NDArray:
    num_rows: int = ceil(raw.shape[0] / n)
    return pad(raw, (0, num_rows * n - raw.shape[0]), mode="edge").reshape(
        (num_rows, n)
    )
