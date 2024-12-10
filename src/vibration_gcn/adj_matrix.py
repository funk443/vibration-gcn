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

from numpy import fill_diagonal
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


def knn(features: NDArray) -> NDArray:
    adj_matrix: NDArray = (
        NearestNeighbors().fit(features).kneighbors_graph(features).toarray()
    )
    fill_diagonal(adj_matrix, 0)

    return adj_matrix
