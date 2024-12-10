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


from torch import Tensor
from torch.nn import Module, ModuleList, Mish, LogSoftmax, CrossEntropyLoss, Dropout
from torch.optim import Optimizer
from tqdm import trange
from ranger import Ranger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(Module):
    def __init__(self, data: Data) -> None:
        super().__init__()
        self.data: Data = data
        self.activation: Module = Mish()
        self.final_activation: Module = LogSoftmax(dim=1)
        self.dropout: Module = Dropout(p=0.1)

        self.layers: ModuleList = ModuleList(
            (
                GCNConv(self.data.num_node_features, 8),
                GCNConv(8, 16),
                GCNConv(16, 8),
                GCNConv(8, 2),
            )
        )
        self.layer_count: int = len(self.layers)

    def forward(self, data: Data) -> Tensor:
        x: Tensor = data.x
        edge_index: Tensor = data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.layer_count - 1:
                x = self.activation(x)
                x = self.dropout(x)

        return self.final_activation(x)

    def go_training(self, n: int = 350, data: Data | None = None) -> None:
        self.train()
        if data is None:
            data = self.data

        optimizer: Optimizer = Ranger(self.parameters())
        loss_fn: Module = CrossEntropyLoss()
        train_mask: Tensor = data.train_mask

        for epoch in trange(n):
            optimizer.zero_grad()
            out: Tensor = self(data)
            loss: Tensor = loss_fn(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

    def go_testing(self, data: Data | None = None) -> dict[str, float]:
        self.eval()
        if data is None:
            data = self.data

        predictions: Tensor = self(data).argmax(dim=1)
        test_mask: Tensor = data.test_mask
        y_true: Tensor = data.y[test_mask]
        y_pred: Tensor = predictions[test_mask]

        confusion_data: dict[str, float] = {
            "tp": ((y_pred == 1) & (y_true == 1)).sum().item(),
            "tn": ((y_pred != 1) & (y_true != 1)).sum().item(),
            "fp": ((y_pred == 1) & (y_true != 1)).sum().item(),
            "fn": ((y_pred != 1) & (y_true == 1)).sum().item(),
        }
        confusion_data["accuracy"] = (confusion_data["tp"] + confusion_data["tn"]) / (
            confusion_data["tp"]
            + confusion_data["tn"]
            + confusion_data["fp"]
            + confusion_data["fn"]
        )
        confusion_data["precision"] = confusion_data["tp"] / (
            confusion_data["tp"] + confusion_data["fp"]
        )
        confusion_data["recall"] = confusion_data["tp"] / (
            confusion_data["tp"] + confusion_data["fn"]
        )
        confusion_data["f1-score"] = (
            2 * confusion_data["precision"] * confusion_data["recall"]
        ) / (confusion_data["precision"] + confusion_data["recall"])

        return confusion_data
