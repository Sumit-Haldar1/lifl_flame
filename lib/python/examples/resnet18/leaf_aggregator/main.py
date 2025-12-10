# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FEMNIST horizontal hierarchical FL leaf level aggregator for PyTorch."""

import logging

from flame.config import Config
from flame.mode.horizontal.lifl_coord_syncfl.leaf_aggregator import LeafAggregator

import torch
import torchvision.models as tormodels

logger = logging.getLogger(__name__)


class PyTorchFemnistLeafAggregator(LeafAggregator):
    """PyTorch FEMNIST Leaf Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config

        # device + model will be initialized in initialize()
        self.device = None
        self.model = None

        # optional: per-leaf log file
        log_file = f"/mydata/image_cls-leaf_aggregator-{config.task_id}.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

    def initialize(self):
        """Initialize role.

        IMPORTANT: we must set self.model here so the FedAvg optimizer
        can detect that this is a PyTorch model.
        """
        # choose device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # use the same architecture as top / trainers (or compatible)
        # if your trainers use resnet152 and top uses resnet18, leaf can
        # use either as long as shapes match the weights coming in.
        self.model = tormodels.__dict__["resnet18"](num_classes=62).to(self.device)

        logger.info(
            f"Leaf aggregator initialized with model resnet18 on device {self.device}"
        )

    def load_data(self) -> None:
        """Load a test dataset (unused at leaf by default)."""
        # Leaf agg typically just aggregates parameters, so no data is needed.
        pass

    def train(self) -> None:
        """Train a model (usually not done at leaf level)."""
        # By default, leaf agg only aggregates and forwards updates.
        pass

    def evaluate(self) -> None:
        """Evaluate a model (optional)."""
        # You can add local evaluation here if you want per-leaf metrics.
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchFemnistLeafAggregator(config)
    a.compose()
    a.run()
