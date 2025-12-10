# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FEMNIST horizontal hierarchical FL middle-level aggregator for PyTorch (LIFL)."""

import logging

from flame.config import Config
from flame.mode.horizontal.lifl_coord_syncfl.middle_aggregator import MiddleAggregator

# the following import makes sure PyTorch is registered as the framework
import torch  # noqa: F401
import torchvision.models as tormodels  # noqa: F401

logger = logging.getLogger(__name__)


class TorchFemnistMiddleAggregator(MiddleAggregator):
    """Torch FEMNIST Middle Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        # Do NOT call super().__init__ – base class wires itself in internal_init tasklet.
        self.config = config

        # Output logs (per-middle-agg log file)
        log_file = f"/mydata/image_cls-mid_aggregator-{config.task_id}.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

        logger.info(f"Middle aggregator initialized with task_id={config.task_id}")

    # For this experiment we don't need middle-level model or data logic.
    # LIFL's MiddleAggregator will:
    #   - receive deltas from leaf/trainer side
    #   - aggregate them
    #   - send up to top aggregator
    #
    # So we keep these methods minimal / no-op.

    def initialize(self):
        """Initialize role (no-op for now)."""
        logger.info("Middle aggregator initialize() called (no-op)")

    def load_data(self) -> None:
        """Load a test dataset (not used at middle level)."""
        logger.info("Middle aggregator load_data() called (no-op)")

    def train(self) -> None:
        """Train a model (aggregation only happens in base class)."""
        logger.info("Middle aggregator train() called (no-op)")

    def evaluate(self) -> None:
        """Evaluate (not used at middle level)."""
        logger.info("Middle aggregator evaluate() called (no-op)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = TorchFemnistMiddleAggregator(config)
    a.compose()
    a.run()
