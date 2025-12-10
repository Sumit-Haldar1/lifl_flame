# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FEMNIST horizontal FL trainer for PyTorch (LIFL)."""

import logging
import random
import time

import torch
import torch.optim as optim
import torchvision.models as tormodels

from flame.config import Config
from flame.mode.horizontal.lifl_coord_syncfl.trainer import Trainer
from flame.fedscale_utils.femnist import FEMNIST
from flame.fedscale_utils.utils_data import get_data_transform

logger = logging.getLogger(__name__)


def override(method):
    return method


class PyTorchFemnistTrainer(Trainer):
    """PyTorch FEMNIST Trainer (LIFL-coordinated)."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None

        self.device = None
        self.train_loader = None

        # Hyperparameters
        self.epochs = getattr(self.config.hyperparameters, "epochs", 1)
        self.batch_size = getattr(self.config.hyperparameters, "batchSize", 16)

        # FEMNIST paths & training control
        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/femnist/"
        self.meta_dir = "/mydata/flame_dataset/femnist/"
        self.partition_id = 1  # randomized in load_data()

        self.loss_squared = 0.0
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_decay = 0.2
        self.local_steps = 30

        # Timing
        self.load_data_delay = 0.0
        self.local_training_delay = 0.0

        # LIFL / BaseTrainer fields
        self._work_done = False
        self.fetch_success = False

        log_file = f"/mydata/image_cls-trainer-{config.task_id}.log"
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

    def initialize(self) -> None:
        """Initialize role (model + device)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tormodels.__dict__["resnet18"](num_classes=62).to(self.device)
        logger.info(f"Trainer initialized with resnet18 on {self.device}")

    def load_data(self) -> None:
        """Load FEMNIST data for *this* FL round."""
        self.LOAD_START_T = time.time()

        # Random partition ID for this round
        self.partition_id = random.randint(1, 2798)

        train_transform, _ = get_data_transform("mnist")
        train_dataset = FEMNIST(
            self.data_dir,
            self.meta_dir,
            self.partition_id,
            dataset="train",
            transform=train_transform,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            timeout=0,
            num_workers=0,
            drop_last=True,
        )

        self.LOAD_END_T = time.time()
        self.load_data_delay = self.LOAD_END_T - self.LOAD_START_T

        logger.info(
            f"[Round {getattr(self, '_round', 0)}] "
            f"Loaded FEMNIST partition_id={self.partition_id} "
            f"with {len(self.train_loader.dataset)} samples"
        )

    def train(self) -> None:
        """Train a model for one FL round on a fresh random partition."""
        # ---- NEW: reload a *new* client partition each round ----
        self.load_data()

        self.TRAIN_START_T = time.time()

        # Reset per-round stats
        self.completed_steps = 0
        self.epoch_train_loss = 1e-4
        self.loss_squared = 0.0

        self.optimizer = optim.Adadelta(self.model.parameters())
        self.model.train()

        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device=self.device)

        for epoch in range(1, self.epochs + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss_vec = criterion(output, target)

                loss_list = loss_vec.tolist()
                loss = loss_vec.mean()

                temp_loss = sum(loss_list) / float(len(loss_list))
                self.loss_squared = (
                    sum([l ** 2 for l in loss_list]) / float(len(loss_list))
                )

                if self.epoch_train_loss == 1e-4:
                    self.epoch_train_loss = temp_loss
                else:
                    self.epoch_train_loss = (
                        (1.0 - self.loss_decay) * self.epoch_train_loss
                        + self.loss_decay * temp_loss
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.completed_steps += 1
                if self.completed_steps >= self.local_steps:
                    break

            if self.completed_steps >= self.local_steps:
                break

        self.dataset_size = len(self.train_loader.dataset)

        self.TRAIN_END_T = time.time()
        self.local_training_delay = self.TRAIN_END_T - self.TRAIN_START_T

        logger.info(
            f"[Round {self._round}] Training on partition_id={self.partition_id} "
            f"done: local_steps={self.completed_steps}, "
            f"epoch_train_loss={self.epoch_train_loss:.6f}"
        )

    def evaluate(self) -> None:
        """Evaluate a model (not used at trainer level here)."""
        pass

    @override
    def save_metrics(self):
        """Save metrics in a model registry and log timing info."""
        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

        logger.info(
            f"Wall-clock time: {time.time()} || "
            f"Training delay (s): {self.local_training_delay:.4f} || "
            f"Loading data delay (s): {self.load_data_delay:.4f} || "
            f"Fetch task delay (s): {getattr(self, 'fetch_delay', 0.0):.4f} || "
            f"MSG delay (s): {getattr(self, 'msg_delay', 0.0):.4f} || "
            f"Send task delay (s): {getattr(self, 'send_delay', 0.0):.4f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchFemnistTrainer(config)
    t.compose()
    t.run()
