# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FEMNIST horizontal FL top aggregator for PyTorch (LIFL)."""

import logging
import random
import time

from flame.mode.composer import Composer
from flame.mode.tasklet import Loop, Tasklet

TAG_DISTRIBUTE = "distribute"
TAG_AGGREGATE = "aggregate"

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as tormodels

from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.lifl_coord_syncfl.top_aggregator import TopAggregator
from flame.fedscale_utils.femnist import FEMNIST
from flame.fedscale_utils.utils_data import get_data_transform

logger = logging.getLogger(__name__)
log_file = "/mydata/image_cls-top_aggregator.log"
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)


def override(method):
    return method


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res


def test_pytorch_model(model, test_data, device="cpu"):
    """Evaluate a PyTorch model and compute loss + top-1/top-5 accuracies."""
    test_loss = 0.0
    correct = 0.0
    top_5 = 0.0

    test_len = 0

    model = model.to(device=device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    with torch.no_grad():
        for data, target in test_data:
            try:
                data, target = Variable(data).to(device=device), Variable(target).to(
                    device=device
                )

                output = model(data)
                loss = criterion(output, target)

                test_loss += loss.data.item()
                acc = accuracy(output, target, topk=(1, 5))

                correct += acc[0].item()
                top_5 += acc[1].item()
                test_len += len(target)
            except Exception as ex:
                logging.info(f"Testing failed as {ex}")
                break

    test_len = max(test_len, 1)
    test_loss /= len(test_data)

    acc_top1 = round(correct / test_len, 4)
    acc_top5 = round(top_5 / test_len, 4)
    test_loss = round(test_loss, 4)

    testRes = {
        "top_1": correct,
        "top_5": top_5,
        "test_loss": test_loss * test_len,
        "test_len": test_len,
    }

    return test_loss, acc_top1, acc_top5, testRes


class PyTorchFemnistAggregator(TopAggregator):
    """PyTorch FEMNIST Top Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.data_dir = "/mydata/FedScale/benchmark/dataset/data/femnist/"
        self.meta_dir = "/mydata/flame_dataset/femnist/"
        self.partition_id = 1

        # Round timing
        self.previous_round_time = time.time()
        self.current_round_time = time.time()

        # Delays / metrics (defaults so logging never crashes)
        self.cpu_time = 0.0
        self.utilization = 0.0
        self.load_data_delay = 0.0
        self.eval_delay = 0.0
        self.agg_delay = 0.0
        self.coordination_delay = 0.0
        self.dist_delay = 0.0
        self.recv_delay = 0.0
        self.queue_delay = 0.0

        # Lists used in logs
        self.msg_from_mid_delays = []
        self.cache_delays = []

    def initialize(self):
        """Initialize role: set device + model."""
        self.device = "cpu"  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tormodels.__dict__["resnet18"](num_classes=62).to(device=self.device)
        logger.info(f"Top Aggregator initialized with resnet18 on {self.device}")

    def load_data(self) -> None:
        """Load a test dataset."""
        self.LOAD_START_T = time.time()

        # Random partition ID for test data each round
        self.partition_id = random.randint(1, 67)

        train_transform, test_transform = get_data_transform("mnist")
        test_dataset = FEMNIST(
            self.data_dir,
            self.meta_dir,
            self.partition_id,
            dataset="test",
            transform=test_transform,
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=True,
            pin_memory=True,
            timeout=0,
            num_workers=0,
            drop_last=False,
        )

        self.dataset = Dataset(dataloader=self.test_loader)

        self.LOAD_END_T = time.time()
        self.load_data_delay = self.LOAD_END_T - self.LOAD_START_T

        logger.info(
            f"Loaded FEMNIST test split partition_id={self.partition_id} "
            f"with {len(self.test_loader.dataset)} samples"
        )

    def train(self) -> None:
        """Train a model (no-op at top aggregator, only evaluation)."""
        pass

    def evaluate(self) -> None:
        """Evaluate (test) the global model on FEMNIST and log detailed metrics."""
        # Measure CPU time only for the evaluation phase
        eval_cpu_start = time.process_time()
        self.EVAL_START_T = time.time()

        test_loss, test_accuracy, acc_5, testRes = test_pytorch_model(
            self.model, self.test_loader, device=self.device
        )

        self.EVAL_END_T = time.time()
        eval_cpu_end = time.process_time()

        # Delays
        self.eval_delay = self.EVAL_END_T - self.EVAL_START_T
        self.cpu_time = eval_cpu_end - eval_cpu_start

        # Round timing
        now = time.time()
        round_duration = now - self.previous_round_time
        self.current_round_time = now

        if round_duration > 0:
            # crude approximation; keeps utilization in [0, 1]
            self.utilization = min(self.cpu_time / round_duration, 1.0)
        else:
            self.utilization = 0.0

        # Safe averages
        if self.msg_from_mid_delays:
            avg_msg_from_mid = sum(self.msg_from_mid_delays) / len(
                self.msg_from_mid_delays
            )
        else:
            avg_msg_from_mid = 0.0

        if self.cache_delays:
            total_cache_delay = sum(self.cache_delays)
            avg_cache_delay = total_cache_delay / len(self.cache_delays)
        else:
            total_cache_delay = 0.0
            avg_cache_delay = 0.0

        logger.info(
            f"Wall-clock time: {self.current_round_time} ||"
            f"Test loss: {test_loss:.4f} || "
            f"Test accuracy: {test_accuracy:.4f} || "
            f"CPU time: {self.cpu_time:.4f} || "
            f"CPU utilization: {self.utilization:.4f} || "
            f"R#{self._round}'s duration (s): {round_duration:.4f} || "
            f"Loading data delay (s): {self.load_data_delay:.4f} || "
            f"Eval delay (s): {self.eval_delay:.4f} || "
            f"Agg delay (s): {self.agg_delay:.4f} || "
            f"Coordination delay (s): {self.coordination_delay:.4f} || "
            f"DIST task delay: {self.dist_delay:.4f} || "
            f"RECV task delay: {self.recv_delay:.4f} || "
            f"Queueing delay: {self.queue_delay:.4f} || "
            f"MSG (from mid) Ave. delay: {avg_msg_from_mid:.4f} || "
            f"Total cache delay: {total_cache_delay:.4f} || "
            f"Ave. cache delay: {avg_cache_delay:.4f}"
        )

        logger.info(
            f"Top Aggregator Timestamps: "
            f"CACHE, CACHE_START_T: {getattr(self, 'CACHE_START_T', 0.0)}, CACHE_END_T: {getattr(self, 'CACHE_END_T', 0.0)} || "
            f"AGG, AGG_START_T: {getattr(self, 'AGG_START_T', 0.0)}, AGG_END_T: {getattr(self, 'AGG_END_T', 0.0)} || "
            f"EVAL, EVAL_START_T: {self.EVAL_START_T}, EVAL_END_T: {self.EVAL_END_T} || "
            f"RECV, RECV_START_T: {getattr(self, 'RECV_START_T', 0.0)}, RECV_END_T: {getattr(self, 'RECV_END_T', 0.0)} || "
            f"DIST, DIST_START_T: {getattr(self, 'DIST_START_T', 0.0)}, DIST_END_T: {getattr(self, 'DIST_END_T', 0.0)} || "
            f"LOAD, LOAD_START_T: {self.LOAD_START_T}, LOAD_END_T: {self.LOAD_END_T} || "
            f"MSG_MTo, MSG_MTo_START_T: {getattr(self, 'MSG_MTo_START_T', 0.0)}, MSG_MTo_END_T: {getattr(self, 'MSG_MTo_END_T', 0.0)}"
        )

        # prepare for next round
        self.previous_round_time = self.current_round_time

        # Update metrics so they can be saved in registry
        self.update_metrics(
            {
                "test-loss": test_loss,
                "test-accuracy": test_accuracy,
            }
        )

    @override
    def compose(self) -> None:
        """Compose role with tasklets (same pipeline, just using our methods)."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)
            task_init = Tasklet("initialize", self.initialize)
            task_load_data = Tasklet("load_data", self.load_data)

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)
            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            task_train = Tasklet("train", self.train)
            task_eval = Tasklet("evaluate", self.evaluate)

            task_analysis = Tasklet("analysis", self.run_analysis)
            task_save_metrics = Tasklet("save_metrics", self.save_metrics)
            task_increment_round = Tasklet("inc_round", self.increment_round)

            task_save_params = Tasklet("save_params", self.save_params)
            task_save_model = Tasklet("save_model", self.save_model)

            # NOTE: coordination with coordinator (which mids to talk to)
            # is handled by lifl_coord_syncfl.TopAggregator through
            # its own compose() + get_coordinated_ends logic, so we
            # don't re-insert that here.

        loop = Loop(loop_check_fn=lambda: self._work_done)
        (
            task_internal_init
            >> task_init
            >> loop(
                task_load_data
                >> task_put
                >> task_get
                >> task_train
                >> task_eval
                >> task_analysis
                >> task_save_metrics
                >> task_increment_round
            )
            >> task_save_params
            >> task_save_model
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchFemnistAggregator(config)
    a.compose()
    a.run()
