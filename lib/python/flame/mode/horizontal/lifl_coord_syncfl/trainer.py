# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from abc import ABCMeta

from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.trainer import TAG_FETCH, TAG_UPLOAD
from flame.mode.horizontal.syncfl.trainer import Trainer as BaseTrainer
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

logger = logging.getLogger(__name__)

TAG_COORDINATE = "coordinate"


class Trainer(BaseTrainer):
    """A trainer class for coordinated synchronous FL (LIFL)."""

    def _liveness_check(self) -> None:
        """Check with coordinator if work should continue and optionally send meta info."""
        logger.debug("calling _liveness_check")
        channel = self.cm.get_by_tag(TAG_COORDINATE)
        if not channel:
            logger.debug(f"channel not found with tag {TAG_COORDINATE}")
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if not msg:
            logger.warning("liveness_check: received empty message; treating as not-done")
            self._work_done = False
            return

        # EOT may or may not be present
        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]
        else:
            self._work_done = False

        if self._work_done:
            logger.debug("work is done (EOT from coordinator)")
            return

        # META_INFO_REQ is optional; if not present, just skip sending meta info
        if MessageType.META_INFO_REQ not in msg:
            logger.debug("no META_INFO_REQ in message; skipping meta info response")
            return

        logger.debug("sending meta info response")
        # TODO: attach real meta info if needed
        channel.send(end, {MessageType.META_INFO_RES: "some useful info"})
        logger.debug("sent meta info response")

        logger.debug("exited _liveness_check")

    def _get_aggregator(self):
        """Ask coordinator which aggregator to talk to."""
        logger.debug("calling _get_aggregator")
        channel = self.cm.get_by_tag(TAG_COORDINATE)
        if not channel:
            logger.debug(f"channel not found with tag {TAG_COORDINATE}")
            return

        channel.await_join()

        end = channel.one_end()
        msg, _ = channel.recv(end)

        if not msg:
            logger.warning("_get_aggregator: received empty message; assuming work not done")
            self._work_done = False
            # fall back: talk to whoever we saw on the coordinate channel
            self.aggregator_id = end
            logger.debug(f"fallback aggregator_id set to {self.aggregator_id}")
            return

        # Handle EOT safely
        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]
        else:
            self._work_done = False

        if self._work_done:
            logger.debug("work is done (EOT from coordinator in _get_aggregator)")
            return

        # Choose aggregator; if no COORDINATED_ENDS, fall back to end
        if MessageType.COORDINATED_ENDS in msg:
            self.aggregator_id = msg[MessageType.COORDINATED_ENDS]
        else:
            self.aggregator_id = end

        logger.debug(f"exited _get_aggregator with aggregator_id={self.aggregator_id}")

    def _fetch_weights(self, tag: str) -> None:
        """Receive global/parent weights from aggregator."""
        logger.debug("calling _fetch_weights")

        self.FETCH_START_T = time.time()
        self.fetch_success = False

        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found with tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # aggregator_id may be None; backend usually handles broadcast/any
        target = getattr(self, "aggregator_id", None)
        msg, _ = channel.recv(target)
        self.FETCH_END_T = time.time()

        if not msg:
            logger.warning("_fetch_weights: received empty message; skipping weight update")
            self._work_done = False
            self.fetch_success = True  # allow training to proceed with local weights
            # no SEND_TIMESTAMP support in this FLAME version, so msg_delay = 0
            self.msg_delay = 0.0
            self.fetch_delay = self.FETCH_END_T - self.FETCH_START_T
            return

        # Update model weights if present
        if MessageType.WEIGHTS in msg:
            logger.debug("received model weights")
            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        # EOT (optional)
        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        # Round index (optional)
        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        # This FLAME version has no SEND_TIMESTAMP enum; use 0 for msg_delay
        self.MSG_MTr_START_T = 0.0
        self.MSG_MTr_END_T = 0.0
        self.msg_delay = 0.0

        self.fetch_delay = self.FETCH_END_T - self.FETCH_START_T
        self.fetch_success = True

        logger.debug(
            f"work_done: {self._work_done}, round: {self._round}, "
            f"msg_delay={self.msg_delay:.4f}, fetch_delay={self.fetch_delay:.4f}"
        )

    def _send_weights(self, tag: str) -> None:
        """Send local delta weights to aggregator."""
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        self._update_weights()

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)
        delta_weights = self.privacy.apply_dp_fn(delta_weights)

        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            # NOTE: we do NOT include SEND_TIMESTAMP here because this FLAME
            #       MessageType enum does not define it.
        }

        target = getattr(self, "aggregator_id", None)
        channel.send(target, msg)
        logger.debug(f"sending weights done to {target}")

    def compose(self) -> None:
        """Extend base syncfl Trainer compose with liveness + coord logic."""
        super().compose()

        with CloneComposer(self.composer) as composer:
            self.composer = composer

            task_liveness_check = Tasklet("liveness_check", self._liveness_check)
            task_get_aggregator = Tasklet("get_aggregator", self._get_aggregator)

        # Insert: liveness_check -> get_aggregator -> fetch
        fetch_task = self.composer.get_tasklet("fetch")
        fetch_task.insert_before(task_get_aggregator)
        task_get_aggregator.insert_before(task_liveness_check)

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_COORDINATE]
