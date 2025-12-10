# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import (
    TopAggregator as BaseTopAggregator,
)
from flame.mode.message import MessageType
from flame.mode.tasklet import Tasklet

logger = logging.getLogger(__name__)

# Coordinate with the coordinator to get which middle aggregators to talk to
TAG_COORDINATE = "coordinate"


class TopAggregator(BaseTopAggregator):
    """Top-level aggregator for LIFL coordinated synchronous FL."""

    def get_channel(self, tag: str):
        """Return channel of a given tag when it is ready to use."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            raise ValueError(f"channel not found for tag {tag}")

        # Wait until there is at least one peer on this channel
        channel.await_join()
        return channel

    def get_coordinated_ends(self):
        """Receive the ends of middle aggregators from coordinator."""
        COORD_START_T = time.time()

        logger.debug("calling get_coordinated_ends()")
        channel = self.get_channel(TAG_COORDINATE)

        # Coordinator is the only end on this coordination channel
        end = channel.one_end()
        msg, _ = channel.recv(end)
        logger.debug(f"received coordination message = {msg} from {end}")

        # Check if training is done
        self._work_done = msg.get(MessageType.EOT, False)
        if self._work_done:
            logger.debug("work is done in get_coordinated_ends()")
            COORD_END_T = time.time()
            self.coordination_delay = COORD_END_T - COORD_START_T
            return

        # Set the 'ends' of the distribute channel to the coordinated middle aggregators
        dist_channel = self.cm.get_by_tag(TAG_DISTRIBUTE)
        if not dist_channel:
            raise ValueError("distribute channel not found")

        # Override distribute channel's ends() to return the coordinated list
        dist_channel.ends = lambda: msg[MessageType.COORDINATED_ENDS]

        logger.debug("exited get_coordinated_ends()")
        COORD_END_T = time.time()
        # FIXED: measure real coordination delay
        self.coordination_delay = COORD_END_T - COORD_START_T

    def compose(self) -> None:
        """Compose role with tasklets."""
        # Let the base class (syncFL.TopAggregator) build its default pipeline
        super().compose()

        # Inject coordination task before "distribute"
        with CloneComposer(self.composer) as composer:
            self.composer = composer
            task_get_coord_ends = Tasklet("get_coord_ends", self.get_coordinated_ends)

        # Insert our coordination step before distribute
        self.composer.get_tasklet("distribute").insert_before(task_get_coord_ends)

        # In coordinated setups, we typically let coordinator manage EOT decision,
        # so we can drop the local "inform_end_of_training" tasklet if present.
        try:
            self.composer.get_tasklet("inform_end_of_training").remove()
        except StopIteration:
            # If the base class doesn't have this tasklet, ignore
            logger.debug("inform_end_of_training tasklet not found; skipping remove()")

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_AGGREGATE, TAG_DISTRIBUTE, TAG_COORDINATE]
