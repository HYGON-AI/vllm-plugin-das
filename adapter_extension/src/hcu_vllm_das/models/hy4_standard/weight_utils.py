# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight stream boundaries shared by standard HY4 target implementations."""

from collections.abc import Iterable, Iterator

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def filter_target_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Exclude native draft tensors while keeping unknown target names strict."""
    unused_draft_tensors = 0
    for name, tensor in weights:
        if name.startswith("model.dfly_layers."):
            unused_draft_tensors += 1
            continue
        yield name, tensor
    if unused_draft_tensors:
        logger.info(
            "HY4 target loading left %d checkpoint DFly tensors unused",
            unused_draft_tensors,
        )
