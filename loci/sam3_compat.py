"""Workarounds for the SAM 3 package (facebookresearch/sam3 @ f1647e8)."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

log = logging.getLogger(__name__)


def build_video_predictor(checkpoint: str, frame_batch: int = 4) -> Any:
    """SAM 3.1 multiplex video predictor that runs on a 16 GB non-Hopper GPU.

    - use_fa3=False: FlashAttention-3 only runs on Hopper.
    - The builder hard-codes 16-frame batches for grounding and postprocessing, which runs out of memory
      on 16 GB; both are plain attributes, so lower them to `frame_batch`.
    - start_session passes kwargs (offload_state_to_cpu, video_loader_type) that the multiplex model's
      init_state does not accept; drop any kwarg its signature does not list.
    """
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint, use_fa3=False)
    predictor.model.batched_grounding_batch_size = frame_batch
    predictor.model.postprocess_batch_size = frame_batch
    init_state = predictor.model.init_state
    accepted = set(inspect.signature(init_state).parameters)

    @functools.wraps(init_state)
    def filtered_init_state(**kwargs: Any) -> Any:
        dropped = sorted(set(kwargs) - accepted)
        if dropped:
            log.debug("init_state: dropping unsupported kwargs %s", dropped)
        return init_state(**{k: v for k, v in kwargs.items() if k in accepted})

    predictor.model.init_state = filtered_init_state
    return predictor
