"""CUDA-graph capture driver (the torch.cuda.graphs protocol).

Survives the trainer's CUDA-graphs removal (2026-07-25 — the fused
action head + last-block pruning in policy.py eliminated the launch
storm capture used to hide, leaving capture as pure overhead there).
Still used where a graph genuinely pays: the ScoreSaber bench's
graphed argmax forward and train_batch's --graph-rollout capture.
(The bench_policy_* microbenches were deleted with the fused-kernel
stack, 2026-08-13; restore from bd07669^ if re-benching — see
PERF_AUDIT_2026-08-16.md Tier 0.)
"""

from __future__ import annotations

import torch

_GRAPH_WARMUP = 3  # eager iterations before capture (torch's documented recipe)


def copy_static(static, value, non_blocking=False):
    """Guarded copy into a fixed forward-path input buffer (2026-08-02):
    an off-shape value would broadcast silently and corrupt every later
    replay/launch; a wrong dtype would silently cast. Used by _Graphed
    subclasses and by the non-graph fused forwards that share the same
    persistent-buffer contract."""
    if value.shape != static.shape or value.dtype != static.dtype:
        raise RuntimeError(
            f"static input expects {tuple(static.shape)} "
            f"{static.dtype}, got {tuple(value.shape)} {value.dtype}")
    static.copy_(value, non_blocking=non_blocking)


class _Graphed:
    """Subclasses copy their inputs into static buffers and call _run(),
    which runs _GRAPH_WARMUP eager iterations on a side stream, then
    captures _compute() into a CUDAGraph, then replays it forever.
    _out holds the outputs (static tensors once captured)."""

    def __init__(self):
        self.graph = None
        self._out = None
        self._iters = 0
        self._side = torch.cuda.Stream()

    def _compute(self):
        raise NotImplementedError

    def _copy_static(self, static, value, non_blocking=False):
        """Subclasses route their __call__ input copies through this."""
        copy_static(static, value, non_blocking=non_blocking)

    def _run(self):
        if self.graph is not None:
            self.graph.replay()
        elif self._iters < _GRAPH_WARMUP:
            self._side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._side):
                self._out = self._compute()
            torch.cuda.current_stream().wait_stream(self._side)
            self._iters += 1
        else:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):  # records, does NOT execute
                self._out = self._compute()
            self.graph.replay()
        return self._out
