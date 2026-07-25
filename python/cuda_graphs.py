"""CUDA-graph capture driver (the torch.cuda.graphs protocol).

Survives the trainer's CUDA-graphs removal (2026-07-25 — the fused
action head + last-block pruning in policy.py eliminated the launch
storm capture used to hide, leaving capture as pure overhead there).
Still used where a graph genuinely pays or serves as the honest
incumbent: the ScoreSaber bench's argmax-forward fallback
(bench_scoresaber._GraphedArgmax) and the forward microbench
(bench_policy_forward.py).
"""

from __future__ import annotations

import torch

_GRAPH_WARMUP = 3  # eager iterations before capture (torch's documented recipe)


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
