"""Rollout-only CUDA graph (2026-07-25 opt-survey round 4, item #9).

Captures policy forward + multinomial sample + the env sim kernel as ONE
graph, replayed T times per update. The round-4 drain test showed the
N=1024 rollout's CPU enqueue (~283 ms) roughly equals its GPU exec
(~260 ms); the graph removes the enqueue side (63 launches + Python
dispatch per step -> 1 replay + 5 buffer stores).

Measured at N=1024 (2026-07-25 prototype):
  fp32 rollout graph:      1.03x vs eager fp32 (exec-bound — little left)
  fp16 rollout graph:      1.20x vs eager fp32 (launch cost gone, the
                           fp16 GEMM kernels finally pay at rollout B)
so the graph is only worth running combined with --fp16 all. NOTE: fp16
rollout changes sampling numerics (~1e-3 logits) and the graph changes
the sampling RNG STREAM — same accepted class as the fused-head commit,
but a learning gate (paired 50M) is required before this becomes any
kind of default.

Graph-safety contract:
- policy fp16 caches are refreshed IN PLACE (fp16_linear._half_params,
  NotePolicy._refresh_head_cache) so captured storages stay live;
  refresh() MUST run after opt.steps and before the next replay.
- venv._gen is registered with the graph so the chart-candidate randint
  draws fresh values each replay; multinomial rides the default
  generator, which torch registers automatically.
- All env state/IO tensors are the venv's preallocated statics; the sim
  kernel mutates them in place, so replay N's policy forward reads
  replay N-1's observation exactly like the eager loop.
"""

from __future__ import annotations

import torch


class GraphedRollout:
    def __init__(self, policy, venv, warmup: int = 3):
        assert venv._obs.is_cuda, "GraphedRollout needs the CUDA env"
        self.policy = policy
        self.venv = venv
        self._version = None
        self.refresh()  # caches must exist (and be stable) before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):  # cuBLAS workspaces, allocator, RNG
                a, _, _, _ = policy.get_action_and_value(
                    venv._obs, need_entropy=False)
                venv.step(a)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(venv._gen)
        with torch.no_grad(), torch.cuda.graph(self.graph):
            # need_entropy=False: the rollout discards entropy; skipping
            # it removes ~5 dead kernels from every replay (no RNG in
            # the entropy path, so the sampling stream is unchanged)
            a, logp, _, v = policy.get_action_and_value(
                venv._obs, need_entropy=False)
            venv.step(a)
        # static outputs: same storages every replay
        self.action, self.logp, self.value = a, logp, v

    def refresh(self):
        """In-place fp16 cache refresh, once per weights version. Cheap
        no-op when the version is unchanged or the rollout is fp32."""
        if getattr(self.policy, "fp16", False):
            import fp16_linear as _fl
            if self._version != _fl._weights_version:
                self.policy.refresh_fp16_caches()
                self._version = _fl._weights_version

    def step(self):
        """One env step: replay the graph, return the static outputs.
        Callers must copy anything they keep before the next replay."""
        self.graph.replay()
        v = self.venv
        return self.action, self.logp, self.value, v._reward, v._done
