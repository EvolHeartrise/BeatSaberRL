"""GEMM microbenchmark for fp16 accumulate (issue #3): the policy's shapes
at the update (B = 16384) and rollout (B = 32768) row counts, fp32 vs fp16
accumulate, through cuBLASLt with our epilogues and through torch with
allow_fp16_accumulation. Also prints the weighted forward / dgrad mix
(time-weighted over the per-forward GEMM counts of the 6-block v58
recipe policy) and every heuristic algo of the aux-epilogue shapes.

    python tools/fp16_accum_gemm_bench.py [--bs 16384 32768] [--algos]

Measured 2026-09-27, RTX 5090 / torch 2.11 cu128: see issue #3. Note the
RELU_AUX_BIAS / DRELU epilogues have no fast COMPUTE_16F kernel on
sm_120 (26-39 TFLOPS across all heuristic algos vs 160-170 at 32F), which
is why fp16_linear keeps those GEMMs fp32.
"""

from __future__ import annotations

import argparse
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, rows per sample, K, N, fwd kind, dgrad kind, count per forward)
# fwd kind: 1 bias (torch addmm), 0 plain (torch mm), 2 relu_aux (Lt);
# dgrad kind: 3 plain (torch mm), 4 drelu (Lt), None = no dgrad
SHAPES = [
    ("qkv", 29, 192, 576, 1, 3, 5),
    ("out", 29, 192, 192, 1, 3, 5),
    ("ffn_up", 29, 192, 768, 2, 3, 5),
    ("ffn_dn", 29, 768, 192, 0, 4, 5),
    ("kv(b5/critic)", 29, 192, 384, 1, 3, 2),
    ("note2", 24, 192, 192, 1, 4, 1),
    ("note0", 24, 32, 192, 2, None, 1),
    ("b5_ffn_up", 5, 192, 768, 2, 3, 1),
    ("b5_ffn_dn", 5, 768, 192, 1, 4, 1),
    ("trunk_proj", 1, 960, 384, 1, 3, 1),
    ("trunk_fc1", 1, 384, 384, 2, 3, 2),
    ("trunk_fc2", 1, 384, 384, 1, 4, 1),
    ("heads", 1, 384, 145, 1, 3, 1),
    ("crit_mlp0", 1, 576, 384, 2, 3, 1),
    ("crit_mlp2", 1, 384, 101, 1, 4, 1),
]


def load_ext():
    from beat_arena_env.jit_cache import arch_list
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", arch_list())
    from torch.utils.cpp_extension import load
    return load("fp16acc_gemm_bench",
                [os.path.join(HERE, "fp16_accum_gemm_bench.cu")],
                extra_ldflags=["-lcublasLt"], verbose=False)


def timeit(f, n=30, warm=5):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        s.record()
        for _ in range(n):
            f()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / n)
    return best


def table(ext, B):
    rows = []
    fmt = lambda v: f"{v:6.1f}" if v else "  n/a "
    for name, rf, K, N, fk, dk, cnt in SHAPES:
        R = B * rf
        x = torch.randn(R, K, device="cuda").half()
        w = (torch.randn(N, K, device="cuda") / K ** 0.5).half()
        b = (torch.randn(N, device="cuda") * 0.1).half()
        g = torch.randn(R, N, device="cuda").half()
        fl = 2 * R * K * N
        r = {"name": name, "R": R, "K": K, "N": N, "fk": fk, "dk": dk,
             "cnt": cnt, "fl": fl}
        auxd = torch.Tensor()
        if dk == 4:
            auxd = torch.randint(0, 256, (R, (K + 127) // 128 * 16),
                                 device="cuda", dtype=torch.uint8)
        for c16 in (0, 1):
            y = ext.run(fk, c16, x, w, b, torch.Tensor(), 1, 0)
            aux = y[1] if (y and fk == 2) else torch.Tensor()
            r[f"lt_f{c16}"] = (fl / timeit(lambda: ext.run(
                fk, c16, x, w, b, aux, 1, 0)) / 1e9 if y else None)
            if dk is not None and ext.run(dk, c16, g, w, b, auxd, 1, 0):
                r[f"lt_d{c16}"] = fl / timeit(lambda: ext.run(
                    dk, c16, g, w, b, auxd, 1, 0)) / 1e9
            else:
                r[f"lt_d{c16}"] = None
        wt = w.t()
        for acc in (0, 1):
            torch.backends.cuda.matmul.allow_fp16_accumulation = bool(acc)
            f = {2: lambda: torch._addmm_activation(b, x, wt),
                 1: lambda: torch.addmm(b, x, wt),
                 0: lambda: torch.mm(x, wt)}[fk]
            r[f"th_f{acc}"] = fl / timeit(f) / 1e9
            r[f"th_d{acc}"] = fl / timeit(lambda: torch.mm(g, w)) / 1e9
        torch.backends.cuda.matmul.allow_fp16_accumulation = False
        rows.append(r)
        print(f"B={B:6d} {name:14s} R={R:7d} K={K:4d} N={N:4d} | "
              f"Lt fwd 32F {fmt(r['lt_f0'])} 16F {fmt(r['lt_f1'])} | "
              f"Lt dgrad 32F {fmt(r['lt_d0'])} 16F {fmt(r['lt_d1'])} | "
              f"torch fwd 32F {fmt(r['th_f0'])} 16F {fmt(r['th_f1'])} | "
              f"torch dgrad 32F {fmt(r['th_d0'])} 16F {fmt(r['th_d1'])}",
              flush=True)
        del x, g, w
    return rows


def weigh(rows, B):
    """ms per forward-GEMM set: production routing at 32F ('base') vs
    the --fp16-accum all routing (torch sites 16F, Lt epilogues 32F)."""
    t = {k: 0.0 for k in ("fb", "ff", "db", "df")}
    for r in rows:
        fl = r["fl"] * r["cnt"] / 1e9
        if r["fk"] == 2:
            t["fb"] += fl / r["lt_f0"]
            t["ff"] += fl / r["lt_f0"]
        else:
            t["fb"] += fl / r["th_f0"]
            t["ff"] += fl / r["th_f1"]
        if r["dk"] == 4:
            t["db"] += fl / r["lt_d0"]
            t["df"] += fl / r["lt_d0"]
        elif r["dk"] == 3:
            t["db"] += fl / r["th_d0"]
            t["df"] += fl / r["th_d1"]
    b, f = t["fb"] + t["db"], t["ff"] + t["df"]
    print(f"B={B}: fwd {t['fb']:.2f} -> {t['ff']:.2f} ms "
          f"({t['fb'] / t['ff']:.3f}x), dgrad {t['db']:.2f} -> "
          f"{t['df']:.2f} ms ({t['db'] / t['df']:.3f}x), fwd+dgrad "
          f"{b:.2f} -> {f:.2f} ms ({b / f:.3f}x)")


def algos(ext, B=16384):
    for name, rf, K, N, kind in [("ffn_up relu_aux fwd", 29, 192, 768, 2),
                                 ("ffn_dn drelu dgrad", 29, 768, 192, 4),
                                 ("qkv bias fwd", 29, 192, 576, 1)]:
        R = B * rf
        w = (torch.randn(N, K, device="cuda") / K ** 0.5).half()
        b = (torch.randn(N, device="cuda") * 0.1).half()
        if kind < 3:
            x = torch.randn(R, K, device="cuda").half()
            aux = torch.empty(R, (N + 127) // 128 * 16, device="cuda",
                              dtype=torch.uint8)
        else:
            x = torch.randn(R, N, device="cuda").half()
            aux = torch.randint(0, 256, (R, (K + 127) // 128 * 16),
                                device="cuda", dtype=torch.uint8)
        fl = 2 * R * K * N
        for c16 in (0, 1):
            ts = []
            for ai in range(ext.nalgo(kind, c16, R, N, K)):
                try:
                    ts.append(f"{fl / timeit(lambda: ext.run(kind, c16, x, w, b, aux, 1, ai), n=10) / 1e9:.0f}")
                except RuntimeError:
                    ts.append("x")
            print(f"{name:22s} {'16F' if c16 else '32F'} algos: "
                  f"{' '.join(ts)} TFLOPS", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bs", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--algos", action="store_true",
                    help="also time every heuristic algo of the "
                         "aux-epilogue shapes")
    a = ap.parse_args()
    torch.manual_seed(0)
    ext = load_ext()
    res = {B: table(ext, B) for B in a.bs}
    for B, rows in res.items():
        weigh(rows, B)
    if a.algos:
        algos(ext)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    main()
