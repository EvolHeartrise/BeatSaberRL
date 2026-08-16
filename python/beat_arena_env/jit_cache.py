"""Trust-the-cache fast path for the torch cpp_extension JIT modules.

torch.utils.cpp_extension.load() re-proves "is the built .pyd current?"
on every process start the expensive way: locate the MSVC/CUDA toolchain
(glob scans), write a ninja file, spawn a ninja subprocess for a no-op
build, then import. Warm, that is seconds per extension — paid by every
bench run and every sweep train leg (2026-07-27: ~8-9 s across the
trainer's four extensions, ~3.5 s across the bench's two).

fast_load() proves the same fact with a hash instead (the chart-cache
pattern, see bench_scoresaber.parser_version): sha256 over the source
text + compile flags + torch/CUDA versions, recorded next to the .pyd
after any real build. Hash matches -> import the .pyd directly; no
toolchain probing, no ninja. Hash differs (source edited, torch
upgraded) -> fall back to the caller's full cpp_extension.load path,
which rebuilds, then re-stamp. Same auto-invalidation guarantee, none
of the re-verification cost.

Real builds also tee their compiler output to <name>.build.log next to
the .pyd, so the -Xptxas -v occupancy numbers survive the build that
produced them (torch discards ninja's output on a successful load).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import os
import sys

def _capability():
    """(major, minor) of the current CUDA device. Deferred on purpose:
    jit_cache is imported only from inside the lazy extension loaders,
    which already require a CUDA device (the policy-side loaders wrap
    the whole load in try/except and fall back to eager on CPU-only
    machines), so probing here never fires without a GPU."""
    import torch
    return torch.cuda.get_device_capability()


def cuda_flags():
    """nvcc flags shared by every extension in this repo, arch pinned to
    the CURRENT device (sm_61 on the GTX 1080 era, sm_89 on RunPod 4090
    pods) — a single arch, never a fatbin: builds are per-machine JIT
    artifacts. -Xptxas -v makes ptxas print per-kernel registers /
    stack frame / spill stores+loads / lmem, which fast_load captures
    into <name>.build.log (2026-08-02: the spill baseline for the fused
    policy kernels). The flag list is part of the srchash digest below,
    so a new arch (or any flag edit) invalidates the stamp and forces
    the rebuild that regenerates the log."""
    maj, mnr = _capability()
    flags = [f"-arch=sm_{maj}{mnr}", "-Xptxas", "-v"]
    # BEAT_LINEINFO=1: source-line info for Nsight Compute source view
    # (profiling only — being in the digest, it forces one rebuild each
    # way, which is the intended behavior)
    if os.environ.get("BEAT_LINEINFO") == "1":
        flags.append("-lineinfo")
    # BEAT_MIN_BLOCKS=K: legacy step_kernel occupancy-ladder rung (see
    # the kernel's BEAT_MIN_BLOCKS comment). Digest membership forces
    # the rebuild + fresh ptxas census per rung.
    mb = os.environ.get("BEAT_MIN_BLOCKS")
    if mb and mb.isdigit() and mb != "1":
        flags.append(f"-DBEAT_MIN_BLOCKS={int(mb)}")
    # BEAT_SW_LANES=W: sub-warp lane-count ladder rung (W=2 default)
    sw = os.environ.get("BEAT_SW_LANES")
    if sw and sw.isdigit() and sw != "2":
        flags.append(f"-DBEAT_SW_LANES={int(sw)}")
    return flags


def arch_list():
    """TORCH_CUDA_ARCH_LIST value matching cuda_flags() ('8.9' where
    nvcc wants sm_89) — keeps torch's own flag-injection path from
    compiling for a default arch zoo."""
    maj, mnr = _capability()
    return f"{maj}.{mnr}"


@contextlib.contextmanager
def _build_log(path):
    """Point fds 1/2 at path for the duration of the build.

    torch's cpp_extension hands ninja the raw fileno 1 when verbose=True
    and throws the output away otherwise, so an fd-level redirect is the
    only way to keep the compiler's stderr (where ptxas -v writes).
    No-ops if fd 1/2 or the log path are unusable (pythonw, read-only
    dir) — a build must never fail over its own logging."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        f = open(path, "w")
        saved = os.dup(1), os.dup(2)
    except OSError:
        yield
        return
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        f.close()


def _tail(path, n=4000):
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError:
        return ""


def _digest(sources, extra_flags) -> str:
    import torch
    h = hashlib.sha256()
    h.update(torch.__version__.encode())
    h.update((torch.version.cuda or "").encode())
    for s in sources:
        with open(s, "rb") as f:
            h.update(f.read())
    h.update(repr(sorted(extra_flags or [])).encode())
    return h.hexdigest()


def fast_load(name, sources, build_directory, extra_flags, full_load):
    """Import build_directory/<name>.pyd directly if its recorded source
    hash is current; otherwise run full_load() (the cpp_extension.load
    path, which builds) and record the new hash. full_load's exceptions
    propagate — callers keep their own build-failure policy."""
    digest = _digest(sources, extra_flags)
    stamp = os.path.join(build_directory, name + ".srchash")
    # torch names the artifact <name>.pyd on Windows, <name>.so on Linux;
    # the literal ".pyd" here (until 2026-08-08) meant the fast path never
    # fired on the pod and every process start re-paid the ~8-9 s
    # toolchain re-verification this module exists to skip
    pyd = next((p for p in (os.path.join(build_directory, name + sfx)
                            for sfx in importlib.machinery.EXTENSION_SUFFIXES)
                if os.path.exists(p)), None)
    if os.path.exists(stamp) and pyd is not None:
        try:
            with open(stamp) as f:
                if f.read().strip() == digest:
                    spec = importlib.util.spec_from_file_location(name, pyd)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
        except Exception:
            pass                        # stale/corrupt stamp: full path
    log = os.path.join(build_directory, name + ".build.log")
    try:
        with _build_log(log):
            mod = full_load()
    except Exception as e:               # verbose=True moved the compiler
        raise RuntimeError(              # output out of the exception
            f"{type(e).__name__}: {e}\n--- {log} (tail) ---\n"
            + _tail(log)) from e
    try:                                # stamp only after a good build
        with open(stamp, "w") as f:
            f.write(digest)
    except OSError:
        pass                            # read-only dir: stays slow, works
    return mod
