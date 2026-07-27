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
"""

from __future__ import annotations

import hashlib
import importlib.util
import os


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
    pyd = os.path.join(build_directory, name + ".pyd")
    if os.path.exists(stamp) and os.path.exists(pyd):
        try:
            with open(stamp) as f:
                if f.read().strip() == digest:
                    spec = importlib.util.spec_from_file_location(name, pyd)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
        except Exception:
            pass                        # stale/corrupt stamp: full path
    mod = full_load()
    try:                                # stamp only after a good build
        with open(stamp, "w") as f:
            f.write(digest)
    except OSError:
        pass                            # read-only dir: stays slow, works
    return mod
