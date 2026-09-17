#!/bin/bash
# Fresh-pod bootstrap for beat-arena on RunPod (2026-08-08).
#
# A stopped/recreated pod is a FRESH container: /root (venv, rustup,
# cargo target) is gone. Everything on /workspace (the network volume)
# survives — including the repo, maps, runs, bench results, AND the
# torch cpp_extension build dirs (csrc/build*), so the CUDA extensions
# fast-load via jit_cache without recompiling as long as the image's
# torch/CUDA versions match the digest stamp.
#
# Idempotent: safe to re-run on a warm pod (skips what exists).
#
#   bash /workspace/claude-teaches/projects/beat-arena/setup_pod.sh
set -euo pipefail

REPO=/workspace/claude-teaches/projects/beat-arena
VENV=/root/venv
mkdir -p /workspace   # RunPod mounts it; VastAI has none
[ -d "$REPO" ] || { echo "$REPO missing — rsync/clone the repo first"; exit 1; }

# 1. PATH for interactive shells (cuda first, then venv)
for line in \
  'export CUDA_HOME=/usr/local/cuda' \
  'export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH'; do
  grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done
export CUDA_HOME=/usr/local/cuda
export PATH=$VENV/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH

# 2. venv over the image's torch (system-site-packages: torch stays the
#    image's cu-build; pip only adds what's missing)
[ -x "$VENV/bin/python" ] || python -m venv "$VENV" --system-site-packages
"$VENV/bin/pip" install -q ninja pytest maturin certifi \
    gymnasium scipy scikit-learn gpytorch

# 3. Rust toolchain + the reference sim (target-cpu=native for THIS host;
#    cargo work on container disk — the network FS is too slow for builds)
if ! command -v cargo >/dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q \
      --default-toolchain stable --profile minimal
  export PATH=$HOME/.cargo/bin:$PATH
fi
cd "$REPO/rust"
CARGO_TARGET_DIR=/root/cargo-target VIRTUAL_ENV=$VENV \
    "$VENV/bin/maturin" develop --release 2>&1 | tail -2
# f32 build of the same sources (module beat_arena_rs32) — the bench's
# default engine since 2026-09-16 (training runs f32). The `cd` matters:
# cargo reads .cargo/config.toml from the invocation dir upward, and
# rust32/.cargo/config.toml is what gives this build target-cpu=native
cd "$REPO/rust32"
CARGO_TARGET_DIR=/root/cargo-target VIRTUAL_ENV=$VENV \
    "$VENV/bin/maturin" develop --release 2>&1 | tail -2

# 4. Verify: sim imports, CUDA extensions load (fast path if the volume's
#    build dirs match this image), full gate suite
"$VENV/bin/python" - << 'PYEOF'
import torch
from beat_arena_rs import VecBeatSim
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"cc={torch.cuda.get_device_capability() if torch.cuda.is_available() else '-'}")
print("beat_arena_rs OK")
PYEOF
cd "$REPO/python"
"$VENV/bin/python" -m pytest tests/ -q 2>&1 | tail -1

echo "=== pod setup complete ==="
