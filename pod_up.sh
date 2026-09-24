#!/bin/bash
# Community-pod bootstrap (consolidates the 2026-08-24 lessons; run ON
# the pod as: bash /root/pod_up.sh [--skip-prefetch]).
# Expects /root/ba_code.tgz (repo root: `git archive -o ba_code.tgz HEAD`)
# + /root/catalog*.gz + /root/failures.json.gz.
set -e
# (3) /workspace may be a small separate partition; the requested
# containerDiskInGb lands on /. Relocate to /data + symlink if so.
WS_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
if [ "$WS_GB" -lt 40 ]; then
  echo "[pod_up] /workspace only ${WS_GB}G -> relocating to /data"
  mkdir -p /data
  # 2026-09-03: the pod VOLUME disk (/workspace) survives a stop while
  # the container disk (/data) does not, so a stale dangling symlink is
  # the normal case on restart -- always (re)create the target.
  mkdir -p /data/BeatSaberRL
  [ -L /workspace/BeatSaberRL ] || {
    rm -rf /workspace/BeatSaberRL
    ln -s /data/BeatSaberRL /workspace/BeatSaberRL; }
else
  mkdir -p /workspace/BeatSaberRL
fi
# (9) community images lack rsync
which rsync > /dev/null 2>&1 || {
  apt-get update -q > /dev/null && apt-get install -y -q rsync > /dev/null; }
# (mfs/overlay chown quirks): no-same-owner everywhere
tar xzf /root/ba_code.tgz -C /workspace/BeatSaberRL --no-same-owner
M=/workspace/BeatSaberRL/python/maps
mkdir -p $M
gunzip -c /root/catalog.json.gz > $M/scoresaber_catalog.json
gunzip -c /root/catalog.orig.json.gz > $M/scoresaber_catalog.orig.json
gunzip -c /root/failures.json.gz > $M/prefetch_failures.json
B=/workspace/BeatSaberRL
bash $B/setup_pod.sh > /workspace/setup.log 2>&1 \
  && echo "SETUP-OK" || echo "SETUP-SOFT-FAIL (gate may need maps)"
export PATH=/root/.cargo/bin:/root/venv/bin:/usr/local/cuda/bin:$PATH \
       VIRTUAL_ENV=/root/venv
cd $B/python
if [ "$1" != "--skip-prefetch" ]; then
  python prefetch_maps.py --skip-pool > /workspace/prefetch.log 2>&1 \
    && echo "PREFETCH-OK" || { echo "PREFETCH-FAILED"; exit 1; }
  python -m pytest tests/ -q > /workspace/gate2.log 2>&1 \
    && echo "GATE2-OK" || { echo "GATE2-FAILED"; tail -3 /workspace/gate2.log; exit 1; }
fi
echo "POD-UP-DONE"
