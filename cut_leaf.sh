#!/bin/bash
# cut_leaf.sh — turn a WSD trunk checkpoint into a decayed leaf on a
# THROWAWAY RunPod pod, without touching the pod training the trunk.
#
#   bash cut_leaf.sh trunk_upd03500.pt [cooldown_updates] [ent_final]
#
# Flow: create a 4090 in the volume's datacenter (EU-RO-1) mounting the
# SAME network volume -> wait for ssh -> setup_pod.sh (idempotent; CUDA
# build cache on the volume makes it fast) -> leaf cooldown + bench,
# all outputs under runs/leafcuts/<ckpt>/ (never the trunk's run dir)
# -> print bench -> ALWAYS terminate the pod (EXIT trap + hard timeout).
#
# Discipline: numbered trunk_updNNNNN.pt only — trunk_final.pt is
# rewritten by live extensions (read-while-write race). Leaf ckpts and
# logs get unique paths, so volume sharing never writes the same file
# from two pods. API key: ~/.runpod_api_key (chmod 600, not in git).
set -euo pipefail

CKPT_NAME="${1:?usage: cut_leaf.sh trunk_updNNNNN.pt [cooldown] [ent_final]}"
COOLDOWN="${2:-400}"
ENT_FINAL="${3:-3e-6}"
# TRUNK_RUN selects which trunk run dir to branch from (overridable:
# TRUNK_RUN=wsd_probe_1h_r3 bash cut_leaf.sh ...). Leaf output paths
# carry it so same-numbered checkpoints of different trunks never
# collide (r3's upd12000 vs v16's upd12000).
TRUNK_RUN="${TRUNK_RUN:-v16_probe_1h}"

case "$CKPT_NAME" in
  trunk_upd*.pt) ;;
  *) echo "refusing '$CKPT_NAME': cut leaves from NUMBERED trunk ckpts only" \
     "(trunk_final.pt is rewritten by live extensions)"; exit 1 ;;
esac

API="https://rest.runpod.io/v1"
KEY=$(tr -d '[:space:]' < ~/.runpod_api_key)
AUTH=(-H "Authorization: Bearer $KEY" -H "Content-Type: application/json")
SSH_KEY=~/.ssh/runpod_ed25519
PUBKEY=$(cat "$SSH_KEY.pub")
REPO=/workspace/BeatSaberRL
TRUNK_DIR="$REPO/python/runs/$TRUNK_RUN"
UPD="${CKPT_NAME//[!0-9]/}"
LEAF_DIR="$REPO/python/runs/leafcuts/${TRUNK_RUN}_upd${UPD}_cd${COOLDOWN}"
HARD_CAP_S=$(( 3 * 3600 ))   # pod lifetime cap: no overnight billing

echo "[cut_leaf] $CKPT_NAME -> $COOLDOWN-update leaf (ent -> $ENT_FINAL)"

POD_ID=$(curl -sf --max-time 60 "${AUTH[@]}" -X POST "$API/pods" -d @- << EOF | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])"
{
  "name": "leafcut-$TRUNK_RUN-upd$UPD",
  "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "cloudType": "SECURE",
  "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
  "gpuCount": 1,
  "containerDiskInGb": 15,
  "networkVolumeId": "k99m38c69w",
  "dataCenterIds": ["EU-RO-1"],
  "ports": ["22/tcp"],
  "env": {"PUBLIC_KEY": "$PUBKEY"}
}
EOF
)
echo "[cut_leaf] pod $POD_ID created"
cleanup() {
  echo "[cut_leaf] terminating pod $POD_ID"
  curl -s --max-time 60 "${AUTH[@]}" -X DELETE "$API/pods/$POD_ID" > /dev/null || \
    echo "[cut_leaf] WARNING: DELETE failed — check the RunPod console!"
}
trap cleanup EXIT

# wait for ssh (publicIp + portMappings["22"])
IP=""; PORT=""
for i in $(seq 1 60); do
  read -r IP PORT < <(curl -s --max-time 30 "${AUTH[@]}" "$API/pods/$POD_ID" | \
    python3 -c "
import json, sys
p = json.load(sys.stdin)
pm = p.get('portMappings') or {}
print(p.get('publicIp') or '', pm.get('22') or '')") || true
  [ -n "$IP" ] && [ -n "$PORT" ] && break
  sleep 10
done
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "[cut_leaf] no ssh after 10 min"; exit 1; }
echo "[cut_leaf] ssh root@$IP:$PORT"
# keepalives: a severed TCP session must fail fast, not hang the
# orchestrator against an already-finished pod (bug found 2026-08-11)
SSH=(ssh -p "$PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=no
     -o ConnectTimeout=15 -o ServerAliveInterval=15
     -o ServerAliveCountMax=4 "root@$IP")
for i in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done

# bootstrap + leaf + bench, under the pod-lifetime cap
timeout "$HARD_CAP_S" "${SSH[@]}" bash -s << EOF
set -euo pipefail
bash $REPO/setup_pod.sh > /workspace/leafcut_${TRUNK_RUN}_upd${UPD}.setup.log 2>&1
export PATH=/root/.cargo/bin:/root/venv/bin:/usr/local/cuda/bin:\$PATH VIRTUAL_ENV=/root/venv
cd $REPO/python
mkdir -p "$LEAF_DIR"
python train_batch.py \
  --num-envs 16384 --horizon 8 --foot-contact --gears 5 --seed 0 \
  --ranked-maps --chart-pool 24576 \
  --gamma 0.965 --gae-lambda 0.98 --ent-coef 0.03 \
  --resume "$TRUNK_DIR/$CKPT_NAME" \
  --cooldown-updates $COOLDOWN --ent-final $ENT_FINAL \
  --run-dir "$LEAF_DIR" 2>&1 | tee "$LEAF_DIR/leaf_train.log" | tail -3
LEAF=\$(ls "$LEAF_DIR"/leaf_from_upd*.pt)
python bench_scoresaber.py --ckpt "\$LEAF" --num-envs 2048 --foot-contact \
  --out "bench/leafcut_${TRUNK_RUN}_upd${UPD}_cd${COOLDOWN}.json" > /dev/null
python - << 'PYEOF'
import json
b = json.load(open("bench/leafcut_${TRUNK_RUN}_upd${UPD}_cd${COOLDOWN}.json"))
print(f"LEAF_BENCH $TRUNK_RUN upd$UPD cd$COOLDOWN upp {b['total_unweighted_pp']:.2f} "
      f"wpp {b['total_weighted_pp']:.2f}")
PYEOF
EOF
echo "[cut_leaf] done — leaf + bench on the volume (runs/leafcuts/upd$UPD, bench/leafcut_${TRUNK_RUN}_upd${UPD}_cd${COOLDOWN}.json)"
