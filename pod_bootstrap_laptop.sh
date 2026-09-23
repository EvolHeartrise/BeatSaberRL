#!/bin/bash
# One-shot setup of a fresh beat-arena community pod from the laptop.
# usage: SHIP_DIR=<dir with ba_code.tgz + catalog*.gz> bash pod_bootstrap_laptop.sh POD_ID
# Runs FROM THE LAPTOP against a freshly created RunPod community pod:
# waits for the tcp/22 mapping, opens a ControlMaster, plants the ssh
# watcher, ships the bundle, runs pod_up.sh --skip-prefetch, uploads
# chart_cache + raw maps from the local repo, re-runs the gate, and
# does a 5-update production-shape smoke that builds the ranked pool.
# Lesson 2026-09-02: NEVER `rsync -a src/ root@pod:/root/` — the trailing-slash
# source applies the laptop uid (1000) to /root itself, and sshd StrictModes
# then rejects authorized_keys (the "lockout" on 4 pods). Always ship with
# --no-owner --no-group. ControlMaster kept as belt-and-braces.
set -u
POD=$1
S=${SHIP_DIR:-/tmp/ba_ship}   # holds ba_code.tgz + catalog*.gz + failures.json.gz (see ship step in README)
LOCAL_MAPS="/home/evolheartrise/git projects/claude-teaches/projects/beat-arena/python/maps"
KEY=$(cat ~/.runpod_api_key)
CTL=$HOME/.ssh/cm_$POD
ts() { date -u +%T; }

# --- wait for the public tcp/22 mapping (GraphQL runtime.ports is the truth)
t0=$(date +%s); IP=""; PORT=""
while [ -z "$PORT" ]; do
  R=$(curl -s -m 20 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" https://api.runpod.io/graphql -d "{\"query\":\"{ pod(input:{podId:\\\"$POD\\\"}) { runtime { ports { ip privatePort publicPort type isIpPublic } } } }\"}")
  L=$(echo "$R" | python3 -c "
import json,sys
try:
    r=json.load(sys.stdin)['data']['pod']['runtime']
    for p in (r or {}).get('ports',[]):
        if p['privatePort']==22 and p['type']=='tcp' and p['isIpPublic']: print(p['ip'], p['publicPort'])
except Exception: pass" | head -1)
  if [ -n "$L" ]; then IP=${L%% *}; PORT=${L#* }; break; fi
  [ $(( $(date +%s) - t0 )) -gt 1200 ] && echo "NO-PORT-IN-20MIN" && exit 1
  sleep 10
done
echo "[$(ts)] PORT-MAPPED $IP:$PORT (+$(( $(date +%s) - t0 ))s)"; echo "$IP $PORT" > $S/pod_addr

BASE="-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/runpod_ed25519 -p $PORT"
R="root@$IP"

# --- first contact: open the ControlMaster the instant sshd answers
rm -f $CTL
until ssh $BASE -M -S $CTL -o ControlPersist=yes -fN $R 2>/dev/null; do
  [ $(( $(date +%s) - t0 )) -gt 1500 ] && echo "NO-SSH-IN-25MIN" && exit 1; sleep 3; done
echo "[$(ts)] CONTROLMASTER-UP (+$(( $(date +%s) - t0 ))s)"
MUX="-o ControlPath=$CTL -o LogLevel=ERROR -p $PORT"
sshm() { ssh $MUX $R "$@"; }

# --- plant the self-heal + forensics watcher through the master
sshm 'cat > /root/sshwatch.sh <<'"'"'EOF'"'"'
#!/bin/bash
L=/workspace/sshwatch.log
for i in $(seq 1 600); do
  { echo "=== $(date -u +%T) i=$i"; ls -ld /root /root/.ssh /root/.ssh/authorized_keys 2>&1
    sha256sum /root/.ssh/authorized_keys 2>&1 | cut -c1-16
    grep -rhE "^(PubkeyAuthentication|AuthorizedKeysFile|AuthorizedKeysCommand|PermitRootLogin|StrictModes|PasswordAuthentication|UsePAM)" /etc/ssh/sshd_config /etc/ssh/sshd_config.d/ 2>/dev/null
    pgrep -a sshd | head -3; tail -3 /var/log/auth.log 2>/dev/null; } >> $L
  mkdir -p /root/.ssh; chmod 700 /root /root/.ssh
  grep -qF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null || { echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; echo "REAPPENDED-KEY $(date -u +%T)" >> $L; }
  chmod 600 /root/.ssh/authorized_keys; sleep 3
done
EOF
chmod +x /root/sshwatch.sh; nohup setsid /root/sshwatch.sh >/dev/null 2>&1 < /dev/null & echo WATCHER-PLANTED'
sshm 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h / /workspace | tail -2; python -c "import torch;print(torch.__version__)"'

echo "[$(ts)] 1/6 rsync + ship bundle"
sshm 'which rsync >/dev/null || { apt-get update -q >/dev/null 2>&1; apt-get install -y -q rsync >/dev/null 2>&1; }; which rsync'
rsync -rlt --no-owner --no-group -e "ssh $MUX" $S/ship/ $R:/root/ || { echo SHIP-FAILED; exit 1; }
rsync -rlt --no-owner --no-group -e "ssh $MUX" "/home/evolheartrise/git projects/claude-teaches/projects/beat-arena/pod_up.sh" $R:/root/pod_up.sh
sshm 'ls -la /root/ba_code.tgz /root/catalog.json.gz /root/pod_up.sh' || exit 1

echo "[$(ts)] 2/6 launch pod_up.sh --skip-prefetch (background on pod)"
sshm 'nohup setsid bash /root/pod_up.sh --skip-prefetch > /root/pod_up.out 2>&1 < /dev/null & echo launched'

echo "[$(ts)] 3/6 wait for repo layout, then upload caches"
for i in $(seq 1 60); do sshm 'test -d /workspace/claude-teaches/projects/beat-arena/python' 2>/dev/null && break; sleep 5; done
M=/workspace/claude-teaches/projects/beat-arena/python/maps
sshm "mkdir -p $M"
t1=$(date +%s); rsync -a --no-owner --no-group -e "ssh $MUX" "$LOCAL_MAPS/chart_cache" $R:$M/ && echo "[$(ts)] chart_cache uploaded in $(( $(date +%s) - t1 ))s"
t1=$(date +%s); rsync -a --no-owner --no-group -e "ssh $MUX" "$LOCAL_MAPS/scoresaber" $R:$M/ && echo "[$(ts)] raw maps uploaded in $(( $(date +%s) - t1 ))s"
sshm "ls $M/chart_cache | wc -l; du -sh $M/scoresaber; df -h / /workspace | tail -2"

echo "[$(ts)] 4/6 wait for pod_up.sh"
for i in $(seq 1 240); do sshm 'grep -q "POD-UP-DONE\|PREFETCH-FAILED\|GATE2-FAILED" /root/pod_up.out' 2>/dev/null && break; sleep 10; done
sshm 'cat /root/pod_up.out; echo ---setup.log tail---; tail -5 /workspace/setup.log'

echo "[$(ts)] key-loss forensics: $(sshm 'grep -c REAPPENDED /workspace/sshwatch.log') re-appends; fresh-auth test:"
ssh $BASE $R 'echo FRESH-AUTH-OK' 2>&1 | tail -1

echo "[$(ts)] 5/6 full gate with maps present"
sshm 'export PATH=/root/.cargo/bin:/root/venv/bin:/usr/local/cuda/bin:$PATH VIRTUAL_ENV=/root/venv; cd /workspace/claude-teaches/projects/beat-arena/python && python -m pytest tests/ -q 2>&1 | tail -2'

echo "[$(ts)] 6/6 production-shape smoke (builds the ranked pool npz, 5 updates)"
sshm 'export PATH=/root/.cargo/bin:/root/venv/bin:/usr/local/cuda/bin:$PATH VIRTUAL_ENV=/root/venv PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /workspace/claude-teaches/projects/beat-arena/python && python train_batch.py --num-envs 32768 --horizon 32 --minibatches 64 --mb-accum 2 --foot-contact --gears 5 --seed 0 --ranked-maps --chart-pool 999999 --mirror-augment --gamma 0.965 --gae-lambda 0.98 --energy-coef 7.5e-6 --ent-coef 0.03 --ent-tau 125 --ent-final 1e-3 --lead-s 0.5 --lr 7e-4 --start-pose-jitter 0.15 --foot-z-coef 0.01 --foot-x-coef 0.01 --foot-hinge-r 0.12 --foot-shape-coef 0.05 --limit-coef 5e-3 --limit-arms --wall-despawn --crouch-start-prob 0.25 --total-steps 5242880 --run-dir runs/v39_smoke 2>&1 | grep -E "ranked-pool|batched PPO|update +5/5|THROUGHPUT|Error|Traceback"; ls pool_cache/'
echo "[$(ts)] BOOTSTRAP-DONE  (master socket kept at $CTL)"
