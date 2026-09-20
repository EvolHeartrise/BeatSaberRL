"""Structured per-update training metrics (2026-09-03, dashboard option B).

Two pure directions over ONE field vocabulary:

* `update_record(...)` / `dumps_record(rec)` — the trainer's formatter.
  `train_batch.py` calls it at the once-per-update print sync (where
  every value is already on the host) and appends one JSON line to
  `<run_dir>/metrics.jsonl`. No new device syncs, no science.
* `parse_log(text)` / `parse_log_file(path)` — the backfill's parser.
  Turns a trainer stdout log into the SAME objects, one per update,
  joining the `update`, `[kl]`, event (`[fall]`/`[tilt]`/`[lift]`/
  `[flail]`/`[limitrest]`/`[pd-net]`/`[envelope]`), `[diag]`,
  `[bench-mem]` and `[ckpt]` lines by update index, plus run-level
  fields from the `argv:`/`config:`/`=== batched PPO`/`[ranked-pool]`/
  `RESUMED`/`WSD`/`THROUGHPUT` lines.

A log may hold several trainer invocations back to back (the pod
"chain" logs: trunk, then a leaf resumed from a trunk checkpoint, then
bench output). `parse_log` therefore returns a list of `Segment`s, one
per invocation, each with its own run-level fields and update records.
The parser tolerates every era in the archive: pre-v27 logs have no
`ev`, pre-v30 no `acc`, pre-v35 no `accL/accR`, pre-v38 no `return`,
pre-af50n no `[kl]` line, and the event lines appear only on the eras
that had the counter. Missing fields are simply absent from the record
(never invented). Values the trainer printed as `nan` are absent too.

stdlib only — the backfill runs under the system python3 (no torch).
"""
from __future__ import annotations

import ast
import json
import math
import re
import shlex

# ----------------------------------------------------------------- schema
# Per-update fields, in the order they are emitted. Kept flat so the
# dashboard can address any of them by name.
UPDATE_FIELDS = (
    "upd", "n_upd", "steps", "t", "elapsed", "sps",
    "acc", "accL", "accR", "return", "ep_return", "ent", "ev",
    "r1", "maxlr", "fp16_overflow", "fp16_scale",
    # [kl]
    "kl_first", "kl_last", "clipfrac_first", "clipfrac_last",
    "gnorm", "clipc",
    # parameter norms (2026-09-12): L2 over the actor / critic params
    "pnorm_actor", "pnorm_critic",
    "ent_spine", "ent_armL", "ent_armR", "ent_legL", "ent_legR",
    "ent_min", "ent_min_head", "entfull", "afsd", "clipfrac_head_last",
    # events
    "falls", "tilts", "lifts", "lifts_L", "lifts_R",
    "flail", "flail_L", "flail_R", "flail_ticks",
    "limitrest", "limitrest_legs", "limitrest_spine", "limitrest_arms",
    # [activation] (2026-09-13): mean commanded |gear fraction| summed
    # over the 29 joints per env-tick (the retired v23 iso base) + groups
    "activation", "activation_spine", "activation_arms", "activation_legs",
    # [torque] (env v43, 2026-09-14): mean applied sum |tau| over the 29
    # joints per env-tick in N*m (the --energy-coef base) + groups
    "torque", "torque_spine", "torque_arms", "torque_legs",
    # env v41 [contact]: mean self-collision impulse per env-tick (N.s)
    # and the fraction of env-ticks with any body-on-body contact
    "contact", "contact_frac",
    "pd_net", "envelope",
    # [diag]
    "diag_gradcos", "diag_gpi", "diag_gv", "diag_dormant",
    "diag_srank", "diag_srank_dim", "diag_psrank", "diag_groups",
    # [bench-mem]
    "vram_resident", "vram_rollout_peak", "vram_update_peak",
    # [ckpt]
    "ckpt", "peak_vram",
)

# Entropy groups in the trainer's HEAD_GROUPS order (policy.HEAD_GROUPS:
# spine 3, armL 7, armR 7, legL 6, legR 6). Duplicated here so this
# module stays torch-free; test_metrics_log pins it against the trainer.
HEAD_GROUPS = (("spine", 0, 3), ("armL", 3, 10), ("armR", 10, 17),
               ("legL", 17, 23), ("legR", 23, 29))

# Event counters that are structurally 0 when their line is absent on
# an era that prints them (the trainer prints `[fall]` only when >0).
COUNT_EVENTS = ("falls", "tilts", "lifts", "pd_net", "envelope")


def _num(v):
    """float or None (nan/inf -> None; JSON cannot carry them)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return float(f"{f:.7g}")  # 7 significant digits: print precision, small files


def _int(v):
    try:
        return int(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def entropy_groups(ent_head):
    """(groups dict, min head index, min head entropy, 29-head sum) —
    the same arithmetic as train_batch.entropy_report, torch-free."""
    eh = [float(x) for x in ent_head]
    groups = {nm: sum(eh[a:b]) / (b - a) for nm, a, b in HEAD_GROUPS
              if b <= len(eh)}
    lo = min(range(len(eh)), key=eh.__getitem__)
    return groups, lo, eh[lo], sum(eh)


def update_record(update, num_updates, global_step, *, t=None,
                  elapsed=None, rate=None, acc=None, acc_l=None,
                  acc_r=None, ret=None, recent=None, ent=None, ev=None,
                  r1=None, maxlr=None, fp16_overflow=None,
                  fp16_scale=None, kl_first=None, kl_last=None,
                  cf_first=None, cf_last=None, gnorm=None, clipc=None,
                  ent_head=None, af_ratio=None, events=None,
                  diag=None, ckpt=None, peak_vram=None, cfh_last=None,
                  pnorm_actor=None, pnorm_critic=None, kl_stop=None):
    """The trainer's formatter: one per-update dict from host values.

    `update` is the 1-based index the `update` line prints. `events`
    is a dict of the event-line values (`falls`, `tilts`, `lifts`,
    `lifts_L`, ... — see UPDATE_FIELDS). None / nan values are
    dropped so the line carries only what this era measured."""
    rec = {"upd": int(update), "n_upd": int(num_updates),
           "steps": int(global_step)}
    if t is not None:
        rec["t"] = round(float(t), 3)
    rec["elapsed"] = _num(elapsed)
    rec["sps"] = _num(rate)
    rec["acc"] = _num(acc)
    rec["accL"] = _num(acc_l)
    rec["accR"] = _num(acc_r)
    rec["return"] = _num(ret)
    rec["ep_return"] = _num(recent)
    rec["ent"] = _num(ent)
    rec["ev"] = _num(ev)
    rec["r1"] = _num(r1)
    rec["maxlr"] = _num(maxlr)
    if fp16_overflow:
        rec["fp16_overflow"] = int(fp16_overflow)
    rec["fp16_scale"] = _num(fp16_scale)
    rec["kl_first"] = _num(kl_first)
    rec["kl_last"] = _num(kl_last)
    rec["clipfrac_first"] = _num(cf_first)
    rec["clipfrac_last"] = _num(cf_last)
    rec["gnorm"] = _num(gnorm)
    rec["clipc"] = _num(clipc)
    rec["pnorm_actor"] = _num(pnorm_actor)
    rec["pnorm_critic"] = _num(pnorm_critic)
    if ent_head:
        groups, lo, lo_v, total = entropy_groups(ent_head)
        for nm, v in groups.items():
            rec["ent_" + nm] = _num(v)
        rec["ent_min"] = _num(lo_v)
        rec["ent_min_head"] = int(lo)
        rec["entfull"] = _num(total)
    rec["afsd"] = _num(af_ratio)
    rec["clipfrac_head_last"] = _num(cfh_last)   # --head-clip only
    if kl_stop is not None:
        rec["kl_stop_mb"] = int(kl_stop)          # --target-kl only
    for k, v in (events or {}).items():
        if k in ("flail", "flail_L", "flail_R", "limitrest",
                 "limitrest_legs", "limitrest_spine", "limitrest_arms",
                 "activation", "activation_spine", "activation_arms",
                 "activation_legs", "torque", "torque_spine",
                 "torque_arms", "torque_legs", "contact", "contact_frac"):
            rec[k] = _num(v)
        else:
            rec[k] = _int(v)
    for k, v in (diag or {}).items():
        rec["diag_" + k] = v
    if ckpt:
        rec["ckpt"] = str(ckpt)
    rec["peak_vram"] = _num(peak_vram)
    return {k: v for k, v in rec.items() if v is not None}


def dumps_record(rec):
    """One compact JSON line (no trailing newline), None fields dropped."""
    clean = {}
    for k, v in rec.items():
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        clean[k] = v
    return json.dumps(clean, separators=(",", ":"), allow_nan=False)


# ------------------------------------------------------------------ parser
_RE_UPDATE = re.compile(r"^update\s+(\d+)/(\d+)\s+steps\s+([\d,]+)")
_RE_KV = {
    "acc": re.compile(r"(?<![\w_])acc (\S+)"),
    "accL": re.compile(r"(?<![\w_])accL (\S+)"),
    "accR": re.compile(r"(?<![\w_])accR (\S+)"),
    "return": re.compile(r"(?<![\w_])return\s+(\S+)"),
    "ep_return": re.compile(r"ep_return\(last100\)\s+(\S+)"),
    "ent": re.compile(r"(?<![\w_])ent (\S+)"),
    "ev": re.compile(r"(?<![\w_])ev (\S+)"),
    "r1": re.compile(r"(?<![\w_])r1 (\S+)"),
    "maxlr": re.compile(r"(?<![\w_])maxlr (\S+)"),
    "fp16_scale": re.compile(r"(?<![\w_])scale (\d+)"),
    # `pn <actor>/<critic>` (2026-09-12): parameter L2 norms
    "pnorm_actor": re.compile(r"(?<![\w_])pn ([\d.]+)/"),
    "pnorm_critic": re.compile(r"(?<![\w_])pn [\d.]+/([\d.]+)"),
}
_RE_SPS = re.compile(r"([\d,]+) steps/s\s+([\d.]+)s")
_RE_OVERFLOW = re.compile(r"\[fp16\] (\d+) overflow step")

_RE_KL = re.compile(
    r"^\[kl\] upd\s+(\d+)\s+kl\(mb1/mbN\) (\S+)/(\S+)\s+clipfrac (\S+)/(\S+)"
    r"\s+gnorm (\S+)\s+clipc (\S+)\s+ent/head (.*?)\s+min (\S+) \(h(\d+)\)"
    r"\s+entfull (\S+)(?:\s+afsd (\S+))?")
_RE_FALL = re.compile(r"^\[fall\] (\d+) fall termination")
_RE_TILT = re.compile(r"^\[tilt\] (\d+) tilt termination")
_RE_LIFT = re.compile(
    r"^\[lift\] (\d+) foot-lift event\(s\)(?:.*?\(L (\d+) / R (\d+))?")
_RE_FLAIL = re.compile(
    r"^\[flail\] (\S+) rad/idle-tick \((?:L (\S+) / R (\S+); )?(\d+) "
    r"note-free")
_RE_LIMITREST = re.compile(
    r"^\[limitrest\] (\S+) N\*m\*s/tick \(legs (\S+) / spine (\S+) / "
    r"arms (\S+)\)")
_RE_ACTIVATION = re.compile(
    r"^\[activation\] (\S+) /env-tick \(spine (\S+) / arms (\S+) / "
    r"legs (\S+)\)")
_RE_TORQUE = re.compile(
    r"^\[torque\] (\S+) N\*m/env-tick \(spine (\S+) / arms (\S+) / "
    r"legs (\S+)\)")
_RE_CONTACT = re.compile(
    r"^\[contact\] (\S+) N\*s/tick \((\S+)% of env-ticks in self-contact\)")
_RE_PDNET = re.compile(r"^\[pd-net\] (\d+) f32 pivot-floor")
_RE_ENVELOPE = re.compile(r"^\[envelope\] (\d+) root-escape")
_RE_DIAG = re.compile(
    r"^\[diag\] upd (\d+)\s+gradcos (\S+)\s+\|g_pi\| (\S+)\s+\|g_v\| (\S+)"
    r"\s+dormant (\S+)\s+srank (\d+)/(\d+)(?:\s+psrank (\d+))?"
    r"\s+pi/v: (.*)$")
_RE_DIAG_GROUP = re.compile(r"(\S+) (\S+)/(\S+)")
_RE_BENCHMEM = re.compile(
    r"^\[bench-mem\] upd (\d+): resident at rollout start (\S+)\s+rollout "
    r"peak (\S+)\s+update peak (\S+) GiB")
_RE_CKPT = re.compile(r"^\[ckpt\] (\S+)\s+peak-vram (\S+) GiB")
_RE_CKPT_UPD = re.compile(r"upd(\d+)\.pt$")

_RE_BANNER = re.compile(
    r"^=== batched PPO: (\d+) envs x (\d+) horizon = (\d+) steps/update, "
    r"minibatch (\d+), device (\S+?)(?:, adv-filter top \|A\| "
    r"(\d+)/(\d+).*?)?(?:, env-order (on|off))?"
    r"(?:, critic-stream (on|off))? ===")
_RE_POOL = re.compile(r"^\[ranked-pool\] (\d+) segments on (\S+) "
                      r"\(C=(\d+), (\d+) MB\)")
_RE_RESUMED = re.compile(
    r"^RESUMED (\S+) at update (\d+)/(\d+) \(step ([\d,]+)")
_RE_WSD_TRUNK = re.compile(
    r"^WSD TRUNK: warmup (\d+) updates -> constant lr ([-\d.e+]+)")
_RE_WSD_EXT = re.compile(
    r"^WSD TRUNK EXTENDED: ([\d,]+) -> ([\d,]+) updates")
_RE_WSD_LEAF = re.compile(
    r"^WSD LEAF: branching from trunk update ([\d,]+)")
_RE_ARCH = re.compile(r"^policy arch (.*?) — ([\d,]+) params")
_RE_WARM = re.compile(r"^warm-started from (\S+)")
_RE_THROUGHPUT = re.compile(
    r"^THROUGHPUT: ([\d,]+) steps/s total \| rollout (\d+)% update (\d+)%")
_RE_TP_SPLIT = re.compile(r"\[([\d.]+)s / ([\d.]+)s over (\d+) upd\]")
_RE_TP_SHAPE = re.compile(r"N=(\d+) T=(\d+)")
_RE_TP_VRAM = re.compile(r"peak-vram ([\d.]+) GiB")
_RE_TP_VRAM_SPLIT = re.compile(
    r"\(rollout ([\d.]+) / update ([\d.]+); resident at rollout start "
    r"([\d.]+)\)")
_RE_TIMECAP = re.compile(r"^Reached --time-cap ([\d.]+)s")


class Segment:
    """One trainer invocation inside a log."""

    def __init__(self, index):
        self.index = index
        self.run = {}           # run-level fields
        self.updates = []       # per-update records, in print order
        self.by_upd = {}        # upd -> record
        self._pending = {}      # event fields for the next update line
        self._diag = {}         # upd -> diag dict (printed before update)
        self._benchmem = {}     # upd -> vram dict
        self._seen = set()      # event fields seen anywhere in the segment
        self.line_counts = {}

    def _count(self, key):
        self.line_counts[key] = self.line_counts.get(key, 0) + 1

    def finish(self):
        """Post-pass: structural zeros for count events, attach the
        pre-printed `[diag]`/`[bench-mem]` values."""
        for rec in self.updates:
            for k in COUNT_EVENTS:
                if k in self._seen and k not in rec:
                    rec[k] = 0
            d = self._diag.pop(rec["upd"], None)
            if d:
                for k, v in d.items():
                    rec["diag_" + k] = v
            b = self._benchmem.pop(rec["upd"], None)
            if b:
                rec.update(b)
        r = self.run
        if self.updates:
            first, last = self.updates[0], self.updates[-1]
            r["start_update"] = first["upd"]
            r["last_update"] = last["upd"]
            r["updates"] = len(self.updates)
            r["samples"] = last["steps"]
            r["elapsed"] = last.get("elapsed")
        r.setdefault("status", "partial")
        r["line_counts"] = dict(self.line_counts)
        return self


def parse_log(text, t_end=None):
    """Parse a trainer stdout log into [Segment].

    `t_end` (epoch seconds, e.g. the log file's mtime) stamps every
    update with `t` = t_end - (elapsed_last - elapsed) of its segment,
    the best wall-clock reconstruction a timestamp-free log allows
    (exact for the last update of a live log, drifting for earlier
    ones by however long the process idled between the last print and
    the file's mtime). Only the LAST segment of a log gets `t`; earlier
    segments' ends are unknown."""
    segs = []
    seg = None

    def new_seg():
        nonlocal seg
        if seg is not None:
            segs.append(seg.finish())
        seg = Segment(len(segs))
        return seg

    seg = new_seg()
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("argv: "):
            if seg.updates or "argv" in seg.run:
                new_seg()
            seg.run["argv"] = shlex.split(line[6:]) \
                if line[6:] != "(bare)" else []
            continue
        if line.startswith("config: "):
            if seg.updates:
                new_seg()
            try:
                seg.run["config"] = ast.literal_eval(line[8:])
            except (ValueError, SyntaxError):
                seg.run["config_raw"] = line[8:]
            continue
        m = _RE_BANNER.match(line)
        if m:
            if seg.updates:
                new_seg()
            seg.run.update({
                "num_envs": int(m.group(1)), "horizon": int(m.group(2)),
                "batch": int(m.group(3)), "minibatch": int(m.group(4)),
                "device": m.group(5)})
            if m.group(6):
                seg.run["adv_filter_keep"] = int(m.group(6))
                seg.run["adv_filter_of"] = int(m.group(7))
            if m.group(8):
                seg.run["env_order"] = m.group(8) == "on"
            if m.group(9):
                seg.run["critic_stream"] = m.group(9) == "on"
            seg._count("banner")
            continue
        m = _RE_UPDATE.match(line)
        if m:
            rec = {"upd": int(m.group(1)), "n_upd": int(m.group(2)),
                   "steps": _int(m.group(3))}
            ms = _RE_SPS.search(line)
            if ms:
                rec["sps"] = _num(ms.group(1).replace(",", ""))
                rec["elapsed"] = _num(ms.group(2))
            for k, rx in _RE_KV.items():
                mk = rx.search(line)
                if mk:
                    v = _num(mk.group(1))
                    if v is not None:
                        rec[k] = v
            mo = _RE_OVERFLOW.search(line)
            if mo:
                rec["fp16_overflow"] = int(mo.group(1))
            rec.update(seg._pending)
            seg._pending = {}
            seg.updates.append(rec)
            seg.by_upd[rec["upd"]] = rec
            seg._count("update")
            continue
        m = _RE_KL.match(line)
        if m:
            upd = int(m.group(1))
            rec = seg.by_upd.get(upd)
            if rec is None:
                # [kl] always follows its update line; tolerate anyway
                rec = seg._pending
            rec["kl_first"] = _num(m.group(2))
            rec["kl_last"] = _num(m.group(3))
            rec["clipfrac_first"] = _num(m.group(4))
            rec["clipfrac_last"] = _num(m.group(5))
            rec["gnorm"] = _num(m.group(6))
            rec["clipc"] = _num(m.group(7))
            toks = m.group(8).split()
            for i in range(0, len(toks) - 1, 2):
                rec["ent_" + toks[i]] = _num(toks[i + 1])
            rec["ent_min"] = _num(m.group(9))
            rec["ent_min_head"] = int(m.group(10))
            rec["entfull"] = _num(m.group(11))
            if m.group(12):
                rec["afsd"] = _num(m.group(12))
            seg._count("kl")
            continue
        m = _RE_FALL.match(line)
        if m:
            seg._pending["falls"] = int(m.group(1))
            seg._seen.add("falls")
            seg._count("fall")
            continue
        m = _RE_TILT.match(line)
        if m:
            seg._pending["tilts"] = int(m.group(1))
            seg._seen.add("tilts")
            seg._count("tilt")
            continue
        m = _RE_LIFT.match(line)
        if m:
            seg._pending["lifts"] = int(m.group(1))
            if m.group(2):
                seg._pending["lifts_L"] = int(m.group(2))
                seg._pending["lifts_R"] = int(m.group(3))
            seg._seen.add("lifts")
            seg._count("lift")
            continue
        m = _RE_FLAIL.match(line)
        if m:
            seg._pending["flail"] = _num(m.group(1))
            if m.group(2):
                seg._pending["flail_L"] = _num(m.group(2))
                seg._pending["flail_R"] = _num(m.group(3))
            seg._pending["flail_ticks"] = int(m.group(4))
            seg._count("flail")
            continue
        m = _RE_LIMITREST.match(line)
        if m:
            seg._pending["limitrest"] = _num(m.group(1))
            seg._pending["limitrest_legs"] = _num(m.group(2))
            seg._pending["limitrest_spine"] = _num(m.group(3))
            seg._pending["limitrest_arms"] = _num(m.group(4))
            seg._count("limitrest")
            continue
        m = _RE_ACTIVATION.match(line)
        if m:
            seg._pending["activation"] = _num(m.group(1))
            seg._pending["activation_spine"] = _num(m.group(2))
            seg._pending["activation_arms"] = _num(m.group(3))
            seg._pending["activation_legs"] = _num(m.group(4))
            seg._count("activation")
            continue
        m = _RE_TORQUE.match(line)
        if m:
            seg._pending["torque"] = _num(m.group(1))
            seg._pending["torque_spine"] = _num(m.group(2))
            seg._pending["torque_arms"] = _num(m.group(3))
            seg._pending["torque_legs"] = _num(m.group(4))
            seg._count("torque")
            continue
        m = _RE_CONTACT.match(line)
        if m:
            seg._pending["contact"] = _num(m.group(1))
            pct = _num(m.group(2))
            seg._pending["contact_frac"] = (pct / 100.0
                                            if pct is not None else None)
            seg._count("contact")
            continue
        m = _RE_PDNET.match(line)
        if m:
            seg._pending["pd_net"] = int(m.group(1))
            seg._seen.add("pd_net")
            continue
        m = _RE_ENVELOPE.match(line)
        if m:
            seg._pending["envelope"] = int(m.group(1))
            seg._seen.add("envelope")
            continue
        m = _RE_DIAG.match(line)
        if m:
            d = {"gradcos": _num(m.group(2)), "gpi": _num(m.group(3)),
                 "gv": _num(m.group(4)), "dormant": _num(m.group(5)),
                 "srank": int(m.group(6)), "srank_dim": int(m.group(7))}
            if m.group(8):
                d["psrank"] = int(m.group(8))
            d["groups"] = {g: [_num(a), _num(b)] for g, a, b in
                           _RE_DIAG_GROUP.findall(m.group(9))}
            seg._diag[int(m.group(1))] = d
            seg._count("diag")
            continue
        m = _RE_BENCHMEM.match(line)
        if m:
            seg._benchmem[int(m.group(1))] = {
                "vram_resident": _num(m.group(2)),
                "vram_rollout_peak": _num(m.group(3)),
                "vram_update_peak": _num(m.group(4))}
            continue
        m = _RE_CKPT.match(line)
        if m:
            name, vram = m.group(1), _num(m.group(2))
            mu = _RE_CKPT_UPD.search(name)
            rec = seg.by_upd.get(int(mu.group(1))) if mu else None
            if rec is None and seg.updates:
                rec = seg.updates[-1]
            if rec is not None:
                rec["ckpt"] = name
                rec["peak_vram"] = vram
            seg.run.setdefault("checkpoints", []).append(name)
            seg.run["peak_vram"] = vram
            seg._count("ckpt")
            continue
        m = _RE_POOL.match(line)
        if m:
            seg.run["pool_segments"] = int(m.group(1))
            seg.run["pool_mb"] = int(m.group(4))
            continue
        m = _RE_RESUMED.match(line)
        if m:
            seg.run["resume_from"] = m.group(1)
            seg.run["resume_update"] = int(m.group(2))
            seg.run["num_updates"] = int(m.group(3))
            continue
        m = _RE_WSD_TRUNK.match(line)
        if m:
            seg.run["role"] = "trunk"
            seg.run["warmup_updates"] = int(m.group(1))
            seg.run["lr"] = _num(m.group(2))
            continue
        m = _RE_WSD_EXT.match(line)
        if m:
            seg.run["role"] = "trunk"
            seg.run["extended_from"] = _int(m.group(1))
            seg.run["num_updates"] = _int(m.group(2))
            continue
        m = _RE_WSD_LEAF.match(line)
        if m:
            seg.run["role"] = "leaf"
            seg.run["leaf_from"] = _int(m.group(1))
            continue
        m = _RE_ARCH.match(line)
        if m:
            seg.run["n_params"] = _int(m.group(2))
            try:
                seg.run["arch"] = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                seg.run["arch"] = m.group(1)
            continue
        m = _RE_WARM.match(line)
        if m:
            seg.run["warm_started_from"] = m.group(1)
            continue
        m = _RE_THROUGHPUT.match(line)
        if m:
            seg.run["status"] = "finished"
            seg.run["sps_total"] = _num(m.group(1).replace(",", ""))
            seg.run["rollout_pct"] = int(m.group(2))
            seg.run["update_pct"] = int(m.group(3))
            ms = _RE_TP_SPLIT.search(line)
            if ms:
                seg.run["t_rollout"] = _num(ms.group(1))
                seg.run["t_update"] = _num(ms.group(2))
            mv = _RE_TP_VRAM.search(line)
            if mv:
                seg.run["peak_vram"] = _num(mv.group(1))
            mv = _RE_TP_VRAM_SPLIT.search(line)
            if mv:
                seg.run["peak_vram_rollout"] = _num(mv.group(1))
                seg.run["peak_vram_update"] = _num(mv.group(2))
                seg.run["vram_resident"] = _num(mv.group(3))
            seg._count("throughput")
            continue
        m = _RE_TIMECAP.match(line)
        if m:
            seg.run["time_cap_hit"] = _num(m.group(1))
            continue
        if line.startswith("Traceback (most recent call last)"):
            seg.run["status"] = "crashed"
            continue
        if line.startswith("--fp16 "):
            seg.run["fp16"] = line.split(":")[0][7:].strip()
            continue
    segs.append(seg.finish())
    segs = [s for s in segs if s.updates or s.run.get("argv")]
    for i, s in enumerate(segs):
        s.index = i
        if s.run.get("argv"):
            s.run["run_dir"] = _argv_value(s.run["argv"], "--run-dir")
            s.run["seed"] = _int(_argv_value(s.run["argv"], "--seed"))
    if t_end is not None and segs and segs[-1].updates:
        last = segs[-1]
        e_last = last.updates[-1].get("elapsed")
        if e_last is not None:
            for rec in last.updates:
                e = rec.get("elapsed")
                if e is not None:
                    rec["t"] = round(t_end - (e_last - e), 3)
            last.run["t_end"] = t_end
    return segs


def _argv_value(argv, flag):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def parse_log_file(path, t_end="mtime"):
    import os
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if t_end == "mtime":
        t_end = os.path.getmtime(path)
    return parse_log(text, t_end=t_end)


def merge_segments(segments):
    """Fold several invocations of ONE run (trunk, then its resumed
    extensions; or a re-run from scratch) into one update list.

    Segments are applied in order; a segment that (re)starts at update
    k discards every earlier record with upd >= k — a resume replays
    from that checkpoint, so the earlier tail is the stale branch.
    Returns (records sorted by upd, per-segment run dicts)."""
    recs = {}
    runs = []
    for s in segments:
        if not s.updates:
            continue
        start = s.updates[0]["upd"]
        for k in [k for k in recs if k >= start]:
            del recs[k]
        for r in s.updates:
            recs[r["upd"]] = r
        runs.append(s.run)
    return [recs[k] for k in sorted(recs)], runs


def load_metrics(path):
    """Read a metrics.jsonl back into a list of dicts (bad lines skipped)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
