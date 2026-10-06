"""Replay format v2 — the compact binary replay (2026-10-02).

The ONLY replay format (the v1 gzip-JSONL format, one JSON object per sim
tick, was retired with this one). A file `<name>.rpl.gz` is gzip (so the
browser inflates it natively with DecompressionStream) around:

    MAGIC  b"BARPLAY2"                      8 bytes
    u32    JSON length (little-endian)       4 bytes
    JSON   {"header": {...}, "ticks": T, "blocks": [...]}   utf-8
    zero padding to a multiple of 8
    the blocks, in "blocks" order, each zero-padded to a multiple of 8

`header` is record_header()'s object (rust_env.py) plus `"format": 2` and
the `body` tables below; `ticks` = T, the number of SIM ticks (tick k spans
[k, k + 1] / tick_hz s). Each block entry is {name, dtype, rows, len} (+
scale / delta / pred): `rows` x `len` values stored row-major, i.e. one
row per per-tick COLUMN (column-major over ticks). dtype "u8" is raw
bytes; "z32" is a signed int32 value v stored zigzagged (2v, or -2v - 1)
as 4 byte planes (all lowest bytes, then all second bytes, ...) — small
deltas leave whole planes zero, which gzip squeezes. A z32 block holds
round(x * scale); `delta: 1` means each row stores successive differences
(the first value against 0); `pred: "omega"` (q only) means the joint rows
7..35 store the delta MINUS the trapezoid prediction from the decoded
omega block (integer-exact, see _omega_pred):

    q        z32 36 x T   1e-4 (m / rad), delta, pred omega (joint rows)
    omega    z32 29 x T   1e-2 rad/s (the gauges print one decimal)
    twist    z32  6 x T   1e-3
    feet     z32  6 x T   1e-3 N.s  (L xyz, R xyz ground reaction impulse)
    reward   z32  1 x T   1e-4 (the v1 record's 4 decimals, exactly)
    action   u8  29 x T   gear ids
    contact_n u8  1 x T   contacts per tick; contact_pair u8 (N),
             contact_p z32 3 x N 1e-4 m, contact_j z32 1 x N 1e-4 N.s
    event_n  u8   1 x T   events per tick; event_code u8 (M), event_note
             z32 1 x M (-1 = no note: wall ticks), event_val z32 3 x M
             1e-2 (points, pre, acc)

Not stored, rebuilt by the decoders (Python here, JS viz/app/src/lib/
replay.js decodeReplay):
  - t = 0..T-1 and time = t / tick_hz (the sim clock at the tick's start,
    to float noise; v1 stamped it rounded to 1e-5 s);
  - sabers (per-hand wrist + blade dir) and head ([pos, wxyz quat]) by
    full-body FK of q (root quaternion renormalized) — verified at write
    time against the sim's own values (<= 1 mm, <= 0.1 deg; measured on
    the published replays: 0.3 mm / 0.02 deg). A recording that fails the
    check carries them as blocks instead (sabers z32 12 x T 1e-4, head z32
    7 x T 1e-6);
  - activation [total, spine, arms, legs] = sum |gear fraction| and torque
    (same groups) = sum |fraction x tau_max| from action and header.body
    ({tau_max: 29 effective per-joint limits (TAU_MAX x the episode's
    tau_scale), gear_frac: 29 rows of per-id signed fractions, omega_cap:
    29 joint speed caps, rad/s (since 2026-10-03; display only)}) — the
    body this replay was played with, so it is right for any body (main's
    v63x2 body or body/original's). Writers before 2026-10-03 put the
    uniform gear table in every row; on the wrist-body runs (v62-v63x2,
    whose bend joints took 2x fine gears) that mismatched, so those files
    store both columns (the viewer's replayBody restores their per-joint
    table for its joint gauges). Also verified at write time
    (<= 1e-4 / 1e-3 N.m, the v1 record's precision); on a mismatch they
    are stored as blocks (activation z32 4 x T 1e-4, torque z32 4 x T
    1e-3).

Size (the 80 published v60-v63 replays re-encoded, 2026-10-02): 96 B per
60 Hz tick and 112 per 30 Hz tick (v1: 342 / 353), 3.4x smaller overall;
writing takes ~18 us/tick incl. the derived-field checks (v1 ~94).
python/tests/test_replay_format.py holds the round trip.
"""

from __future__ import annotations

import gzip
import json
import struct

import numpy as np

from .constants import (EVENT_REASONS, N_JOINTS, OMEGA_CAP, TAU_AXIS_OF_JOINT,
                        TAU_MAX, TENDON_HANDS, TENDON_R0,
                        TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV,
                        joint_gear_table)
from .kinematics import (FRAMES, F_HEAD, F_L_HAND, F_L_WRIST, F_R_HAND,
                         F_R_WRIST)

MAGIC = b"BARPLAY2"
FORMAT = 2
EXT = ".rpl.gz"

# VecBeatSim recorder frame (take_record_env "frame", lanes per tick; the
# REC_* layout in rust/src/lib.rs)
REC_LANES = 105
L_Q, L_TWIST, L_OMEGA, L_SABERS, L_HEAD = 0, 36, 42, 71, 83
L_FEET, L_ACT, L_TORQUE, L_REWARD = 90, 96, 100, 104

# quantization scales (value = int / scale)
S_Q, S_OMEGA, S_TWIST, S_FEET, S_REWARD = 10000, 100, 1000, 1000, 10000
S_CONTACT, S_EVENT = 10000, 100
S_SABERS, S_HEAD, S_ACT, S_TORQUE = 10000, 1000000, 10000, 1000

# write-time checks of the derived fields (else they are stored)
TOL_POS = 1e-3                     # m: wrists, head
TOL_ANG = np.radians(0.1)          # rad: blade dirs, head orientation
TOL_ACT, TOL_TORQUE = 1e-4, 1e-3

GROUP_OF_JOINT = np.array([0 if j < 3 else 1 if j < 17 else 2
                           for j in range(N_JOINTS)])


def body_tables(gears, tau_scale):
    """header.body for a sim with `gears` action ids and the 16 per-axis
    tau_scale values: the effective per-joint torque limits and per-joint
    signed gear fractions this build's sim applies (constants.TAU_MAX /
    joint_gear_table mirror rust lib.rs Body::tick, incl. the bend
    joints' 2x fine gears). Per joint, so a decoder derives torque for
    whichever body wrote the file: replays written before 2026-10-03
    carry the body/original tables (1.06 bend, uniform gears). Row j has
    joint j's own id count (per-joint gear counts, 2026-10-04: a body
    branch's extra gears make some rows longer — ragged).

    "tendon" (exp/wrist-tendon, 2026-10-05): the passive wrist tendon
    this body ran with — {hands [(bend, dev)] x2, tau_peak_bend,
    tau_peak_dev (N.m, fixed), r0}. Its torque is NOT in the derived
    torque (not muscle effort); a header without the key is a tendon-free
    body."""
    s = [float(tau_scale[TAU_AXIS_OF_JOINT[j]]) for j in range(N_JOINTS)]
    return {"tau_max": [TAU_MAX[j] * s[j] for j in range(N_JOINTS)],
            "gear_frac": [[float(v) for v in joint_gear_table(int(gears), j)]
                          for j in range(N_JOINTS)],
            "omega_cap": [float(c) for c in OMEGA_CAP],
            "tendon": {"hands": [list(h) for h in TENDON_HANDS],
                       "tau_peak_bend": float(TENDON_TAU_PEAK_BEND),
                       "tau_peak_dev": float(TENDON_TAU_PEAK_DEV),
                       "r0": float(TENDON_R0)}}


def derive_effort(action, body):
    """(activation, torque), each (T, 4) [total, spine, arms, legs], from
    action (T, 29) gear ids and header.body."""
    rows = body["gear_frac"]                                     # 29 rows
    # rows may be ragged (per-joint gear counts): pad with 0 (coast) to
    # the longest — ids past a joint's count are never recorded
    frac = np.zeros((N_JOINTS, max(len(r) for r in rows)))
    for j, r in enumerate(rows):
        frac[j, :len(r)] = r                                     # (29, K)
    tau = np.asarray(body["tau_max"], dtype=np.float64)
    a = np.asarray(action, dtype=np.int64)
    f = frac[np.arange(N_JOINTS)[None, :], a]                     # (T, 29)
    act, trq = np.abs(f), np.abs(f * tau)
    out = []
    for x in (act, trq):
        g = np.zeros((len(a), 4))
        g[:, 0] = x.sum(1)
        for k in range(3):
            g[:, k + 1] = x[:, GROUP_OF_JOINT == k].sum(1)
        out.append(g)
    return out[0], out[1]


# ---- full-body FK, vectorized over ticks (mirror of replay.js bodyFK) ----

def _rot_axis(axis, ang):
    s, c = np.sin(ang), np.cos(ang)
    k = 1.0 - c
    x, y, z = axis
    r = np.empty(ang.shape + (3, 3))
    r[:, 0, 0] = c + k * x * x
    r[:, 0, 1] = k * x * y - s * z
    r[:, 0, 2] = k * x * z + s * y
    r[:, 1, 0] = k * x * y + s * z
    r[:, 1, 1] = c + k * y * y
    r[:, 1, 2] = k * y * z - s * x
    r[:, 2, 0] = k * x * z - s * y
    r[:, 2, 1] = k * y * z + s * x
    r[:, 2, 2] = c + k * z * z
    return r


def _quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3))
    r[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                        2 * (x * z + w * y)], 1)
    r[:, 1] = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                        2 * (y * z - w * x)], 1)
    r[:, 2] = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                        1 - 2 * (x * x + y * y)], 1)
    return r


def rot_to_quat(r):
    """wxyz of (T, 3, 3) rotations — rust dynamics/fk.rs rot_to_quat's
    branches (so the sign matches the recorded head quaternion)."""
    m00, m11, m22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    tr = m00 + m11 + m22
    out = np.empty((len(r), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        s0 = np.sqrt(tr + 1.0) * 2.0
        c0 = np.stack([0.25 * s0, (r[:, 2, 1] - r[:, 1, 2]) / s0,
                       (r[:, 0, 2] - r[:, 2, 0]) / s0,
                       (r[:, 1, 0] - r[:, 0, 1]) / s0], 1)
        s1 = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        c1 = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s1, 0.25 * s1,
                       (r[:, 0, 1] + r[:, 1, 0]) / s1,
                       (r[:, 0, 2] + r[:, 2, 0]) / s1], 1)
        s2 = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        c2 = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s2,
                       (r[:, 0, 1] + r[:, 1, 0]) / s2, 0.25 * s2,
                       (r[:, 1, 2] + r[:, 2, 1]) / s2], 1)
        s3 = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        c3 = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s3,
                       (r[:, 0, 2] + r[:, 2, 0]) / s3,
                       (r[:, 1, 2] + r[:, 2, 1]) / s3, 0.25 * s3], 1)
    b0 = tr > 0.0
    b1 = ~b0 & (m00 > m11) & (m00 > m22)
    b2 = ~b0 & ~b1 & (m11 > m22)
    b3 = ~b0 & ~b1 & ~b2
    for b, c in ((b0, c0), (b1, c1), (b2, c2), (b3, c3)):
        out[b] = c[b]
    return out


def fk_poses(q):
    """(sabers (T, 12) [L wrist, L blade, R wrist, R blade], head (T, 7)
    [pos, wxyz]) from q (T, 36), the root quaternion renormalized."""
    q = np.asarray(q, dtype=np.float64)
    quat = q[:, 3:7] / np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    r0, p0 = _quat_to_rot(quat), q[:, 0:3]
    rots, pos = [], []
    for parent, anchor, axis, coord, _m, _c in FRAMES:
        rl, pl = (r0, p0) if parent < 0 else (rots[parent], pos[parent])
        pos.append(pl + rl @ np.asarray(anchor))
        rots.append(rl @ _rot_axis(axis, q[:, 7 + coord])
                    if coord >= 0 else rl)
    sab = np.concatenate([pos[F_L_WRIST], rots[F_L_HAND][:, :, 2],
                          pos[F_R_WRIST], rots[F_R_HAND][:, :, 2]], 1)
    head = np.concatenate([pos[F_HEAD], rot_to_quat(rots[F_HEAD])], 1)
    return sab, head


def _dir_err(a, b):
    """Angle (rad) between unit-ish vectors, rows of (T, 3)."""
    c = np.linalg.norm(np.cross(a, b), axis=1)
    d = (a * b).sum(1)
    return np.arctan2(c, d)


def _quat_err(a, b):
    d = np.abs((a * b).sum(1)) / (np.linalg.norm(a, axis=1)
                                   * np.linalg.norm(b, axis=1))
    return 2.0 * np.arccos(np.minimum(1.0, d))


# ---- block codec ---------------------------------------------------------

def _zigzag_planes(v):
    """int64 array (any shape) -> z32 byte planes."""
    v = np.asarray(v, dtype=np.int64).ravel()
    if v.size and (v.max() > 2**31 - 1 or v.min() < -2**31):
        raise OverflowError("replay value out of int32 range")
    z = np.where(v >= 0, 2 * v, -2 * v - 1).astype("<u4")
    return z.view(np.uint8).reshape(-1, 4).T.tobytes()


def _unplanes(buf, n):
    z = np.frombuffer(buf, dtype=np.uint8, count=4 * n).reshape(4, n)
    z = np.ascontiguousarray(z.T).view("<u4").ravel().astype(np.int64)
    return np.where(z & 1, -((z + 1) >> 1), z >> 1)


def _omega_pred(om, c, hz):
    """Trapezoid prediction of the joint-angle deltas (q units) from the
    quantized omega rows (29, T): round-half-up((om[t] + om[t-1]) * c /
    (2 hz)), om[-1] = 0, 0 at t = 0 — in integers, so the JS decoder's
    Math.floor of the same exact quotient agrees bit for bit."""
    s = om.copy()
    s[:, 1:] += om[:, :-1]
    p = (c * s + hz) // (2 * hz)
    p[:, :1] = 0
    return p


class _Writer:
    def __init__(self):
        self.blocks, self.data = [], []

    def add(self, name, a, dtype="z32", scale=None, **kw):
        a = np.asarray(a)
        a = a.reshape(1, -1) if a.ndim == 1 else a
        ent = {"name": name, "dtype": dtype, "rows": int(a.shape[0]),
               "len": int(a.shape[1])}
        if scale is not None:
            ent["scale"] = scale
        ent.update(kw)
        if dtype == "u8":
            if a.size and (a.min() < 0 or a.max() > 255):
                raise OverflowError(f"replay block {name} out of u8 range")
            b = a.astype(np.uint8).tobytes()
        else:
            b = _zigzag_planes(a)
        self.blocks.append(ent)
        self.data.append(b)


def _quant(x, scale):
    return np.round(np.asarray(x, dtype=np.float64) * scale).astype(np.int64)


def _pad8(n):
    return b"\0" * (-n % 8)


def pack(meta, data):
    """MAGIC + JSON + blocks (uncompressed)."""
    js = json.dumps(meta, separators=(",", ":")).encode()
    out = [MAGIC, struct.pack("<I", len(js)), js, _pad8(12 + len(js))]
    for b in data:
        out += [b, _pad8(len(b))]
    return b"".join(out)


def unpack(raw):
    """(meta, [block bytes]) of an uncompressed v2 payload."""
    if raw[:8] != MAGIC:
        raise ValueError("not a beat-arena replay (format 2)")
    (n,) = struct.unpack_from("<I", raw, 8)
    meta = json.loads(raw[12:12 + n])
    off = 12 + n
    off += -off % 8
    data = []
    for b in meta["blocks"]:
        size = b["rows"] * b["len"] * (1 if b["dtype"] == "u8" else 4)
        data.append(raw[off:off + size])
        off += size + (-size % 8)
    return meta, data


def gz(raw, level=6):
    """gzip with no name and mtime 0 (byte-identical re-runs)."""
    return gzip.compress(raw, compresslevel=level, mtime=0)


def encode(header, rec, max_ticks=None, stats=None):
    """The uncompressed v2 payload of a recorder capture.

    header: record_header()'s object ({"header": {...}}, with `body`);
    rec: a VecBeatSim recorder capture (take_record_env) or
    RustBeatEnv's equivalent — `frame` (T * 105 lanes), `action`,
    contact_* / event_* with prefix offsets, `tick`. max_ticks: keep only
    the first N ticks. stats: optional dict, filled with the derived-
    field check results (max errors, which fields were stored)."""
    h = header["header"]
    lanes = int(rec["lanes"])
    assert lanes == REC_LANES, f"recorder frame has {lanes} lanes"
    T = len(rec["tick"])
    if max_ticks is not None:
        T = min(T, int(max_ticks))
    ticks = np.asarray(rec["tick"][:T])
    assert (ticks == np.arange(T)).all(), "recorder ticks are not 0..T-1"
    hz = int(h.get("tick_hz") or 30)
    fr = np.asarray(rec["frame"], dtype=np.float64)[:T * lanes].reshape(
        T, lanes)
    act = np.asarray(rec["action"][:T * N_JOINTS]).reshape(T, N_JOINTS)
    w = _Writer()

    om = _quant(fr[:, L_OMEGA:L_SABERS], S_OMEGA).T            # (29, T)
    w.add("omega", om, scale=S_OMEGA)
    qi = _quant(fr[:, L_Q:L_TWIST], S_Q).T                     # (36, T)
    d = np.diff(qi, axis=1, prepend=0)
    d[7:] -= _omega_pred(om, S_Q // S_OMEGA, hz)
    w.add("q", d, scale=S_Q, delta=1, pred="omega")
    w.add("twist", _quant(fr[:, L_TWIST:L_OMEGA], S_TWIST).T, scale=S_TWIST)
    w.add("feet", _quant(fr[:, L_FEET:L_ACT], S_FEET).T, scale=S_FEET)
    w.add("reward", _quant(fr[:, L_REWARD], S_REWARD), scale=S_REWARD)
    w.add("action", act.T, dtype="u8")

    # derived fields: drop them only when the decoders reproduce them
    st = {} if stats is None else stats
    q_dec = qi.T / S_Q
    sab, head = fk_poses(q_dec)
    rs, rh = fr[:, L_SABERS:L_HEAD], fr[:, L_HEAD:L_FEET]
    def mx(*errs):                   # max over arrays (0 when empty)
        return float(max(np.max(e, initial=0.0) for e in errs))

    st["wrist_m"] = mx(np.abs(sab[:, 0:3] - rs[:, 0:3]),
                       np.abs(sab[:, 6:9] - rs[:, 6:9]))
    st["blade_rad"] = mx(_dir_err(sab[:, 3:6], rs[:, 3:6]),
                         _dir_err(sab[:, 9:12], rs[:, 9:12]))
    st["head_m"] = mx(np.abs(head[:, 0:3] - rh[:, 0:3]))
    st["head_rad"] = mx(_quat_err(head[:, 3:7], rh[:, 3:7]))
    st["stored"] = []
    if st["wrist_m"] > TOL_POS or st["blade_rad"] > TOL_ANG:
        w.add("sabers", _quant(rs, S_SABERS).T, scale=S_SABERS)
        st["stored"].append("sabers")
    if st["head_m"] > TOL_POS or st["head_rad"] > TOL_ANG:
        w.add("head", _quant(rh, S_HEAD).T, scale=S_HEAD)
        st["stored"].append("head")
    da, dt = derive_effort(act, h["body"])
    st["activation"] = mx(np.abs(da - fr[:, L_ACT:L_TORQUE]))
    st["torque"] = mx(np.abs(dt - fr[:, L_TORQUE:L_REWARD]))
    if st["activation"] > TOL_ACT:
        w.add("activation", _quant(fr[:, L_ACT:L_TORQUE], S_ACT).T,
              scale=S_ACT)
        st["stored"].append("activation")
    if st["torque"] > TOL_TORQUE:
        w.add("torque", _quant(fr[:, L_TORQUE:L_REWARD], S_TORQUE).T,
              scale=S_TORQUE)
        st["stored"].append("torque")

    coff = np.asarray(rec["contact_off"][:T + 1], dtype=np.int64)
    n_c = int(coff[-1])
    w.add("contact_n", np.diff(coff), dtype="u8")
    w.add("contact_pair", np.asarray(rec["contact_pair"][:n_c]), dtype="u8")
    cval = np.asarray(rec["contact_val"], dtype=np.float64)[:4 * n_c]
    cval = cval.reshape(n_c, 4)
    w.add("contact_p", _quant(cval[:, 0:3], S_CONTACT).T, scale=S_CONTACT)
    w.add("contact_j", _quant(cval[:, 3], S_CONTACT), scale=S_CONTACT)
    eoff = np.asarray(rec["event_off"][:T + 1], dtype=np.int64)
    n_e = int(eoff[-1])
    w.add("event_n", np.diff(eoff), dtype="u8")
    w.add("event_code", np.asarray(rec["event_code"][:n_e]), dtype="u8")
    note = np.asarray(rec["event_note"][:n_e], dtype=np.int64)
    w.add("event_note", np.where(note >= 2**31, -1, note))
    ev = np.asarray(rec["event_val"], dtype=np.float64)[:3 * n_e]
    w.add("event_val", _quant(ev.reshape(n_e, 3), S_EVENT).T, scale=S_EVENT)

    meta = {"header": h, "ticks": T, "blocks": w.blocks}
    return pack(meta, w.data)


def write_replay(path, header, rec, *, gzip_level=6, max_ticks=None,
                 stats=None):
    """Write a v2 replay (gzip; no name, mtime 0: byte-identical re-runs)
    from a recorder capture; returns the tick count."""
    raw = encode(header, rec, max_ticks=max_ticks, stats=stats)
    with open(path, "wb") as f:
        f.write(gz(raw, gzip_level))
    T = len(rec["tick"])
    return T if max_ticks is None else min(T, int(max_ticks))


# ---- decoding ------------------------------------------------------------

def read_raw(path):
    """The uncompressed payload of a replay file (gzip or not)."""
    with open(path, "rb") as f:
        b = f.read()
    return gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b


def _block(meta, data, name):
    for ent, b in zip(meta["blocks"], data):
        if ent["name"] == name:
            n = ent["rows"] * ent["len"]
            if ent["dtype"] == "u8":
                a = np.frombuffer(b, dtype=np.uint8, count=n).astype(np.int64)
            else:
                a = _unplanes(b, n)
            return ent, a.reshape(ent["rows"], ent["len"])
    return None, None


def _vals(meta, data, name):
    """Block `name` as (rows, len) values (int / scale), None if absent."""
    ent, a = _block(meta, data, name)
    if ent is None:
        return None
    return a / ent["scale"] if "scale" in ent else a


def decode(raw):
    """Columnar decode of a v2 payload: a dict with `header`, `T` and
    numpy arrays — t, time (T,), q (T, 36), omega (T, 29), twist (T, 6),
    feet (T, 6), reward (T,), action (T, 29) int, sabers (T, 12), head
    (T, 7), activation / torque (T, 4), contact_off (T+1,), contact_pair,
    contact_p (N, 3), contact_j, event_off (T+1,), event_code,
    event_note (-1 = none), event_val (M, 3)."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    meta, data = unpack(raw)
    h, T = meta["header"], int(meta["ticks"])
    if h.get("format") != FORMAT:
        raise ValueError(f"replay format {h.get('format')} != {FORMAT}")
    hz = int(h.get("tick_hz") or 30)
    out = {"header": h, "T": T, "t": np.arange(T), "time": np.arange(T) / hz}
    e_om, om = _block(meta, data, "omega")
    e_q, qd = _block(meta, data, "q")
    if e_q.get("pred") == "omega":
        qd = qd.copy()
        qd[7:] += _omega_pred(om, e_q["scale"] // e_om["scale"], hz)
    if e_q.get("delta"):
        qd = np.cumsum(qd, axis=1)
    out["q"] = qd.T / e_q["scale"]
    out["omega"] = om.T / e_om["scale"]
    for name in ("twist", "feet", "sabers", "head", "activation", "torque"):
        v = _vals(meta, data, name)
        if v is not None:
            out[name] = v.T
    out["reward"] = _vals(meta, data, "reward")[0]
    out["action"] = _vals(meta, data, "action").T
    if "sabers" not in out or "head" not in out:
        sab, head = fk_poses(out["q"])
        out.setdefault("sabers", sab)
        out.setdefault("head", head)
    if "activation" not in out or "torque" not in out:
        da, dt = derive_effort(out["action"], h["body"])
        out.setdefault("activation", da)
        out.setdefault("torque", dt)
    cn = _vals(meta, data, "contact_n")[0]
    out["contact_off"] = np.concatenate([[0], np.cumsum(cn)])
    out["contact_pair"] = _vals(meta, data, "contact_pair")[0]
    out["contact_p"] = _vals(meta, data, "contact_p").T
    out["contact_j"] = _vals(meta, data, "contact_j")[0]
    out.update(_event_cols(meta, data))
    return out


def _event_cols(meta, data):
    n = _vals(meta, data, "event_n")[0]
    return {"event_off": np.concatenate([[0], np.cumsum(n)]),
            "event_code": _vals(meta, data, "event_code")[0],
            "event_note": _vals(meta, data, "event_note")[0],
            "event_val": _vals(meta, data, "event_val").T}


def event_json(i, code, pts, pre, acc):
    """One sim event (note_idx, code, points, pre, acc) in the replay's
    per-tick "events" form (the viewer's, and walk_events')."""
    return ({"miss": {"note": int(i)}} if code == 4 else
            {"post": {"note": int(i), "points": round(float(pts), 2)}}
            if code == 5 else
            {"bomb": {"note": int(i)}} if code == 6 else
            # env v17 (spec §12b): wall drain tick / wall-entry
            # combo break — no note id
            {"wall": {}} if code == 7 else
            {"wall_enter": {}} if code == 8 else
            # good cuts carry the component split (2026-07-25):
            # points = pre + acc (+30 on arc-headed free_post grants)
            {"cut": {"note": int(i), "good": code == 0,
                     "reason": EVENT_REASONS[code],
                     "points": round(float(pts), 2),
                     "pre": round(float(pre), 2),
                     "acc": round(float(acc), 2)}})


def tick_events(d):
    """Per-tick event lists (event_json form) of a decode() result."""
    off, code, note, val = (d["event_off"], d["event_code"].tolist(),
                            d["event_note"].tolist(), d["event_val"].tolist())
    return [[event_json(note[m], code[m], *val[m])
             for m in range(off[k], off[k + 1])] for k in range(d["T"])]


def events_only(meta, data):
    """Per-tick event lists straight from unpack()'s blocks — no pose
    decode (publish_replays.py walks the score from these)."""
    return tick_events({"T": int(meta["ticks"]), **_event_cols(meta, data)})


def to_ticks(d):
    """Per-tick dicts (the in-memory tick objects the viewer builds:
    t, time, q, twist, omega, sabers [[6], [6]], head, action, reward,
    contacts [{pair, p, j}], feet [[3], [3]], activation, torque,
    events) of a decode() result."""
    T = d["T"]
    q, tw, om = d["q"].tolist(), d["twist"].tolist(), d["omega"].tolist()
    sab, hd, ft = d["sabers"].tolist(), d["head"].tolist(), d["feet"].tolist()
    act, ac, tq = (d["action"].tolist(), d["activation"].tolist(),
                   d["torque"].tolist())
    rw, tm = d["reward"].tolist(), d["time"].tolist()
    coff, cpair = d["contact_off"], d["contact_pair"].tolist()
    cp, cj = d["contact_p"].tolist(), d["contact_j"].tolist()
    evs = tick_events(d)
    return [{"t": k, "time": tm[k], "q": q[k], "twist": tw[k],
             "omega": om[k], "sabers": [sab[k][0:6], sab[k][6:12]],
             "head": hd[k], "action": act[k], "reward": rw[k],
             "contacts": [{"pair": cpair[m], "p": cp[m], "j": cj[m]}
                          for m in range(coff[k], coff[k + 1])],
             "feet": [ft[k][0:3], ft[k][3:6]], "activation": ac[k],
             "torque": tq[k], "events": evs[k]} for k in range(T)]


def read_replay(path):
    """(header, per-tick dicts) of a v2 replay file — what the old JSONL
    readers got from its lines."""
    d = decode(read_raw(path))
    h = d["header"]
    assert h.get("version") == "v10", (
        f"need replay v10 (env v10 dynamic body), got {h.get('version')}")
    return h, to_ticks(d)
