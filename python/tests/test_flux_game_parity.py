"""Env ↔ WebXR game parity, live (SPEC.md §Flux WebXR game).

The browser game judges with rust/flux-wasm — the env's own geom.rs /
flux.rs / combo.rs compiled to wasm32. This test records Flux episodes on
the env built HERE (python/tools/make_flux_game_fixture.py: every tick's
full judge input at full precision + the env's events) and makes the
wasm build replay them (viz/app/test/flux-parity.test.js, under node):
identical events tick for tick, identical ending and tallies. (A stale
COMMITTED fixture needs no test here: npm test replays it through the
current wasm judge, which then disagrees with its old events.) It also
pins the bundled chart pack to the generator that produced it.

Needs node and the wasm target (`rustup target add
wasm32-unknown-unknown`; `npm ci` in viz/app is not required — the
parity test only imports the app's own modules). Skips without node or
cargo.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(ROOT, "viz", "app")
TOOLS = os.path.join(ROOT, "python", "tools")
sys.path.insert(0, TOOLS)

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or shutil.which("cargo") is None,
    reason="node + cargo (wasm32 target) needed for the wasm side")


@pytest.fixture(scope="module")
def wasm_built():
    r = subprocess.run(["node", "tools/build-wasm.mjs"], cwd=APP,
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("cannot build rust/flux-wasm: " + r.stderr[-400:])
    return True


@needs_node
def test_wasm_judge_replays_the_live_env_exactly(tmp_path, wasm_built):
    import make_flux_game_fixture as fx
    data = fx.build()
    path = tmp_path / "flux_game_parity.json"
    path.write_text(json.dumps(data))
    r = subprocess.run(["node", "test/flux-parity.test.js", str(path)],
                       cwd=APP, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "wasm judge == env" in r.stdout


def _pack_from_js(text):
    return json.loads(re.search(r"export default (\[.*\]);\s*$", text, re.S).group(1))


def test_chart_pack_is_the_generator_output():
    """The bundled pack is flux_chart.py's output for its listed configs
    and seeds (to float round-off: the generator's trig is libm's)."""
    import make_flux_chart_pack as pack
    fresh = pack.build()
    with open(pack.DEFAULT_OUT) as f:
        committed = _pack_from_js(f.read())
    assert [c["id"] for c in fresh] == [c["id"] for c in committed]
    for a, b in zip(fresh, committed):
        assert a["config"] == b["config"] and a["seed"] == b["seed"]
        assert len(a["chart"]) == len(b["chart"]), a["id"]
        for oa, ob in zip(a["chart"], b["chart"]):
            assert oa["kind"] == ob["kind"] and oa["color"] == ob["color"]
            for k in ("t", "x", "y", "rot", "speed", "len", "sweet"):
                assert oa[k] == pytest.approx(ob[k], rel=1e-12, abs=1e-12), (a["id"], k)
