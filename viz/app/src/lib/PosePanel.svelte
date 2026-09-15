<script>
  // Manual pose mode panel (env v10): the FULL body — root pose
  // (position + yaw/pitch/roll) and all 29 joints (§1.4 actuated
  // order), grouped spine / arms / legs, clamped to the anatomical
  // ranges. Live FK readout (wrists + blades + head — the numbers a
  // pose is checked against), mirror L<->R, reset, an optional static
  // reference note, and a copy-as-set_joints export: the q36 array the
  // sim's full-state hook (and the scenario tests) consume. Pure DOM
  // overlay; the 3D body is driven from the shared pose state through
  // the normal bodyFK path.
  import {
    pb, pose, mirrorPoseLR, resetPose, setPoseJoint, setPoseRoot,
  } from "./state.svelte.js";
  import {
    F_HEAD, GROUPS, JOINT_LIMITS, JOINT_NAMES, LANE_X, ROW_Y,
    SABER_COLORS, bodyFK, fkSaber, poseQ36, viewerTrackOffset,
  } from "./replay.js";

  const DEG = 180 / Math.PI;
  const q36 = $derived(poseQ36(pose));
  const fk = $derived(bodyFK(q36));
  // env v15: the hit plane rides the head (same gate the scene uses), so
  // the reference note is at z = zOff, not a fixed 0.
  const zOff = $derived(viewerTrackOffset(pb.rep, true, fk));
  const sabers = $derived([0, 1].map((h) => fkSaber(fk, h)));
  const v3 = (v) => v.map((x) => x.toFixed(3)).join(", ");

  const groupColor = (g) =>
    g === 1 || g === 3 ? SABER_COLORS[0]
    : g === 2 || g === 4 ? SABER_COLORS[1]
    : "#c5b3f0";

  // root controls: [key, label, lo, hi, step, isAngle]
  const ROOT_CTLS = [
    ["x", "x", -1.5, 1.5, 0.01, false],
    ["y", "y", 0.0, 2.0, 0.01, false],
    ["z", "z", -2.0, 1.0, 0.01, false],
    ["yaw", "yaw", -180, 180, 1, true],
    ["pitch", "pitch", -180, 180, 1, true],
    ["roll", "roll", -180, 180, 1, true],
  ];

  let copied = $state(false);
  function copyQ36() {
    const txt = JSON.stringify(q36.map((v) => +v.toFixed(6)));
    navigator.clipboard?.writeText(txt);
    copied = true;
    setTimeout(() => (copied = false), 1200);
  }

  const refValue = $derived(
    pose.ref ? `${pose.ref.col},${pose.ref.row}` : "off");
  function setRef(v) {
    if (v === "off") { pose.ref = null; return; }
    const [col, row] = v.split(",").map(Number);
    pose.ref = { col, row };
  }
</script>

<div id="pose">
  <div class="pp">
    <div class="head">
      <b>manual pose</b> <span class="dim">ω = 0 · p = exit</span>
      <button onclick={mirrorPoseLR}>mirror L↔R</button>
      <button onclick={resetPose}>reset</button>
    </div>
    <div class="cols">
      <div class="col">
        <div class="gt" style="color: #7aa2f7">root</div>
        {#each ROOT_CTLS as [key, name, lo, hi, st, isAngle] (key)}
          {@const val = isAngle ? pose.root[key] * DEG : pose.root[key]}
          <label>
            <span>{name} {val.toFixed(isAngle ? 0 : 2)}{isAngle ? "°" : ""}</span>
            <input
              type="range"
              min={lo}
              max={hi}
              step={st}
              value={val}
              oninput={(e) => setPoseRoot(key,
                isAngle ? +e.target.value / DEG : +e.target.value)}
            />
          </label>
        {/each}
        {#each [0] as g (g)}
          <div class="gt" style="color: {groupColor(0)}">spine</div>
          {#each GROUPS[0].joints as j (j)}
            {@const [lo, hi] = JOINT_LIMITS[j]}
            <label>
              <span>{JOINT_NAMES[j]} {(pose.q[j] * DEG).toFixed(1)}°</span>
              <input
                type="range"
                min={lo * DEG}
                max={hi * DEG}
                step="0.5"
                value={pose.q[j] * DEG}
                oninput={(e) => setPoseJoint(j, +e.target.value / DEG)}
              />
            </label>
          {/each}
        {/each}
      </div>
      {#each [[1, 3], [2, 4]] as pair (pair[0])}
        <div class="col">
          {#each pair as g (g)}
            <div class="gt" style="color: {groupColor(g)}">{GROUPS[g].name}</div>
            {#each GROUPS[g].joints as j (j)}
              {@const [lo, hi] = JOINT_LIMITS[j]}
              <label>
                <span>{JOINT_NAMES[j]} {(pose.q[j] * DEG).toFixed(1)}°</span>
                <input
                  type="range"
                  min={lo * DEG}
                  max={hi * DEG}
                  step="0.5"
                  value={pose.q[j] * DEG}
                  oninput={(e) => setPoseJoint(j, +e.target.value / DEG)}
                />
              </label>
            {/each}
          {/each}
        </div>
      {/each}
    </div>
    <div class="fk">
      wrist L ({v3(sabers[0].slice(0, 3))}) · blade L ({v3(sabers[0].slice(3))})<br />
      wrist R ({v3(sabers[1].slice(0, 3))}) · blade R ({v3(sabers[1].slice(3))})<br />
      head ({v3(fk.p[F_HEAD])})
    </div>
    <div class="ref">
      <span class="dim">reference note (hit plane z = {zOff.toFixed(3)}):</span>
      <select
        value={refValue}
        onchange={(e) => setRef(e.target.value)}
      >
        <option value="off">off</option>
        {#each [0, 1, 2, 3] as col (col)}
          {#each [0, 1, 2] as row (row)}
            <option value="{col},{row}">
              col {col} (x {LANE_X[col]}) · row {row} (y {ROW_Y[row]})
            </option>
          {/each}
        {/each}
      </select>
      <button onclick={copyQ36}>
        {copied ? "copied!" : "copy q36 (set_joints)"}
      </button>
    </div>
  </div>
</div>

<style>
  /* container passes pointer events through; the panel itself is live */
  #pose {
    /* --panel-top is set by App.svelte: the map card owns the very
       top-left corner when it is showing, so start below it */
    position: absolute; top: var(--panel-top, 64px); left: 12px; z-index: 3;
    pointer-events: none;
    max-height: calc(100vh - var(--panel-top, 64px) - 156px);
  }
  .pp {
    pointer-events: auto;
    background: rgba(17, 19, 25, 0.72);
    border: 1px solid rgba(58, 64, 80, 0.6);
    border-radius: 10px;
    backdrop-filter: blur(6px);
    padding: 10px 12px;
    font-size: 12px;
    color: #9aa0ae;
    width: 520px;
    max-height: inherit;
    overflow-y: auto;
  }
  .head {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    color: #e8eaf0;
  }
  .head button, .ref button {
    margin-left: auto;
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 1px 8px; cursor: pointer; font: inherit;
  }
  .head button:hover, .ref button:hover { background: #2d3342; }
  .head button + button { margin-left: 0; }
  .dim { color: #6b7180; }
  .cols { display: flex; gap: 12px; }
  .col { flex: 1; min-width: 0; }
  .gt { font-weight: 600; margin: 6px 0 2px; }
  label { display: block; margin: 1px 0; }
  label span {
    display: block; font-variant-numeric: tabular-nums; color: #9aa0ae;
    font-size: 11px;
  }
  input[type="range"] { width: 100%; margin: 0; height: 12px; }
  .fk {
    margin-top: 8px; font-size: 11px; color: #7dd3a0;
    font-variant-numeric: tabular-nums; line-height: 1.5;
    word-break: keep-all; white-space: nowrap; overflow-x: auto;
  }
  .ref { margin-top: 8px; display: flex; align-items: center; gap: 6px; }
  .ref select {
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 1px 4px; font: inherit;
  }
</style>
