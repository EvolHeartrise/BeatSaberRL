<script>
  // Manual pose mode panel: 12 sliders (per hand: pitch, yaw, twist,
  // elbow, roll, bend) clamped to the anatomical ranges, per-hand reset,
  // mirror L->R, live FK readout (wrist + blade unit dir — the numbers a
  // pose is usually checked against), and an optional static reference
  // note at a chosen (col, row) cell. Pure DOM overlay; the 3D body is
  // driven from the shared pose state through the normal armFK path.
  import {
    pose, mirrorPoseLR, resetPoseHand, setPoseJoint,
  } from "./state.svelte.js";
  import {
    JOINT_NAMES, LANE_X, ROW_Y, SABER_COLORS, armFK, jointLimits,
  } from "./replay.js";

  const DEG = 180 / Math.PI;
  const HANDS = ["Left", "Right"];
  const fk = $derived([0, 1].map((h) => armFK(h, pose.q[h])));
  const v3 = (v) => v.map((x) => x.toFixed(3)).join(", ");

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
      <button onclick={mirrorPoseLR}>mirror L→R</button>
    </div>
    <div class="cols">
      {#each [0, 1] as hand (hand)}
        <div class="hand">
          <div class="ht" style="color: {SABER_COLORS[hand]}">
            {HANDS[hand]}
            <button onclick={() => resetPoseHand(hand)}>reset</button>
          </div>
          {#each JOINT_NAMES as name, j (name)}
            {@const [lo, hi] = jointLimits(hand)[j]}
            <label>
              <span>{name} {(pose.q[hand][j] * DEG).toFixed(1)}°</span>
              <input
                type="range"
                min={lo * DEG}
                max={hi * DEG}
                step="0.5"
                value={pose.q[hand][j] * DEG}
                oninput={(e) => setPoseJoint(hand, j, +e.target.value / DEG)}
              />
            </label>
          {/each}
          <div class="fk">
            wrist ({v3(fk[hand].wrist)})<br />
            blade ({v3(fk[hand].blade)})
          </div>
        </div>
      {/each}
    </div>
    <div class="ref">
      <span class="dim">reference note (z = 0):</span>
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
    </div>
  </div>
</div>

<style>
  /* container passes pointer events through; the panel itself is live */
  #pose {
    position: absolute; top: 64px; left: 12px; z-index: 3;
    pointer-events: none;
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
    width: 340px;
  }
  .head {
    display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
    color: #e8eaf0;
  }
  .head button, .ht button {
    margin-left: auto;
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 1px 8px; cursor: pointer; font: inherit;
  }
  .head button:hover, .ht button:hover { background: #2d3342; }
  .dim { color: #6b7180; }
  .cols { display: flex; gap: 12px; }
  .hand { flex: 1; min-width: 0; }
  .ht {
    display: flex; align-items: center; font-weight: 600;
    margin-bottom: 4px;
  }
  label { display: block; margin: 2px 0; }
  label span {
    display: block; font-variant-numeric: tabular-nums; color: #9aa0ae;
  }
  input[type="range"] { width: 100%; margin: 0; height: 14px; }
  .fk {
    margin-top: 6px; font-size: 11px; color: #7dd3a0;
    font-variant-numeric: tabular-nums; line-height: 1.5;
    word-break: keep-all; white-space: nowrap;
  }
  .ref { margin-top: 8px; display: flex; align-items: center; gap: 6px; }
  .ref select {
    background: #232733; color: #cfd3dc; border: 1px solid #3a4050;
    border-radius: 6px; padding: 1px 4px; font: inherit;
  }
</style>
