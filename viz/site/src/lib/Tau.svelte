<script>
	// A play's applied torque (Σ|τ| N·m·s). Torque belongs to the physical
	// play, not to a leaderboard, so it is one number: the play the row
	// shows (the ScoreSaber play, else the BeatLeader one on a BL-only
	// level), the same play as the score and combo columns. A best-of
	// composite can take a level's BL play from another run; that play's
	// torque is in the tooltip. Benches before 2026-10-01 recorded no
	// torque: "—".
	import { nms } from './format.js';

	/** @type {{R: import('./data.js').Results, RB: import('./data.js').Results | null | undefined, i: number}} */
	let { R, RB, i } = $props();
	let ss = $derived(R.score[i] != null ? (R.tau?.[i] ?? null) : null);
	let bl = $derived(RB?.tau?.[i] ?? null);
	let v = $derived(ss ?? (R.score[i] == null ? bl : null));
	let title = $derived(
		v == null
			? ''
			: `${v.toLocaleString('en-US')} N·m·s` +
					(ss != null && bl != null && bl !== ss ? ` (the BeatLeader pick is another play: ${bl.toLocaleString('en-US')} N·m·s)` : '')
	);
</script>

{#if v != null}
	<span {title}>{nms(v)}</span>
{:else}
	<span class="text-muted-foreground">—</span>
{/if}
