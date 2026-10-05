<script>
	// A play's accuracy: ScoreSaber's (score over SS max score), with
	// BeatLeader's on a second line when it differs (BL max score differs on
	// chain maps, and the BL play may be another play). On a
	// BeatLeader-only level it is the BL accuracy, marked as such.
	import { pct } from './format.js';

	/** @type {{R: import('./data.js').Results, RB: import('./data.js').Results | null | undefined, i: number}} */
	let { R, RB, i } = $props();
	let ss = $derived(R.acc[i]);
	let bl = $derived(RB?.acc[i] ?? null);
</script>

{#if ss != null}
	{pct(ss)}
	{#if bl != null && bl !== ss}<div class="text-muted-foreground text-[0.7rem]" title="BeatLeader accuracy">BL {pct(bl)}</div>{/if}
{:else if bl != null}
	<span class="text-muted-foreground text-[0.7rem]" title="BeatLeader accuracy (not ranked on ScoreSaber)">BL</span> {pct(bl)}
{:else}
	<span class="text-muted-foreground">—</span>
{/if}
