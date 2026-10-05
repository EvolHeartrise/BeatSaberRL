<script>
	// "SS" / "BL" badges: the leaderboards that rank level `i`.
	/** @type {{L: import('./data.js').Levels, i: number, long?: boolean}} */
	let { L, i, long = false } = $props();
</script>

{#if L.ssId[i] != null}
	<span
		class="rounded-sm border border-(--ss)/40 bg-(--ss)/10 px-1 text-[0.65rem] leading-4 font-semibold text-(--ss)"
		title="Ranked on ScoreSaber">{long ? 'Ranked on ScoreSaber' : 'SS'}</span
	>
{/if}
{#if L.blId[i] != null}
	<span
		class="rounded-sm border border-(--bl)/40 bg-(--bl)/10 px-1 text-[0.65rem] leading-4 font-semibold text-(--bl)"
		title="Ranked on BeatLeader">{long ? 'Ranked on BeatLeader' : 'BL'}</span
	>
{/if}
