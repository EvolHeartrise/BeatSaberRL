<script>
	import Abbr from '$lib/Abbr.svelte';
	import { BLWPP, WPP } from '$lib/pp-labels.js';
	import ArrowRightIcon from '@lucide/svelte/icons/arrow-right';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import ModelName from '$lib/ModelName.svelte';
	import { num, pp } from '$lib/format.js';

	let { data } = $props();
	let models = $derived(data.models);
	// models.json is ranked by ScoreSaber weighted PP
	let top = $derived(models[0]);
	let totalLevels = $derived(top?.levels ?? 0);
</script>

<svelte:head>
	<title>Flee's Beat Saber Lab</title>
</svelte:head>

<div class="space-y-10">
	<section class="space-y-5 pt-2 md:pt-6">
		<div class="flex items-center gap-3">
			<svg viewBox="0 0 32 32" class="size-10" aria-hidden="true">
				<path d="M7 25 21 6" stroke="var(--beat-red)" stroke-width="3.2" stroke-linecap="round" />
				<path d="M11 6l14 19" stroke="var(--beat-blue)" stroke-width="3.2" stroke-linecap="round" />
			</svg>
			<span class="text-muted-foreground text-sm font-medium tracking-wide uppercase">Flee's Beat Saber Lab</span>
		</div>
		<h1 class="max-w-3xl text-4xl font-bold tracking-tight md:text-5xl">Reinforcement-learning agents playing ranked Beat Saber with a whole body</h1>
		<p class="text-muted-foreground max-w-2xl text-lg">
			Flee's Beat Saber Lab trains policies to play Beat Saber inside a physics simulation, then benchmarks every model on every
			ScoreSaber and BeatLeader ranked map. This site keeps the records.
		</p>
		<div class="flex flex-wrap gap-3">
			<a href="/rankings" class={buttonVariants({ size: 'lg' })}>See the rankings <ArrowRightIcon /></a>
			<a href="/maps" class={buttonVariants({ size: 'lg', variant: 'outline' })}>Browse maps</a>
		</div>
		{#if top}
			<p class="text-muted-foreground flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
				Current top model:
				<ModelName model={top} />
				with <span class="text-foreground font-mono tabular-nums">{pp(top.totalPp)}</span> <Abbr {...WPP} /> across
				{num(totalLevels)} ScoreSaber ranked maps{#if top.bl}
					and <span class="text-foreground font-mono tabular-nums">{pp(top.bl.totalPp)}</span> <Abbr {...BLWPP} /> across
					{num(top.bl.levels)}{#if top.bl.coverage && top.bl.coverage.played < top.bl.coverage.catalog}{' '}of {num(top.bl.coverage.catalog)}{/if} BeatLeader ranked maps{/if}.
			</p>
		{/if}
	</section>

	<p class="text-muted-foreground text-xs">
		Flee's Beat Saber Lab is an independent research project and is not affiliated with Beat Games, ScoreSaber, BeatLeader or
		BeatSaver. Map data and cover art come from ScoreSaber, BeatLeader and BeatSaver.
	</p>
</div>
