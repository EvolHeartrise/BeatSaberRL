<script>
	import SearchIcon from '@lucide/svelte/icons/search';
	import { Input } from '$lib/components/ui/input/index.js';
	import * as Select from '$lib/components/ui/select/index.js';
	import { DIFFS } from './format.js';
	import { starScale } from './filter.js';

	/** @type {{filter: import('./filter.js').LevelFilter, children?: import('svelte').Snippet}} */
	let { filter = $bindable(), children } = $props();

	const rankedItems = [
		{ value: 'any', label: 'Any leaderboard' },
		{ value: 'ss', label: 'Ranked on ScoreSaber' },
		{ value: 'bl', label: 'Ranked on BeatLeader' },
		{ value: 'both', label: 'Ranked on both' },
		{ value: 'ssonly', label: 'ScoreSaber only' },
		{ value: 'blonly', label: 'BeatLeader only' }
	];
	let scale = $derived(starScale(filter) === 'bl' ? 'BL' : 'SS');
	const diffItems = [{ value: '-1', label: 'All difficulties' }, ...DIFFS.map((d, i) => ({ value: String(i), label: d }))];
	/** @param {string | number} v */
	const numOrNull = (v) => (v === '' || v == null || Number.isNaN(Number(v)) ? null : Number(v));
</script>

<div class="flex flex-wrap items-center gap-2">
	<div class="relative min-w-52 flex-1">
		<SearchIcon class="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2" />
		<Input type="search" placeholder="Search song or mapper…" class="pl-8" bind:value={filter.q} aria-label="Search song or mapper" />
	</div>
	<Select.Root type="single" bind:value={() => String(filter.diff), (v) => (filter.diff = Number(v))}>
		<Select.Trigger class="w-40" aria-label="Difficulty">{diffItems.find((d) => d.value === String(filter.diff))?.label}</Select.Trigger>
		<Select.Content>
			{#each diffItems as d (d.value)}<Select.Item value={d.value} label={d.label} />{/each}
		</Select.Content>
	</Select.Root>
	<Select.Root type="single" bind:value={() => filter.ranked, (v) => (filter.ranked = /** @type {any} */ (v))}>
		<Select.Trigger class="w-48" aria-label="Leaderboard">{rankedItems.find((d) => d.value === filter.ranked)?.label}</Select.Trigger>
		<Select.Content>
			{#each rankedItems as d (d.value)}<Select.Item value={d.value} label={d.label} />{/each}
		</Select.Content>
	</Select.Root>
	<div class="flex items-center gap-1.5" title="The star range uses BeatLeader stars when the leaderboard filter is BeatLeader, else ScoreSaber stars">
		<Input
			type="number"
			min="0"
			max="16"
			step="0.5"
			placeholder="Min {scale} ★"
			class="w-28"
			aria-label="Minimum {scale} stars"
			bind:value={() => filter.minStars ?? '', (v) => (filter.minStars = numOrNull(v))}
		/>
		<span class="text-muted-foreground">–</span>
		<Input
			type="number"
			min="0"
			max="16"
			step="0.5"
			placeholder="Max {scale} ★"
			class="w-28"
			aria-label="Maximum {scale} stars"
			bind:value={() => filter.maxStars ?? '', (v) => (filter.maxStars = numOrNull(v))}
		/>
	</div>
	{@render children?.()}
</div>
