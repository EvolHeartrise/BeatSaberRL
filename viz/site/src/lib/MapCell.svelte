<script>
	// The "Map" cell of the level tables: song + difficulty badge + the
	// leaderboards ranking it, mapper below. `children`: extras on the song
	// line (the model page's Watch link).
	import * as Table from '$lib/components/ui/table/index.js';
	import Diff from './Diff.svelte';
	import Ranked from './Ranked.svelte';
	import { mapHref } from './data.js';

	/** @type {{L: import('./data.js').Levels, i: number, children?: import('svelte').Snippet}} */
	let { L, i, children } = $props();
</script>

<Table.Cell class="max-w-96 min-w-60 whitespace-normal">
	<div class="flex flex-wrap items-center gap-x-2 gap-y-1">
		<a class="font-medium underline-offset-4 hover:underline" href={mapHref(L, i)}>{L.song[i]}</a>
		<Diff diff={L.diff[i]} />
		<span class="inline-flex gap-1"><Ranked {L} {i} /></span>
		{@render children?.()}
	</div>
	<div class="text-muted-foreground text-xs">{L.mapper[i]}</div>
</Table.Cell>
