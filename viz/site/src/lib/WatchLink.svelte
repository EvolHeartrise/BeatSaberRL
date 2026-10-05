<script>
	// "Watch" link of a play row: opens the model's replay of level `i` in the
	// hosted 3D viewer (replays.flee.systems) in a new tab. Renders nothing
	// when the play has no replay.
	import CirclePlayIcon from '@lucide/svelte/icons/circle-play';
	import { watchUrl } from './data.js';

	/** @type {{model: import('./data.js').Model, i: number}} */
	let { model, i } = $props();
	let href = $derived(watchUrl(model, i));
</script>

{#if href}
	<a
		{href}
		target="_blank"
		rel="noopener"
		class="text-primary inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap underline-offset-4 hover:underline"
		title="Watch this play in the 3D replay viewer"
		aria-label="Watch the replay of this play (opens a new tab)"
	>
		<CirclePlayIcon class="size-4" />Watch
	</a>
{/if}
