<script>
	// "2,520 / 3,878 maps" under a BL total whose model did not play the whole
	// BeatLeader catalog (BL PP priced from its ScoreSaber bench: shared maps
	// only). Renders nothing for full coverage.
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { num } from './format.js';

	/** @type {{bl: {coverage?: {played: number, catalog: number}}, class?: string}} */
	let { bl, class: className = '' } = $props();
	let c = $derived(bl.coverage);
</script>

{#if c && c.played < c.catalog}
	<Tooltip.Root>
		<Tooltip.Trigger class="text-muted-foreground cursor-help text-[0.7rem] font-normal underline decoration-dotted underline-offset-4 {className}">
			{num(c.played)} / {num(c.catalog)} maps
		</Tooltip.Trigger>
		<Tooltip.Content>
			<span class="block max-w-72 text-left">
				BL PP from the ScoreSaber bench’s plays on maps both leaderboards rank; this model was not benched on the {num(c.catalog - c.played)}
				BeatLeader-only maps.
			</span>
		</Tooltip.Content>
	</Tooltip.Root>
{/if}
