<script>
	// A play's BeatLeader PP ("—" when BeatLeader does not rank the level or
	// the model was not benched there), with its pass / acc / tech split, BL
	// accuracy and (when it differs from the row's play) the BL play itself
	// in a tooltip.
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { num, pct, pp } from './format.js';

	/** @type {{R: import('./data.js').Results, RB: import('./data.js').Results | null | undefined, i: number, model: import('./data.js').Model}} */
	let { R, RB, i, model } = $props();
	let has = $derived(!!RB && RB.score[i] != null);
	// the BL play differs from the row's (SS) play
	let otherPlay = $derived(has && R.score[i] != null && RB?.score[i] !== R.score[i]);
</script>

{#if has && RB}
	<Tooltip.Root>
		<Tooltip.Trigger class="cursor-help underline decoration-dotted decoration-from-font underline-offset-4">{pp(RB.pp[i])}</Tooltip.Trigger>
		<Tooltip.Content>
			<dl class="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 text-left tabular-nums">
				<dt class="opacity-70">Pass / Acc / Tech PP</dt>
				<dd class="text-right">{pp(RB.passPp?.[i])} / {pp(RB.accPp?.[i])} / {pp(RB.techPp?.[i])}</dd>
				<dt class="opacity-70">BL accuracy</dt>
				<dd class="text-right">{pct(RB.acc[i])}</dd>
				{#if otherPlay || R.score[i] == null}
					<dt class="opacity-70">BL play score</dt>
					<dd class="text-right">{num(RB.score[i])}</dd>
					<dt class="opacity-70">BL play misses</dt>
					<dd class="text-right">{num(RB.misses[i])}</dd>
				{/if}
			</dl>
		</Tooltip.Content>
	</Tooltip.Root>
	{#if RB.failed?.[i]}<Badge variant="destructive" class="ml-1" title="The episode was cut short (fall or tilt): a failed play scores 0 BeatLeader PP">failed</Badge>{/if}
{:else}
	<span class="text-muted-foreground">—</span>
{/if}
