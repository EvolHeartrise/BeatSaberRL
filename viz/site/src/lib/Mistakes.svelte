<script>
	// A play's mistakes: misses + bad cuts + bombs hit, with the breakdown
	// in a tooltip. Blank when the model has no play on the level.
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { num } from './format.js';

	/** @type {{p: import('./data.js').Results | null | undefined, i: number}} */
	let { p, i } = $props();
	let has = $derived(!!p && p.score[i] != null);
	let misses = $derived(p?.misses[i] ?? 0);
	let bad = $derived(p?.bad[i] ?? 0);
	let bombs = $derived(p?.bombs[i] ?? 0);
	let total = $derived(misses + bad + bombs);
</script>

{#if has}
	{#if total === 0}
		<span class="text-muted-foreground">0</span>
	{:else}
		<Tooltip.Root>
			<Tooltip.Trigger class="cursor-help underline decoration-dotted decoration-from-font underline-offset-4">{num(total)}</Tooltip.Trigger>
			<Tooltip.Content>
				<dl class="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 text-left tabular-nums">
					<dt class="opacity-70">Misses</dt>
					<dd class="text-right">{num(misses)}</dd>
					<dt class="opacity-70">Bad cuts</dt>
					<dd class="text-right">{num(bad)}</dd>
					<dt class="opacity-70">Bombs hit</dt>
					<dd class="text-right">{num(bombs)}</dd>
				</dl>
			</Tooltip.Content>
		</Tooltip.Root>
	{/if}
{/if}
