<script>
	import ArrowDownIcon from '@lucide/svelte/icons/arrow-down';
	import ArrowUpIcon from '@lucide/svelte/icons/arrow-up';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Table from '$lib/components/ui/table/index.js';
	import { cn } from '$lib/utils.js';

	/** @type {{key: string, label?: string, sort: {key: string, desc: boolean}, num?: boolean, defaultDesc?: boolean, class?: string, children?: import('svelte').Snippet}} */
	let { key, label = '', sort = $bindable(), num = false, defaultDesc = true, class: className = '', children } = $props();
	let active = $derived(sort.key === key);

	function toggle() {
		sort = active ? { key, desc: !sort.desc } : { key, desc: defaultDesc };
	}
</script>

<Table.Head class={cn(num && 'text-right', className)} aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
	<button
		class={cn(
			'hover:text-foreground inline-flex items-center gap-1 rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50',
			active ? 'text-foreground' : 'text-muted-foreground',
			num && 'flex-row-reverse'
		)}
		onclick={toggle}
	>
		{#if children}{@render children()}{:else}{label}{/if}
		{#if active}
			{#if sort.desc}<ArrowDownIcon class="size-3.5" />{:else}<ArrowUpIcon class="size-3.5" />{/if}
		{:else}
			<ChevronsUpDownIcon class="size-3.5 opacity-50" />
		{/if}
	</button>
</Table.Head>
