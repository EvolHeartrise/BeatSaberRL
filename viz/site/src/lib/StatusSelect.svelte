<script>
	import * as Select from '$lib/components/ui/select/index.js';

	/** @type {{value: string, items: {value: string, label: string}[], label: string, class?: string}} */
	let { value = $bindable(), items, label, class: className = 'w-48' } = $props();
</script>

<Select.Root type="single" bind:value>
	<Select.Trigger class={className} aria-label={label}>{items.find((i) => i.value === value)?.label}</Select.Trigger>
	<Select.Content>
		{#each items as i (i.value)}<Select.Item value={i.value} label={i.label} />{/each}
	</Select.Content>
</Select.Root>
