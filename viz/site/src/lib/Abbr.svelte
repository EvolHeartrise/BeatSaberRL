<script>
	// An abbreviated label with its full wording in a tooltip, plus an optional
	// "Learn more" link (the content is hoverable, so the link can be clicked).
	// `inline` renders the trigger as a span, for use inside another button
	// (a sort header).
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';

	/** @type {{label: string, full: string, href?: string, inline?: boolean, class?: string}} */
	let { label, full, href, inline = false, class: className = '' } = $props();
	let cls = $derived(`cursor-help underline decoration-dotted decoration-from-font underline-offset-4 ${className}`);
</script>

<Tooltip.Root>
	{#if inline}
		<Tooltip.Trigger>
			{#snippet child({ props })}<span {...props} class={cls}>{label}</span>{/snippet}
		</Tooltip.Trigger>
	{:else}
		<Tooltip.Trigger class={cls}>{label}</Tooltip.Trigger>
	{/if}
	<Tooltip.Content>
		<div class="flex flex-col gap-0.5 text-left">
			<span>{full}</span>
			{#if href}
				<a {href} target="_blank" rel="noopener" class="font-medium underline underline-offset-2">Learn more</a>
			{/if}
		</div>
	</Tooltip.Content>
</Tooltip.Root>
