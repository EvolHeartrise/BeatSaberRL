<script>
	/** @type {{model: import('./data.js').Model, link?: boolean, class?: string}} */
	let { model, link = true, class: className = '' } = $props();
</script>

<svelte:element
	this={link ? 'a' : 'span'}
	href={link ? `/models/${model.id}` : undefined}
	class="inline-flex items-center gap-2 font-mono text-[0.82rem] {link ? 'hover:underline underline-offset-4' : ''} {className}"
>
	<span class="size-2.5 shrink-0 rounded-full" style:background={model.color}></span>{model.id}
</svelte:element>
