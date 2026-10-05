<script>
	import { Badge } from '$lib/components/ui/badge/index.js';
	import { DIFFS, DIFF_KEYS } from './format.js';

	/** @type {{diff: number, href?: string, class?: string}} */
	let { diff, href, class: className = '' } = $props();
</script>

<Badge
	variant="outline"
	{href}
	class="border-[color-mix(in_srgb,var(--c)_45%,transparent)] bg-[color-mix(in_srgb,var(--c)_12%,transparent)] text-(--c) {className}"
	style="--c: var(--diff-{DIFF_KEYS[diff]})">{DIFFS[diff]}</Badge
>
