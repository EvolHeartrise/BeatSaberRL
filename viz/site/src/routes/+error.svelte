<script>
	import { page } from '$app/state';
	import { Button } from '$lib/components/ui/button/index.js';
</script>

<svelte:head><title>{page.status} | Flee's Beat Saber Lab</title></svelte:head>

<div class="flex flex-col items-center gap-3 py-24 text-center">
	<h1 class="font-mono text-5xl font-bold">{page.status}</h1>
	<p class="text-muted-foreground">{page.error?.message}</p>
	<Button href="/" variant="outline">Back to the home page</Button>
</div>
