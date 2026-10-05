<script>
	import '../app.css';
	import { ModeWatcher } from 'mode-watcher';
	import AppSidebar from '$lib/components/AppSidebar.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import { beforeNavigate } from '$app/navigation';
	import { updated } from '$app/state';

	// a newer deploy is live (version poll, vite.config.js): load the next
	// page from the server instead of importing chunks that no longer exist
	beforeNavigate(({ willUnload, to }) => {
		if (updated.current && !willUnload && to?.url) location.href = to.url.href;
	});

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
	<meta name="description" content="Flee's Beat Saber Lab: records of reinforcement-learning agents on every ScoreSaber and BeatLeader ranked Beat Saber map." />
</svelte:head>

<ModeWatcher />

<!-- the desktop sidebar is always open; on phones it is a sheet opened from the trigger below -->
<Sidebar.Provider bind:open={() => true, () => {}}>
	<AppSidebar />
	<!-- min-w-0: a wide table scrolls inside its container instead of widening the page -->
	<Sidebar.Inset class="min-w-0">
		<div class="px-2 pt-2 md:hidden">
			<Sidebar.Trigger />
		</div>
		<main class="mx-auto w-full max-w-6xl flex-1 px-4 py-6 md:px-6">
			{@render children()}
		</main>
	</Sidebar.Inset>
</Sidebar.Provider>
