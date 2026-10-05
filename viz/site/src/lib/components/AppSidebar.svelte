<script>
	import { page } from '$app/state';
	import { mode, toggleMode } from 'mode-watcher';
	import InfoIcon from '@lucide/svelte/icons/info';
	import MapIcon from '@lucide/svelte/icons/map';
	import MoonIcon from '@lucide/svelte/icons/moon';
	import SunIcon from '@lucide/svelte/icons/sun';
	import TrophyIcon from '@lucide/svelte/icons/trophy';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';

	const sidebar = Sidebar.useSidebar();

	const nav = [
		{ href: '/', label: 'About', icon: InfoIcon },
		{ href: '/rankings', label: 'Rankings', icon: TrophyIcon },
		{ href: '/maps', label: 'Maps', icon: MapIcon }
	];
	let path = $derived(page.url.pathname);
	// close the mobile sheet after navigating (the link keeps the menu
	// button's own click handler from `props`)
	const closeMobile = () => sidebar.isMobile && sidebar.setOpenMobile(false);
</script>

<Sidebar.Root>
	<Sidebar.Header>
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton size="lg" tooltipContent="Flee's Beat Saber Lab">
					{#snippet child({ props })}
						<a href="/" {...props} onclick={(e) => { props.onclick?.(e); closeMobile(); }}>
							<div class="bg-sidebar-primary/10 flex aspect-square size-8 items-center justify-center rounded-lg">
								<svg viewBox="0 0 32 32" class="size-5!" aria-hidden="true">
									<path d="M7 25 21 6" stroke="var(--beat-red)" stroke-width="3.2" stroke-linecap="round" />
									<path d="M11 6l14 19" stroke="var(--beat-blue)" stroke-width="3.2" stroke-linecap="round" />
								</svg>
							</div>
							<div class="grid flex-1 text-left leading-tight">
								<span class="truncate font-semibold">Flee's Beat Saber Lab</span>
								<span class="text-muted-foreground truncate text-xs">RL agent records</span>
							</div>
						</a>
					{/snippet}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Header>

	<Sidebar.Content>
		<Sidebar.Group>
			<Sidebar.GroupLabel>Browse</Sidebar.GroupLabel>
			<Sidebar.Menu>
				{#each nav as n (n.href)}
					<Sidebar.MenuItem>
						<Sidebar.MenuButton isActive={n.href === '/' ? path === '/' : n.href === '/rankings' ? path === '/rankings' || path.startsWith('/models') : path.startsWith(n.href)} tooltipContent={n.label}>
							{#snippet child({ props })}
								<a href={n.href} {...props} onclick={(e) => { props.onclick?.(e); closeMobile(); }}><n.icon /><span>{n.label}</span></a>
							{/snippet}
						</Sidebar.MenuButton>
					</Sidebar.MenuItem>
				{/each}
			</Sidebar.Menu>
		</Sidebar.Group>
	</Sidebar.Content>

	<Sidebar.Footer>
		<Sidebar.Menu>
			<Sidebar.MenuItem>
				<Sidebar.MenuButton onclick={toggleMode} tooltipContent="Toggle theme">
					{#if mode.current === 'dark'}<SunIcon /><span>Light mode</span>{:else}<MoonIcon /><span>Dark mode</span>{/if}
				</Sidebar.MenuButton>
			</Sidebar.MenuItem>
		</Sidebar.Menu>
	</Sidebar.Footer>
</Sidebar.Root>
