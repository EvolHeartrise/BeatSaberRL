<script>
	import Abbr from '$lib/Abbr.svelte';
	import { BLTAU, BLUPP, BLWPP, SSTAU, UPP, WPP } from '$lib/pp-labels.js';
	import BlCoverage from '$lib/BlCoverage.svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import * as ToggleGroup from '$lib/components/ui/toggle-group/index.js';
	import ModelName from '$lib/ModelName.svelte';
	import SortTh from '$lib/SortTh.svelte';
	import { sortRows } from '$lib/filter.js';
	import StarChart from '$lib/StarChart.svelte';
	import { date, nms, num, pct, pp, ppWhole } from '$lib/format.js';

	let { data } = $props();
	let models = $derived(data.models);
	// the models benched on BeatLeader, with their BL star buckets for the chart
	let blModels = $derived(models.flatMap((m) => (m.bl ? [{ ...m, stars: m.bl.stars }] : [])));
	let ssLevels = $derived(models[0]?.levels ?? 0);

	let sort = $state({ key: 'ssw', desc: true });
	// BL columns are null for models not benched on BeatLeader: sortRows puts
	// nulls last in either direction
	/** @type {Record<string, (m: import('$lib/data.js').Model) => number | string | null>} */
	const cols = {
		id: (m) => m.id,
		ssw: (m) => m.totalPp,
		ssu: (m) => m.unweightedPp,
		blw: (m) => m.bl?.totalPp ?? null,
		blu: (m) => m.bl?.unweightedPp ?? null,
		sst: (m) => m.tau ?? null,
		blt: (m) => m.bl?.tau ?? null,
		updated: (m) => m.updated
	};
	let rows = $derived(sortRows(models, cols[sort.key], sort.desc));

	/** @type {'fcRate' | 'acc'} */
	let metric = $state('fcRate');
	/** @type {string | null} */
	let hover = $state(null);
</script>

<svelte:head>
	<title>Rankings | Flee's Beat Saber Lab</title>
</svelte:head>

<div class="space-y-6">
	<h1 class="text-3xl font-bold tracking-tight">Rankings</h1>

	<Card.Root>
		<Card.Header>
			<Card.Title>Model rankings</Card.Title>
			<Card.Description>
				ScoreSaber and BeatLeader totals side by side (different PP formulas and star scales: compare within a column). BL totals
				marked “N / M maps” cover only the maps both leaderboards rank (BL PP of the model’s ScoreSaber bench plays).
			</Card.Description>
		</Card.Header>
		<Card.Content>
			<Table.Root>
				<Table.Header>
					<Table.Row>
						<Table.Head class="w-10 text-right">#</Table.Head>
						<SortTh key="id" label="Model" bind:sort defaultDesc={false} />
						<SortTh key="ssw" bind:sort num><Abbr {...WPP} inline /></SortTh>
						<SortTh key="ssu" bind:sort num><Abbr {...UPP} inline /></SortTh>
						<SortTh key="blw" bind:sort num><Abbr {...BLWPP} inline /></SortTh>
						<SortTh key="blu" bind:sort num><Abbr {...BLUPP} inline /></SortTh>
						<SortTh key="sst" bind:sort num><Abbr {...SSTAU} inline /></SortTh>
						<SortTh key="blt" bind:sort num><Abbr {...BLTAU} inline /></SortTh>
						<SortTh key="updated" label="Benched" bind:sort num />
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each rows as m, k (m.id)}
						<Table.Row onmouseenter={() => (hover = m.id)} onmouseleave={() => (hover = null)}>
							<Table.Cell class="num text-muted-foreground">{k + 1}</Table.Cell>
							<Table.Cell class="whitespace-nowrap"><ModelName model={m} /></Table.Cell>
							<Table.Cell class="num font-semibold">{pp(m.totalPp)}</Table.Cell>
							<Table.Cell class="num">{ppWhole(m.unweightedPp)}</Table.Cell>
							<Table.Cell class="num font-semibold">
								{#if m.bl}{pp(m.bl.totalPp)}<div><BlCoverage bl={m.bl} /></div>
								{:else}<span class="text-muted-foreground font-normal">—</span>{/if}
							</Table.Cell>
							<Table.Cell class="num">{#if m.bl}{ppWhole(m.bl.unweightedPp)}{:else}<span class="text-muted-foreground">—</span>{/if}</Table.Cell>
							<Table.Cell class="num" title={m.tau != null ? `${m.tau.toLocaleString('en-US')} N·m·s` : 'not recorded by this bench'}>{#if m.tau != null}{nms(m.tau)}{:else}<span class="text-muted-foreground">—</span>{/if}</Table.Cell>
							<Table.Cell class="num" title={m.bl?.tau != null ? `${m.bl.tau.toLocaleString('en-US')} N·m·s` : 'not recorded'}>{#if m.bl?.tau != null}{nms(m.bl.tau)}{:else}<span class="text-muted-foreground">—</span>{/if}</Table.Cell>
							<Table.Cell class="num text-muted-foreground">{date(m.updated)}</Table.Cell>
						</Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header>
			<Card.Title>By star rating</Card.Title>
			<Card.Description>How each model holds up as maps get harder.</Card.Description>
			<Card.Action>
				<ToggleGroup.Root type="single" variant="outline" size="sm" bind:value={() => metric, (v) => v && (metric = /** @type {any} */ (v))}>
					<ToggleGroup.Item value="fcRate">FC rate</ToggleGroup.Item>
					<ToggleGroup.Item value="acc">Mean accuracy</ToggleGroup.Item>
				</ToggleGroup.Root>
			</Card.Action>
		</Card.Header>
		<Card.Content class="space-y-6">
			<div class="grid gap-6 {blModels.length ? 'lg:grid-cols-2' : ''}">
				<figure class="space-y-1">
					<figcaption class="text-sm font-medium">ScoreSaber <span class="text-muted-foreground font-normal">(SS ★, ScoreSaber ranked maps)</span></figcaption>
					<StarChart {models} {metric} highlight={hover} />
				</figure>
				{#if blModels.length}
					<figure class="space-y-1">
						<figcaption class="text-sm font-medium">
							BeatLeader <span class="text-muted-foreground font-normal">(BL ★, a different scale; BeatLeader ranked maps)</span>
						</figcaption>
						<StarChart models={blModels} {metric} highlight={hover} />
					</figure>
				{/if}
			</div>
			<div class="flex flex-wrap gap-x-5 gap-y-2">
				{#each models as m (m.id)}
					<span role="presentation" onmouseenter={() => (hover = m.id)} onmouseleave={() => (hover = null)}>
						<ModelName model={m} class="text-xs" />
					</span>
				{/each}
			</div>
		</Card.Content>
	</Card.Root>
</div>
