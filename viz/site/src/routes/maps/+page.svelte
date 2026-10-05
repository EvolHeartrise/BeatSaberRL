<script>
	import Abbr from '$lib/Abbr.svelte';
	import { playOf } from '$lib/data.js';
	import { BLSTARS, SSSTARS } from '$lib/pp-labels.js';
	import * as Card from '$lib/components/ui/card/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import MapCell from '$lib/MapCell.svelte';
	import LevelFilters from '$lib/LevelFilters.svelte';
	import ModelName from '$lib/ModelName.svelte';
	import Pager from '$lib/Pager.svelte';
	import SortTh from '$lib/SortTh.svelte';
	import StatusSelect from '$lib/StatusSelect.svelte';
	import { emptyFilter, filterLevels, sortIdx } from '$lib/filter.js';
	import { num, pct, pp, stars } from '$lib/format.js';

	let { data } = $props();
	let L = $derived(data.levels);
	let models = $derived(data.models);
	let nSs = $derived(L.ssId.reduce((n, x) => n + (x != null ? 1 : 0), 0));
	let nBl = $derived(L.blId.reduce((n, x) => n + (x != null ? 1 : 0), 0));
	let nBoth = $derived(L.ssId.reduce((n, x, i) => n + (x != null && L.blId[i] != null ? 1 : 0), 0));

	// per-level aggregates across models: the best play (highest score of each
	// model's play: SS, else BL on a BeatLeader-only map), the top PP on each
	// board, and FC counts
	let agg = $derived.by(() => {
		const n = L.hash.length;
		const best = new Int16Array(n).fill(-1);
		const bestScore = new Float64Array(n);
		const topSs = new Float64Array(n).fill(NaN);
		const topBl = new Float64Array(n).fill(NaN);
		const fcs = new Int16Array(n);
		const played = new Int16Array(n);
		models.forEach((_, k) => {
			const R = data.results[k],
				RB = data.resultsBl[k];
			for (let i = 0; i < n; i++) {
				const P = playOf(R, RB, i);
				if (!P) continue;
				played[i]++;
				fcs[i] += P.fc[i];
				if (best[i] < 0 || P.score[i] > bestScore[i]) (best[i] = k), (bestScore[i] = P.score[i]);
				if (R.pp[i] != null && !(R.pp[i] <= topSs[i])) topSs[i] = R.pp[i];
				if (RB && RB.pp[i] != null && !(RB.pp[i] <= topBl[i])) topBl[i] = RB.pp[i];
			}
		});
		return { best, topSs, topBl, fcs, played };
	});
	/** the best play's accuracy (SS, else BL) @param {number} i */
	const bestAcc = (i) => {
		const k = agg.best[i];
		if (k < 0) return null;
		return data.results[k].acc[i] ?? data.resultsBl[k]?.acc[i] ?? null;
	};
	/** @param {Float64Array} a @param {number} i */
	const nn = (a, i) => (Number.isNaN(a[i]) ? null : a[i]);

	const PER = 50;
	let filter = $state(emptyFilter());
	let status = $state('all');
	let sort = $state({ key: 'ssStars', desc: true });
	let page = $state(0);

	/** @type {Record<string, (i: number) => number | string | null>} */
	const cols = {
		song: (i) => L.song[i],
		ssStars: (i) => L.ssStars[i],
		blStars: (i) => L.blStars[i],
		notes: (i) => L.notes[i],
		acc: (i) => bestAcc(i),
		ssPp: (i) => nn(agg.topSs, i),
		blPp: (i) => nn(agg.topBl, i),
		best: (i) => (agg.best[i] < 0 ? null : models[agg.best[i]].id)
	};

	let rows = $derived.by(() => {
		const idx = filterLevels(L, filter, (i) => {
			if (status === 'all') return true;
			const f = agg.fcs[i];
			return status === 'every' ? f === agg.played[i] && f > 0 : status === 'none' ? f === 0 : f > 0 && f < agg.played[i];
		});
		return sortIdx(idx, cols[sort.key], sort.desc);
	});
	$effect(() => {
		rows;
		page = 0;
	});
	let shown = $derived(rows.slice(page * PER, (page + 1) * PER));
</script>

<svelte:head>
	<title>Maps | Flee's Beat Saber Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="space-y-2">
		<h1 class="text-3xl font-bold tracking-tight">Maps</h1>
		<p class="text-muted-foreground">
			All {num(L.hash.length)} ranked difficulties: {num(nSs)} ranked on ScoreSaber, {num(nBl)} on BeatLeader, {num(nBoth)} on both.
			ScoreSaber and BeatLeader stars are different scales. "Best" is the highest-scoring model on the map; open a map to compare
			every model. Models benched before BeatLeader support have no plays on BeatLeader-only maps.
		</p>
	</div>

	<Card.Root>
		<Card.Content class="space-y-4">
			<LevelFilters bind:filter>
				<StatusSelect
					bind:value={status}
					label="Full-combo status"
					items={[
						{ value: 'all', label: 'All maps' },
						{ value: 'every', label: "FC'd by every model" },
						{ value: 'some', label: "FC'd by some models" },
						{ value: 'none', label: "FC'd by no model" }
					]}
				/>
			</LevelFilters>
			<Table.Root>
				<Table.Header>
					<Table.Row>
						<SortTh key="song" label="Map" bind:sort defaultDesc={false} />
						<SortTh key="ssStars" bind:sort num><Abbr {...SSSTARS} inline /></SortTh>
						<SortTh key="blStars" bind:sort num><Abbr {...BLSTARS} inline /></SortTh>
						<SortTh key="notes" label="Notes" bind:sort num />
						<SortTh key="best" label="Best model" bind:sort defaultDesc={false} />
						<SortTh key="acc" label="Best acc" bind:sort num />
						<SortTh key="ssPp" bind:sort num><Abbr label="Top SS PP" full="The highest ScoreSaber PP any model scored on the map" inline /></SortTh>
						<SortTh key="blPp" bind:sort num><Abbr label="Top BL PP" full="The highest BeatLeader PP any model scored on the map" inline /></SortTh>
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each shown as i (i)}
						{@const b = agg.best[i] < 0 ? null : models[agg.best[i]]}
						<Table.Row>
							<MapCell {L} {i} />
							<Table.Cell class="num">{L.ssStars[i] != null ? stars(L.ssStars[i]) : '—'}</Table.Cell>
							<Table.Cell class="num">{L.blStars[i] != null ? stars(L.blStars[i]) : '—'}</Table.Cell>
							<Table.Cell class="num">{num(L.notes[i])}</Table.Cell>
							<Table.Cell class="whitespace-nowrap">{#if b}<ModelName model={b} class="text-xs" />{/if}</Table.Cell>
							<Table.Cell class="num">{pct(bestAcc(i))}</Table.Cell>
							<Table.Cell class="num">{pp(nn(agg.topSs, i))}</Table.Cell>
							<Table.Cell class="num">{pp(nn(agg.topBl, i))}</Table.Cell>
						</Table.Row>
					{:else}
						<Table.Row><Table.Cell colspan={8} class="text-muted-foreground py-8 text-center">No maps match.</Table.Cell></Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
			<Pager bind:page total={rows.length} per={PER} />
		</Card.Content>
	</Card.Root>
</div>
