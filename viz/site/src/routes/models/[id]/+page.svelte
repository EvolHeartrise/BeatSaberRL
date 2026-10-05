<script>
	import Abbr from '$lib/Abbr.svelte';
	import Acc from '$lib/Acc.svelte';
	import BlPp from '$lib/BlPp.svelte';
	import { playOf } from '$lib/data.js';
	import { BLPP, BLSTARS, BLTAU, BLUPP, BLWPP, SSPP, SSSTARS, SSTAU, TAU, UPP, WPP } from '$lib/pp-labels.js';
	import BlCoverage from '$lib/BlCoverage.svelte';
	import * as Card from '$lib/components/ui/card/index.js';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import MapCell from '$lib/MapCell.svelte';
	import Mistakes from '$lib/Mistakes.svelte';
	import Tau from '$lib/Tau.svelte';
	import LevelFilters from '$lib/LevelFilters.svelte';
	import Pager from '$lib/Pager.svelte';
	import ScoreCloud from '$lib/ScoreCloud.svelte';
	import SortTh from '$lib/SortTh.svelte';
	import StarChart from '$lib/StarChart.svelte';
	import StatusSelect from '$lib/StatusSelect.svelte';
	import WatchLink from '$lib/WatchLink.svelte';
	import { emptyFilter, filterLevels, sortIdx, sortRows } from '$lib/filter.js';
	import { date, nms, num, pct, pp, ppWhole, stars } from '$lib/format.js';

	let { data } = $props();
	let m = $derived(data.model);
	let L = $derived(data.levels);
	/** ScoreSaber results, aligned to the unified levels */
	let R = $derived(data.results);
	/** BeatLeader results (null = not benched on BeatLeader) */
	let RB = $derived(data.resultsBl);
	/** the play the shared columns show (SS, else BL on a BeatLeader-only map) @param {number} i */
	const P = (i) => playOf(R, RB, i);

	const PER = 50;
	let filter = $state(emptyFilter());
	let status = $state('all');
	let sort = $state({ key: 'ssPp', desc: true });
	let page = $state(0);

	/** @type {Record<string, (i: number) => number | string | null>} */
	const cols = {
		ssPp: (i) => R.pp[i],
		blPp: (i) => RB?.pp[i] ?? null,
		acc: (i) => R.acc[i] ?? RB?.acc[i] ?? null,
		score: (i) => P(i)?.score[i] ?? null,
		ssStars: (i) => (R.score[i] != null ? L.ssStars[i] : null),
		blStars: (i) => (RB?.score[i] != null ? L.blStars[i] : null),
		mistakes: (i) => {
			const q = P(i);
			return q && q.score[i] != null ? (q.misses[i] ?? 0) + (q.bad[i] ?? 0) + (q.bombs[i] ?? 0) : null;
		},
		song: (i) => L.song[i],
		combo: (i) => P(i)?.combo[i] ?? null,
		tau: (i) => (R.score[i] != null ? (R.tau?.[i] ?? null) : (RB?.tau?.[i] ?? null))
	};

	// plays with a replay on replays.flee.systems get a Watch column (and a filter)
	let hasReplays = $derived(!!m.replays && Object.keys(m.replays).length > 0);
	$effect(() => {
		// a model without replays has no "With replay" option to be on
		if (!hasReplays && status === 'replay') status = 'all';
	});
	/** @param {number} i */
	const statusOk = (i) =>
		status === 'all' ? true : status === 'replay' ? m.replays?.[i] != null : (status === 'fc') === (P(i)?.fc[i] === 1);

	let rows = $derived.by(() => {
		const idx = filterLevels(L, filter, (i) => P(i) != null && statusOk(i));
		return sortIdx(idx, cols[sort.key], sort.desc);
	});
	$effect(() => {
		// back to the first page whenever the result set changes
		rows;
		page = 0;
	});
	let shown = $derived(rows.slice(page * PER, (page + 1) * PER));
	let played = $derived(L.hash.reduce((n, _, i) => n + (P(i) != null ? 1 : 0), 0));

	// star-rating breakdowns: ScoreSaber, and BeatLeader when benched there
	let bucketSort = $state({ key: 'lo', desc: false });
	/** @param {import('$lib/data.js').StarBucket[]} b */
	const sortBuckets = (b) => sortRows(b, (x) => x[/** @type {'lo' | 'n' | 'acc' | 'fcRate'} */ (bucketSort.key)], bucketSort.desc);
	let boards = $derived([
		{ key: 'ss', name: 'ScoreSaber', STARS: SSSTARS, stats: /** @type {import('$lib/data.js').BoardStats} */ (m) },
		...(m.bl ? [{ key: 'bl', name: 'BeatLeader', STARS: BLSTARS, stats: /** @type {import('$lib/data.js').BoardStats} */ (m.bl) }] : [])
	]);

	/** BL totals over the shared maps only (BL PP priced from the SS bench) */
	let partialBl = $derived(!!m.bl?.coverage && m.bl.coverage.played < m.bl.coverage.catalog);
	let statCards = $derived([
		{ k: WPP.label, abbr: WPP, v: pp(m.totalPp) },
		{ k: UPP.label, abbr: UPP, v: ppWhole(m.unweightedPp) },
		{
			k: BLWPP.label,
			abbr: BLWPP,
			v: m.bl ? pp(m.bl.totalPp) : '—',
			cov: true,
			sub: !m.bl ? 'not benched on BeatLeader' : undefined
		},
		{ k: BLUPP.label, abbr: BLUPP, v: m.bl ? ppWhole(m.bl.unweightedPp) : '—', cov: true, sub: m.bl ? undefined : 'not benched on BeatLeader' },
		{ k: 'Full combos', v: num(m.fcs), sub: `of ${num(m.levels)} SS maps` + (m.bl ? ` · BL: ${num(m.bl.fcs)} of ${num(m.bl.levels)}` : '') },
		{ k: 'Mean accuracy', v: pct(m.meanAcc), sub: 'SS maps' + (m.bl ? ` · BL: ${pct(m.bl.meanAcc)}` : '') },
		{ k: 'Good-cut rate', v: pct(m.goodCutRate), sub: 'SS maps' + (m.bl ? ` · BL: ${pct(m.bl.goodCutRate)}` : '') },
		{
			k: 'Misses',
			v: num(m.misses),
			sub: `SS maps · ${num(m.badCuts)} bad cuts · ${num(m.bombsHit)} bombs` + (m.bl ? ` · BL: ${num(m.bl.misses)} misses` : '')
		},
		{
			k: SSTAU.label,
			abbr: SSTAU,
			v: m.tau != null ? nms(m.tau) : '—',
			sub: m.tau != null ? `N·m·s over ${num(m.levels)} SS maps` : 'not recorded by this bench'
		},
		{
			k: BLTAU.label,
			abbr: BLTAU,
			v: m.bl?.tau != null ? nms(m.bl.tau) : '—',
			sub: m.bl?.tau != null ? `N·m·s over ${num(m.bl.levels)} BL maps` : !m.bl ? 'not benched on BeatLeader' : 'not recorded by this bench'
		}
	]);
	const ncol = 13;
</script>

<svelte:head>
	<title>{m.id} | Flee's Beat Saber Lab</title>
</svelte:head>

<div class="space-y-6">
	<div class="space-y-2">
		<div class="flex flex-wrap items-center gap-3">
			<span class="size-3.5 rounded-full" style:background={m.color}></span>
			<h1 class="font-mono text-2xl font-bold tracking-tight break-all">{m.id}</h1>
			{#if m.tickHz !== 30}<Badge variant="outline">{m.tickHz} Hz control</Badge>{/if}
		</div>
		{#if m.tickHz !== 30}
			<p class="text-muted-foreground text-sm">
				Same weights as a 30 Hz model, benched at {m.tickHz} Hz: the policy acts {m.tickHz} times per second
				instead of the 30 it was trained at (physics substep and per-second limits unchanged).
			</p>
		{/if}
		<p class="text-muted-foreground text-sm break-all">
			Benched {date(m.updated)} ·
			{#if m.weightsHash}
				weights <span class="font-mono" title={m.weightsHash}>{m.weightsHash.replace('sha256:', '').slice(0, 12)}</span>
			{:else}
				<span class="font-mono" title={m.modelHash}>{m.modelHash.replace('sha256:', '').slice(0, 12)}</span>
			{/if}
			{#if !m.bl}· not benched on BeatLeader (the bench predates the BeatLeader catalog){/if}
		</p>
	</div>

	<div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
		{#each statCards as s (s.k)}
			<Card.Root>
				<Card.Header>
					<Card.Description>{#if s.abbr}<Abbr {...s.abbr} />{:else}{s.k}{/if}</Card.Description>
					<Card.Title class="font-mono text-xl tabular-nums">{s.v}</Card.Title>
				</Card.Header>
				{#if s.sub || (s.cov && partialBl)}
					<Card.Content class="text-muted-foreground space-y-0.5 text-xs">
						{#if s.sub}<div>{s.sub}</div>{/if}
						{#if s.cov && partialBl}<div class="flex"><BlCoverage bl={m.bl} class="text-xs" /></div>{/if}
					</Card.Content>
				{/if}
			</Card.Root>
		{/each}
	</div>

	{#each boards as b (b.key)}
		<div class="grid items-start gap-4 lg:grid-cols-5">
			<Card.Root class="lg:col-span-3">
				<Card.Header>
					<Card.Title>FC rate by {b.name} star rating</Card.Title>
					{#if b.key === 'bl'}<Card.Description>BeatLeader stars: a different scale from ScoreSaber’s.</Card.Description>{/if}
				</Card.Header>
				<Card.Content><StarChart models={[{ ...m, stars: b.stats.stars }]} metric="fcRate" /></Card.Content>
			</Card.Root>
			<Card.Root class="lg:col-span-2">
				<Card.Header>
					<Card.Title>{b.name} star breakdown</Card.Title>
				</Card.Header>
				<Card.Content>
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<SortTh key="lo" bind:sort={bucketSort} defaultDesc={false}><Abbr {...b.STARS} inline /></SortTh>
								<SortTh key="n" label="Maps" bind:sort={bucketSort} num />
								<SortTh key="acc" label="Mean acc" bind:sort={bucketSort} num />
								<SortTh key="fcRate" label="FC rate" bind:sort={bucketSort} num />
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each sortBuckets(b.stats.stars) as x (x.lo)}
								<Table.Row>
									<Table.Cell class="font-mono text-[0.82rem]">{x.hi == null ? `${x.lo}+` : `${x.lo}–${x.hi}`} ★</Table.Cell>
									<Table.Cell class="num">{num(x.n)}</Table.Cell>
									<Table.Cell class="num">{pct(x.acc)}</Table.Cell>
									<Table.Cell class="num">{pct(x.fcRate, 1)}</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</Card.Content>
			</Card.Root>
		</div>
	{/each}

	<section class="space-y-4" aria-labelledby="plays-h">
		<div class="space-y-1">
			<h2 id="plays-h" class="text-lg font-semibold tracking-tight">Plays</h2>
			<p class="text-muted-foreground text-sm">
				The model’s greedy play on every ranked difficulty{m.bl ? ' of ScoreSaber and BeatLeader' : ' of ScoreSaber'}.
				Score, mistakes and combo are the ScoreSaber play{#if m.bl}{' '}(the BeatLeader play on BeatLeader-only maps); hover BL PP for its split{/if}.
				The filters apply to the score cloud and the table.
			</p>
		</div>
		<LevelFilters bind:filter>
			<StatusSelect
				bind:value={status}
				label="Combo status"
				class="w-40"
				items={[
					{ value: 'all', label: 'All plays' },
					{ value: 'fc', label: 'Full combos' },
					{ value: 'nofc', label: 'Combo broken' },
					...(hasReplays ? [{ value: 'replay', label: 'With replay' }] : [])
				]}
			/>
		</LevelFilters>
		<Card.Root>
			<Card.Header>
				<Card.Title>Score cloud</Card.Title>
				<Card.Description>Every play as one dot. Hover for details, click to open the map.</Card.Description>
			</Card.Header>
			<Card.Content><ScoreCloud {L} {R} {RB} idx={rows} total={played} bind:diff={filter.diff} /></Card.Content>
		</Card.Root>
		<Card.Root>
			<Card.Content class="space-y-4">
				<Table.Root>
					<Table.Header>
						<Table.Row>
							<Table.Head class="w-10 text-right">#</Table.Head>
							<SortTh key="song" label="Map" bind:sort defaultDesc={false} />
							<SortTh key="ssStars" bind:sort num><Abbr {...SSSTARS} inline /></SortTh>
							<SortTh key="blStars" bind:sort num><Abbr {...BLSTARS} inline /></SortTh>
							<SortTh key="ssPp" bind:sort num><Abbr {...SSPP} inline /></SortTh>
							<SortTh key="blPp" bind:sort num><Abbr {...BLPP} inline /></SortTh>
							<SortTh key="acc" bind:sort num><Abbr label="Accuracy" full="ScoreSaber accuracy; BeatLeader’s below it when different" inline /></SortTh>
							<SortTh key="score" label="Score" bind:sort num />
							<SortTh key="mistakes" bind:sort num><Abbr label="Mistakes" full="Misses + bad cuts + bombs hit (hover a value for the breakdown)" inline /></SortTh>
							<SortTh key="combo" label="Combo" bind:sort num />
							<SortTh key="tau" bind:sort num><Abbr {...TAU} inline /></SortTh>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each shown as i, k (i)}
							{@const p = P(i)}
							<Table.Row>
								<Table.Cell class="num text-muted-foreground">{page * PER + k + 1}</Table.Cell>
								<MapCell {L} {i}>{#if hasReplays}<WatchLink model={m} {i} />{/if}</MapCell>
								<Table.Cell class="num">{L.ssStars[i] != null ? stars(L.ssStars[i]) : '—'}</Table.Cell>
								<Table.Cell class="num">{L.blStars[i] != null ? stars(L.blStars[i]) : '—'}</Table.Cell>
								<Table.Cell class="num font-semibold">{#if R.pp[i] != null}{pp(R.pp[i])}{:else}<span class="text-muted-foreground font-normal">—</span>{/if}</Table.Cell>
								<Table.Cell class="num font-semibold"><BlPp {R} {RB} {i} model={m} /></Table.Cell>
								<Table.Cell class="num"><Acc {R} {RB} {i} /></Table.Cell>
								<Table.Cell class="num">{num(p?.score[i])}</Table.Cell>
								<Table.Cell class="num"><Mistakes {p} {i} /></Table.Cell>
								<Table.Cell class="num">
									{#if p?.fc[i]}<span class="font-semibold text-(--fc)">FC</span>{:else}{num(p?.combo[i])}<span class="text-muted-foreground">/{num(L.notes[i])}</span>{/if}
								</Table.Cell>
								<Table.Cell class="num"><Tau {R} {RB} {i} /></Table.Cell>
							</Table.Row>
						{:else}
							<Table.Row><Table.Cell colspan={ncol} class="text-muted-foreground py-8 text-center">No plays match.</Table.Cell></Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
				<Pager bind:page total={rows.length} per={PER} />
			</Card.Content>
		</Card.Root>
	</section>
</div>
