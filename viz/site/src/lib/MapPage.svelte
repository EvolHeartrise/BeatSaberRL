<script>
	// One ranked level with both leaderboards' facts and every model's play.
	// Served at /maps/<ScoreSaber id> (levels ScoreSaber ranks) and
	// /maps/bl/<BeatLeader id> (BeatLeader-only levels); see map-load.js.
	import Abbr from './Abbr.svelte';
	import Acc from './Acc.svelte';
	import BlPp from './BlPp.svelte';
	import Ranked from './Ranked.svelte';
	import { BLPP, SSPP, TAU } from './pp-labels.js';
	import Tau from './Tau.svelte';
	import ExternalLinkIcon from '@lucide/svelte/icons/external-link';
	import MusicIcon from '@lucide/svelte/icons/music';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import * as Card from '$lib/components/ui/card/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import Diff from './Diff.svelte';
	import ModelName from './ModelName.svelte';
	import SortTh from './SortTh.svelte';
	import WatchLink from './WatchLink.svelte';
	import { mapHref, playOf } from './data.js';
	import { sortRows } from './filter.js';
	import { DIFFS, beatleader, cover, duration, num, pct, pp, scoresaber, stars } from './format.js';

	/** @type {{data: Awaited<ReturnType<typeof import('./map-load.js').loadMap>>}} */
	let { data } = $props();

	let L = $derived(data.levels);
	let i = $derived(data.i);
	let song = $derived(L.song[i]);
	let hash = $derived(L.hash[i]);
	let diff = $derived(L.diff[i]);
	let onSs = $derived(L.ssId[i] != null);
	let onBl = $derived(L.blId[i] != null);

	// every model's play on this level (SS play, else BL on a BeatLeader-only level)
	let played = $derived(
		data.models.flatMap((m, k) => {
			const R = data.results[k],
				RB = data.resultsBl[k];
			const P = playOf(R, RB, i);
			return P ? [{ m, R, RB, P }] : [];
		})
	);
	// models without a BeatLeader bench have no play on a BeatLeader-only level
	let missing = $derived(data.models.filter((m) => !played.some((p) => p.m === m)));
	let sort = $state({ key: 'score', desc: true });
	/** @type {Record<string, (p: (typeof played)[number]) => number | string | null>} */
	const cols = {
		id: (p) => p.m.id,
		score: (p) => p.P.score[i],
		acc: (p) => p.R.acc[i] ?? p.RB?.acc[i] ?? null,
		ssPp: (p) => p.R.pp[i],
		blPp: (p) => p.RB?.pp[i] ?? null,
		combo: (p) => p.P.combo[i],
		misses: (p) => p.P.misses[i],
		bad: (p) => p.P.bad[i],
		bombs: (p) => p.P.bombs[i],
		tau: (p) => (p.R.score[i] != null ? (p.R.tau?.[i] ?? null) : (p.RB?.tau?.[i] ?? null))
	};
	let plays = $derived(sortRows(played, cols[sort.key], sort.desc));
	// a Watch column only when some model's play here has a replay
	let anyReplay = $derived(played.some((p) => p.m.replays?.[i] != null));

	// the song's other ranked difficulties (same characteristic), on either board
	let siblings = $derived.by(() => {
		/** @type {{diff: number, j: number}[]} */
		const out = [];
		for (let j = 0; j < L.hash.length; j++) if (L.hash[j] === hash && L.char[j] === L.char[i]) out.push({ diff: L.diff[j], j });
		return out.sort((a, b) => a.diff - b.diff);
	});
	let coverOk = $state(true);
	$effect(() => {
		hash;
		coverOk = true;
	});

	let ssFacts = $derived(
		onSs
			? [
					{ k: 'SS ★', v: stars(/** @type {number} */ (L.ssStars[i])) },
					{ k: 'Max PP', v: pp(L.maxPp[i]) }
				]
			: []
	);
	let blFacts = $derived(
		onBl
			? [
					{ k: 'BL ★', v: stars(/** @type {number} */ (L.blStars[i])) },
					{ k: 'Pass', v: stars(/** @type {number} */ (L.passRating[i])) },
					{ k: 'Acc', v: stars(/** @type {number} */ (L.accRating[i])) },
					{ k: 'Tech', v: stars(/** @type {number} */ (L.techRating[i])) },
					{ k: 'Predicted acc', v: pct(L.predAcc[i]) }
				]
			: []
	);
	let maxScoreFacts = $derived.by(() => {
		const s = L.maxScore[i],
			b = L.blMaxScore[i];
		if (s != null && b != null && s !== b)
			return [
				{ k: 'SS max score', v: num(s) },
				{ k: 'BL max score', v: num(b) }
			];
		return [{ k: 'Max score', v: num(s ?? b) }];
	});
	let songFacts = $derived([...maxScoreFacts, { k: 'Notes', v: num(L.notes[i]) }, { k: 'Length', v: duration(L.duration[i]) }]);
</script>

<svelte:head>
	<title>{song} ({DIFFS[diff]}) | Flee's Beat Saber Lab</title>
</svelte:head>

{#snippet factList(/** @type {{k: string, v: string}[]} */ fs)}
	<dl class="flex flex-wrap gap-x-5 gap-y-1 text-sm">
		{#each fs as f (f.k)}
			<div class="flex gap-2">
				<dt class="text-muted-foreground">{f.k}</dt>
				<dd class="font-mono tabular-nums">{f.v}</dd>
			</div>
		{/each}
	</dl>
{/snippet}

<div class="space-y-6">
	<Card.Root>
		<Card.Content class="flex flex-col gap-5 sm:flex-row">
			{#if coverOk}
				<img
					src={cover(hash)}
					alt=""
					class="ring-foreground/10 size-28 shrink-0 rounded-lg object-cover ring-1 sm:size-36"
					onerror={() => (coverOk = false)}
				/>
			{:else}
				<div class="bg-muted text-muted-foreground grid size-28 shrink-0 place-items-center rounded-lg sm:size-36" aria-hidden="true">
					<MusicIcon class="size-10" />
				</div>
			{/if}
			<div class="min-w-0 flex-1 space-y-3">
				<div class="min-w-0">
					<h1 class="text-2xl font-bold tracking-tight break-words">{song}</h1>
					<p class="text-muted-foreground">mapped by {L.mapper[i]}</p>
				</div>
				<div class="flex flex-wrap items-center gap-1.5">
					{#each siblings as s (s.j)}
						<Diff diff={s.diff} href={mapHref(L, s.j)} class={s.j === i ? 'ring-2 ring-(--c)/40' : 'opacity-60 hover:opacity-100'} />
					{/each}
					{#if L.chars[L.char[i]] !== 'Standard'}<span class="text-muted-foreground text-xs">{L.chars[L.char[i]]}</span>{/if}
					<span class="ml-1 inline-flex gap-1"><Ranked {L} {i} long /></span>
				</div>
				<div class="grid gap-3 sm:grid-cols-2">
					<div class="space-y-1">
						<div class="text-xs font-semibold text-(--ss)">ScoreSaber</div>
						{#if onSs}{@render factList(ssFacts)}{:else}<p class="text-muted-foreground text-sm">Not ranked on ScoreSaber.</p>{/if}
					</div>
					<div class="space-y-1">
						<div class="text-xs font-semibold text-(--bl)">BeatLeader</div>
						{#if onBl}{@render factList(blFacts)}{:else}<p class="text-muted-foreground text-sm">Not ranked on BeatLeader.</p>{/if}
					</div>
				</div>
				{@render factList(songFacts)}
				<div class="flex flex-wrap gap-2">
					{#if onSs}
						<a class={buttonVariants({ variant: 'outline', size: 'sm' })} href={scoresaber(/** @type {number} */ (L.ssId[i]))} target="_blank" rel="noopener">
							ScoreSaber leaderboard <ExternalLinkIcon />
						</a>
					{/if}
					{#if onBl}
						<a class={buttonVariants({ variant: 'outline', size: 'sm' })} href={beatleader(/** @type {string} */ (L.blId[i]))} target="_blank" rel="noopener">
							BeatLeader leaderboard <ExternalLinkIcon />
						</a>
					{/if}
				</div>
			</div>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header>
			<Card.Title>Model results</Card.Title>
			<Card.Description>
				Score, combo and misses are the ScoreSaber play{onBl && !onSs ? ' (here, BeatLeader-only, the BeatLeader play)' : ''}.
				{#if onBl}BL PP is BeatLeader’s pass + acc + tech formula (hover for the split); BeatLeader accuracy is the score over
					BeatLeader’s max score.{/if}
			</Card.Description>
		</Card.Header>
		<Card.Content>
			<Table.Root>
				<Table.Header>
					<Table.Row>
						<Table.Head class="w-10 text-right">#</Table.Head>
						<SortTh key="id" label="Model" bind:sort defaultDesc={false} />
						<SortTh key="score" label="Score" bind:sort num />
						<SortTh key="acc" bind:sort num><Abbr label="Accuracy" full="ScoreSaber accuracy; BeatLeader’s below it when different" inline /></SortTh>
						<SortTh key="ssPp" bind:sort num><Abbr {...SSPP} inline /></SortTh>
						<SortTh key="blPp" bind:sort num><Abbr {...BLPP} inline /></SortTh>
						<SortTh key="combo" label="Combo" bind:sort num />
						<SortTh key="misses" label="Misses" bind:sort num />
						<SortTh key="bad" label="Bad cuts" bind:sort num />
						<SortTh key="bombs" label="Bombs" bind:sort num />
						<SortTh key="tau" bind:sort num><Abbr {...TAU} inline /></SortTh>
						{#if anyReplay}<Table.Head><span class="sr-only">Replay</span></Table.Head>{/if}
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each plays as p, k (p.m.id)}
						{@const { m, R, RB, P } = p}
						<Table.Row>
							<Table.Cell class="num text-muted-foreground">{k + 1}</Table.Cell>
							<Table.Cell class="whitespace-nowrap">
								<ModelName model={m} />
							</Table.Cell>
							<Table.Cell class="num">{num(P.score[i])}</Table.Cell>
							<Table.Cell class="num font-semibold"><Acc {R} {RB} {i} /></Table.Cell>
							<Table.Cell class="num">{#if R.pp[i] != null}{pp(R.pp[i])}{:else}<span class="text-muted-foreground">—</span>{/if}</Table.Cell>
							<Table.Cell class="num"><BlPp {R} {RB} {i} model={m} /></Table.Cell>
							<Table.Cell class="num">{#if P.fc[i]}<span class="font-semibold text-(--fc)">FC</span>{:else}{num(P.combo[i])}{/if}</Table.Cell>
							<Table.Cell class="num">{P.misses[i] || ''}</Table.Cell>
							<Table.Cell class="num">{P.bad[i] || ''}</Table.Cell>
							<Table.Cell class="num">{P.bombs[i] || ''}</Table.Cell>
							<Table.Cell class="num"><Tau {R} {RB} {i} /></Table.Cell>
							{#if anyReplay}<Table.Cell><WatchLink model={m} {i} /></Table.Cell>{/if}
						</Table.Row>
					{:else}
						<Table.Row><Table.Cell colspan={11} class="text-muted-foreground py-8 text-center">No model has played this map.</Table.Cell></Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
			{#if missing.length}
				<p class="text-muted-foreground mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
					No play here (BeatLeader-only map; benched before BeatLeader support):
					{#each missing as m (m.id)}<ModelName model={m} class="text-xs" />{/each}
				</p>
			{/if}
		</Card.Content>
	</Card.Root>
</div>
