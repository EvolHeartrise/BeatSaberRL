<script>
	// Per-model line chart over star-rating buckets: FC rate or mean accuracy.
	import { pct } from './format.js';

	/** @type {{models: import('./data.js').Model[], metric: 'fcRate' | 'acc', highlight?: string | null}} */
	let { models, metric, highlight = null } = $props();

	const W = 720,
		H = 280,
		M = { t: 14, r: 16, b: 34, l: 50 };
	const iw = W - M.l - M.r,
		ih = H - M.t - M.b;

	let buckets = $derived(models[0]?.stars ?? []);
	let domain = $derived.by(() => {
		if (metric === 'fcRate') return [0, 1];
		const vals = models.flatMap((m) => m.stars.map((b) => b.acc ?? 1));
		const lo = Math.floor(Math.min(...vals) * 100 - 0.5) / 100;
		return [Math.max(0, lo), 1];
	});
	let ticks = $derived.by(() => {
		const [lo, hi] = domain;
		const step = hi - lo > 0.5 ? 0.2 : hi - lo > 0.1 ? 0.02 : 0.01;
		const out = [];
		for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(4));
		return out;
	});

	/** @param {number} i */
	const x = (i) => M.l + (buckets.length < 2 ? iw / 2 : (i / (buckets.length - 1)) * iw);
	/** @param {number} v */
	const y = (v) => M.t + ih - ((v - domain[0]) / (domain[1] - domain[0])) * ih;
	/** @param {import('./data.js').StarBucket} b */
	const bucketLabel = (b) => (b.hi == null ? `${b.lo}★+` : `${b.lo}–${b.hi}★`);
	/** @param {import('./data.js').Model} m */
	const pts = (m) =>
		m.stars
			.map((b, i) => (b[metric] == null ? null : `${x(i)},${y(/** @type {number} */ (b[metric]))}`))
			.filter(Boolean)
			.join(' ');
</script>

<svg viewBox="0 0 {W} {H}" role="img" aria-label="{metric === 'fcRate' ? 'Full-combo rate' : 'Mean accuracy'} by star rating">
	{#each ticks as t (t)}
		<line class="grid" x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} />
		<text class="axis" x={M.l - 8} y={y(t)} dy="0.32em" text-anchor="end">{pct(t, 0)}</text>
	{/each}
	{#each buckets as b, i (b.lo)}
		<text class="axis" x={x(i)} y={H - 10} text-anchor="middle">{bucketLabel(b)}</text>
	{/each}
	{#each models as m (m.id)}
		<g class="series" class:dim={highlight && highlight !== m.id} style="--c: {m.color}">
			<polyline points={pts(m)} />
			{#each m.stars as b, i (b.lo)}
				{#if b[metric] != null}
					<circle cx={x(i)} cy={y(/** @type {number} */ (b[metric]))} r="3.5">
						<title>{m.id} — {bucketLabel(b)}: {pct(b[metric])} ({b.n} maps)</title>
					</circle>
				{/if}
			{/each}
		</g>
	{/each}
</svg>

<style>
	svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.grid {
		stroke: var(--border);
	}
	.axis {
		fill: var(--muted-foreground);
		font-size: 11px;
		font-family: var(--font-mono);
	}
	.series polyline {
		fill: none;
		stroke: var(--c);
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.series circle {
		fill: var(--c);
		stroke: var(--card);
		stroke-width: 1.5;
	}
	.series.dim {
		opacity: 0.18;
	}
</style>
