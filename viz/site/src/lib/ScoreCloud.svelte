<script>
	// Scatter of every play of one model: one dot per level, coloured by
	// difficulty, filled for a full combo and a ring when the combo broke.
	// Dots are painted on a canvas (thousands of them); axes, the hovered dot
	// and the tooltip are SVG/HTML on top. Hover picks the nearest dot through
	// a uniform grid, so a pointer only has to be close, not dead-centre.
	import { goto } from '$app/navigation';
	import { mode } from 'mode-watcher';
	import { Toggle } from '$lib/components/ui/toggle/index.js';
	import StatusSelect from './StatusSelect.svelte';
	import { DIFFS, DIFF_KEYS, duration, num, pct, pp, stars } from './format.js';
	import { mapHref, playOf } from './data.js';

	// L: the unified catalog; R: the model's ScoreSaber results; RB: its
	// BeatLeader results (null = not benched on BeatLeader). A dot needs a
	// value on both axes, so an SS axis leaves out BeatLeader-only plays and a
	// BL axis ScoreSaber-only ones; misses and combo are the play's own (SS,
	// else BL on a BeatLeader-only map).
	/** @type {{L: import('./data.js').Levels, R: import('./data.js').Results, RB: import('./data.js').Results | null, idx: number[], total: number, diff: number}} */
	let { L, R, RB, idx, total, diff = $bindable() } = $props();
	/** @param {number} i */
	const P = (i) => playOf(R, RB, i);

	/** @typedef {{label: string, get: (i: number) => number | null, tick: (v: number, step: number) => string, int?: boolean, cap?: number}} Axis */

	/** @type {Record<string, Axis>} */
	const XS = {
		stars: { label: 'ScoreSaber stars', get: (i) => L.ssStars[i], tick: (v) => `${+v.toFixed(2)}★` },
		blStars: { label: 'BeatLeader stars', get: (i) => L.blStars[i], tick: (v) => `${+v.toFixed(2)}★` },
		nps: { label: 'Notes per second', get: (i) => (L.duration[i] > 0 ? L.notes[i] / L.duration[i] : null), tick: (v) => `${+v.toFixed(2)}` },
		duration: { label: 'Song length', get: (i) => L.duration[i] / 60, tick: (v) => duration(v * 60) }
	};
	/** @type {Record<string, Axis>} */
	const YS = {
		acc: { label: 'SS accuracy', get: (i) => R.acc[i], tick: (v, s) => pct(v, s >= 0.01 ? 0 : s >= 0.001 ? 1 : 2), cap: 1 },
		blAcc: { label: 'BL accuracy', get: (i) => RB?.acc[i] ?? null, tick: (v, s) => pct(v, s >= 0.01 ? 0 : s >= 0.001 ? 1 : 2), cap: 1 },
		pp: { label: 'ScoreSaber PP', get: (i) => R.pp[i], tick: (v) => num(Math.round(v)) },
		blPp: { label: 'BeatLeader PP', get: (i) => RB?.pp[i] ?? null, tick: (v) => num(Math.round(v)) },
		misses: { label: 'Misses', get: (i) => P(i)?.misses[i] ?? null, tick: (v) => num(v), int: true }
	};
	let xItems = $derived([
		{ value: 'stars', label: 'x: SS stars' },
		...(RB ? [{ value: 'blStars', label: 'x: BL stars' }] : []),
		{ value: 'nps', label: 'x: Notes/sec' },
		{ value: 'duration', label: 'x: Song length' }
	]);
	let yItems = $derived([
		{ value: 'acc', label: 'y: SS accuracy' },
		...(RB ? [{ value: 'blAcc', label: 'y: BL accuracy' }] : []),
		{ value: 'pp', label: 'y: SS PP' },
		...(RB ? [{ value: 'blPp', label: 'y: BL PP' }] : []),
		{ value: 'misses', label: 'y: Misses' }
	]);

	let xKey = $state('stars');
	let yKey = $state('acc');
	let fullRange = $state(false);
	let X = $derived(XS[xKey]);
	let Y = $derived(YS[yKey]);

	// ---- geometry
	let w = $state(0);
	let narrow = $derived(w > 0 && w < 560);
	let h = $derived(narrow ? 300 : 380);
	const M = { t: 24, r: 14, b: 38, l: 52 };
	let iw = $derived(Math.max(10, w - M.l - M.r));
	let ih = $derived(h - M.t - M.b);
	let rad = $derived(narrow ? 2.3 : 2.8);

	/** @param {number} lo @param {number} hi @param {number} n */
	function niceStep(lo, hi, n) {
		const raw = Math.max(hi - lo, 1e-9) / n;
		const p = 10 ** Math.floor(Math.log10(raw));
		const e = raw / p;
		return p * (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1);
	}
	/** @param {number} lo @param {number} hi @param {number} n @param {boolean} [int] */
	function scale(lo, hi, n, int) {
		if (!(hi > lo)) hi = lo + (int ? 1 : Math.abs(lo) * 0.1 || 1);
		let step = niceStep(lo, hi, n);
		if (int) step = Math.max(1, Math.round(step));
		const a = Math.floor(lo / step + 1e-9) * step,
			b = Math.ceil(hi / step - 1e-9) * step;
		const ticks = [];
		for (let v = a; v <= b + step * 1e-6; v += step) ticks.push(+v.toFixed(6));
		return { lo: a, hi: b, step, ticks };
	}

	// values of the shown plays (nulls dropped), plus the domains
	let pts = $derived.by(() => {
		const xs = [],
			ys = [],
			ii = [];
		for (const i of idx) {
			const xv = X.get(i),
				yv = Y.get(i);
			if (xv == null || yv == null) continue;
			xs.push(xv);
			ys.push(yv);
			ii.push(i);
		}
		return { xs, ys, ii };
	});
	let xs = $derived.by(() => {
		let lo = Infinity,
			hi = -Infinity;
		for (const v of pts.xs) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
		return lo === Infinity ? scale(0, 1, 5) : scale(lo, hi, narrow ? 5 : 8);
	});
	// accuracy has a long thin tail towards 0 (a handful of failed plays); by
	// default the axis starts at the 1st percentile and those plays sit on the
	// bottom edge, counted in the note below the chart
	let yFloor = $derived.by(() => {
		if (!pts.ys.length) return { min: 0, clip: null };
		const s = Float64Array.from(pts.ys).sort();
		const min = s[0];
		if ((yKey !== 'acc' && yKey !== 'blAcc') || fullRange || s.length < 50) return { min, clip: null };
		const p1 = s[Math.floor(0.01 * (s.length - 1))];
		return p1 - min > 0.05 ? { min: p1, clip: p1 } : { min, clip: null };
	});
	let ys = $derived.by(() => {
		if (!pts.ys.length) return scale(0, 1, 5);
		let hi = -Infinity;
		for (const v of pts.ys) hi = Math.max(hi, v);
		const lo = Y.cap === 1 ? yFloor.min : Math.min(0, yFloor.min);
		const sc = scale(lo, hi, narrow ? 5 : 6, Y.int);
		if (Y.int) sc.lo -= 0.5; // room for the jittered band at 0
		if (Y.cap != null && sc.hi > Y.cap) {
			sc.hi = Y.cap;
			sc.ticks = sc.ticks.filter((t) => t <= Y.cap + 1e-9);
		}
		return sc;
	});
	let clipped = $derived(yFloor.clip == null ? 0 : pts.ys.filter((v) => v < ys.lo).length);

	/** @param {number} v */
	const px = (v) => M.l + ((v - xs.lo) / (xs.hi - xs.lo)) * iw;
	/** @param {number} v */
	const py = (v) => M.t + ih - ((Math.max(v, ys.lo) - ys.lo) / (ys.hi - ys.lo)) * ih;
	// integer axes stack whole numbers on one line; spread each dot a little
	// (stable per level) so the stack reads as a band
	/** @param {number} i */
	const jitter = (i) => (((i * 2654435761) >>> 0) / 4294967296 - 0.5) * 0.55;

	// screen positions, drawn in difficulty order; rebuilt with the layout
	let screen = $derived.by(() => {
		const n = pts.ii.length;
		const sx = new Float32Array(n),
			sy = new Float32Array(n);
		for (let k = 0; k < n; k++) {
			sx[k] = px(pts.xs[k]);
			sy[k] = py(Y.int ? pts.ys[k] + jitter(pts.ii[k]) : pts.ys[k]);
		}
		return { sx, sy };
	});

	// ---- hit-testing grid: cell size = the pick radius, look in the 3x3 around
	const CELL = 22;
	let grid = $derived.by(() => {
		const cols = Math.ceil((w || 1) / CELL) + 1,
			rows = Math.ceil(h / CELL) + 1;
		/** @type {Map<number, number[]>} */
		const g = new Map();
		const { sx, sy } = screen;
		for (let k = 0; k < sx.length; k++) {
			const key = Math.floor(sy[k] / CELL) * cols + Math.floor(sx[k] / CELL);
			const b = g.get(key);
			if (b) b.push(k);
			else g.set(key, [k]);
		}
		return { g, cols, rows };
	});
	/** @param {number} x @param {number} y @param {number} r */
	function nearest(x, y, r) {
		const { g, cols } = grid;
		const cx = Math.floor(x / CELL),
			cy = Math.floor(y / CELL);
		let best = -1,
			bd = r * r;
		for (let dy = -1; dy <= 1; dy++)
			for (let dx = -1; dx <= 1; dx++) {
				const b = g.get((cy + dy) * cols + cx + dx);
				if (!b) continue;
				for (const k of b) {
					const ex = screen.sx[k] - x,
						ey = screen.sy[k] - y,
						d = ex * ex + ey * ey;
					if (d <= bd) (bd = d), (best = k);
				}
			}
		return best;
	}

	// ---- drawing
	/** @type {HTMLCanvasElement | undefined} */
	let canvas = $state();
	let colors = $state(['#888', '#888', '#888', '#888', '#888']);
	$effect(() => {
		mode.current; // re-read the theme's difficulty colours when it flips
		requestAnimationFrame(() => {
			const cs = getComputedStyle(document.documentElement);
			colors = DIFF_KEYS.map((k) => cs.getPropertyValue(`--diff-${k}`).trim() || '#888');
		});
	});
	$effect(() => {
		if (!canvas || !w) return;
		const dpr = window.devicePixelRatio || 1;
		canvas.width = Math.round(w * dpr);
		canvas.height = Math.round(h * dpr);
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, w, h);
		const { sx, sy } = screen;
		const ii = pts.ii;
		// full combos as soft filled dots first, broken combos as rings on top
		for (const fc of [1, 0]) {
			for (let d = 0; d < 5; d++) {
				ctx.beginPath();
				let any = false;
				for (let k = 0; k < ii.length; k++) {
					const i = ii[k];
					if (L.diff[i] !== d || (P(i)?.fc[i] ? 1 : 0) !== fc) continue;
					ctx.moveTo(sx[k] + rad, sy[k]);
					ctx.arc(sx[k], sy[k], rad, 0, Math.PI * 2);
					any = true;
				}
				if (!any) continue;
				if (fc) {
					ctx.globalAlpha = 0.5;
					ctx.fillStyle = colors[d];
					ctx.fill();
				} else {
					ctx.globalAlpha = 0.9;
					ctx.lineWidth = 1.3;
					ctx.strokeStyle = colors[d];
					ctx.stroke();
				}
			}
		}
		ctx.globalAlpha = 1;
	});

	// ---- interaction
	let hover = $state(-1); // index into pts
	let touch = $state(false);
	let hi = $derived(hover >= 0 && hover < pts.ii.length ? pts.ii[hover] : -1);
	$effect(() => {
		pts; // a new point set invalidates the hovered index
		hover = -1;
	});
	/** @param {PointerEvent} e */
	function local(e) {
		const r = /** @type {HTMLElement} */ (e.currentTarget).getBoundingClientRect();
		return [e.clientX - r.left, e.clientY - r.top];
	}
	/** @param {PointerEvent} e */
	function onmove(e) {
		if (e.pointerType !== 'mouse') return;
		touch = false;
		const [x, y] = local(e);
		hover = nearest(x, y, 14);
	}
	/** @param {MouseEvent} e */
	function onclick(e) {
		const pe = /** @type {PointerEvent} */ (e);
		const isTouch = pe.pointerType === 'touch' || pe.pointerType === 'pen';
		const [x, y] = local(pe);
		const k = nearest(x, y, isTouch ? 22 : 14);
		if (k < 0) {
			hover = -1;
			return;
		}
		// a mouse click opens the map; a tap first shows the tooltip, and a
		// second tap on the same dot opens it
		if (!isTouch || (touch && k === hover)) goto(mapHref(L, pts.ii[k]));
		else {
			touch = true;
			hover = k;
		}
	}

	let hp = $derived(hi < 0 ? null : P(hi));
	let tip = $derived.by(() => {
		if (hi < 0) return null;
		const x = screen.sx[hover],
			y = screen.sy[hover];
		return { x, y, left: x > w * 0.55 };
	});

	// ---- legend: a click narrows the shared difficulty filter to that difficulty
	let counts = $derived.by(() => {
		const c = [0, 0, 0, 0, 0];
		for (const i of pts.ii) c[L.diff[i]]++;
		return c;
	});
	/** @param {number} d */
	const pickDiff = (d) => (diff = diff === d ? -1 : d);

	let fcShown = $derived(pts.ii.reduce((n, i) => n + (P(i)?.fc[i] ? 1 : 0), 0));
	let summary = $derived(
		`Scatter plot of ${num(pts.ii.length)} of ${num(total)} plays: ${Y.label} against ${X.label}, ` +
			`coloured by difficulty; ${num(fcShown)} full combos (filled dots), ${num(pts.ii.length - fcShown)} broken combos (rings). ` +
			`The same plays are listed in the table below.`
	);
</script>

<div class="space-y-3">
	<div class="flex flex-wrap items-center gap-2">
		<StatusSelect bind:value={xKey} items={xItems} label="Horizontal axis" class="min-w-0 flex-1 sm:w-40 sm:flex-none" />
		<StatusSelect bind:value={yKey} items={yItems} label="Vertical axis" class="min-w-0 flex-1 sm:w-40 sm:flex-none" />
		<span class="text-muted-foreground w-full text-sm tabular-nums sm:ml-auto sm:w-auto" aria-live="polite">
			<span class="text-foreground font-medium">{num(pts.ii.length)}</span> of {num(total)} plays
		</span>
	</div>

	<div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs" role="group" aria-label="Difficulty (click to filter)">
		{#each DIFFS as name, d (d)}
			<button
				type="button"
				class="hover:bg-muted inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 transition-opacity"
				class:opacity-40={diff >= 0 && diff !== d}
				aria-pressed={diff === d}
				title={diff === d ? 'Show all difficulties' : `Only ${name}`}
				onclick={() => pickDiff(d)}
			>
				<span class="size-2.5 rounded-full" style:background="var(--diff-{DIFF_KEYS[d]})"></span>
				<span>{name}</span>
				<span class="text-muted-foreground tabular-nums">{num(counts[d])}</span>
			</button>
		{/each}
		<span class="text-muted-foreground inline-flex items-center gap-3 sm:ml-auto">
			<span class="inline-flex items-center gap-1.5"><span class="bg-muted-foreground/60 size-2.5 rounded-full"></span>Full combo</span>
			<span class="inline-flex items-center gap-1.5"><span class="border-muted-foreground size-2.5 rounded-full border-[1.5px]"></span>Combo broken</span>
		</span>
	</div>

	<div class="relative select-none" bind:clientWidth={w} style:height="{h}px">
		{#if w}
			<canvas bind:this={canvas} class="absolute inset-0" style:width="{w}px" style:height="{h}px" aria-hidden="true"></canvas>
			<!-- the table below is the keyboard route to the same plays -->
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
			<svg
				class="plot absolute inset-0"
				width={w}
				height={h}
				role="img"
				aria-label={summary}
				onpointermove={onmove}
				onpointerleave={(e) => e.pointerType === 'mouse' && (hover = -1)}
				{onclick}
				style:cursor={hi >= 0 && !touch ? 'pointer' : 'crosshair'}
			>
				{#each ys.ticks as t (t)}
					<line class="grid" x1={M.l} x2={M.l + iw} y1={py(t)} y2={py(t)} />
					<text class="axis" x={M.l - 7} y={py(t)} dy="0.32em" text-anchor="end">{Y.tick(t, ys.step)}</text>
				{/each}
				{#each xs.ticks as t (t)}
					<line class="grid" x1={px(t)} x2={px(t)} y1={M.t} y2={M.t + ih} />
					<text class="axis" x={px(t)} y={M.t + ih + 16} text-anchor="middle">{X.tick(t, xs.step)}</text>
				{/each}
				<text class="axis title" x={M.l + iw} y={h - 4} text-anchor="end">{X.label} →</text>
				<text class="axis title" x={M.l - 7} y={10} text-anchor="start">↑ {Y.label}</text>
				{#if tip}
					<circle class="ring" cx={tip.x} cy={tip.y} r={rad + 4} style:--c="var(--diff-{DIFF_KEYS[L.diff[hi]]})" />
				{/if}
			</svg>
			{#if pts.ii.length === 0}
				<p class="text-muted-foreground pointer-events-none absolute inset-0 grid place-items-center text-sm">No plays match.</p>
			{/if}
			{#if tip}
				<div
					class="bg-foreground text-background pointer-events-none absolute z-10 w-max max-w-64 rounded-md px-3 py-2 text-xs shadow-md"
					style:left="{tip.left ? tip.x - 12 : tip.x + 12}px"
					style:top="{Math.min(Math.max(tip.y - 40, 0), h - 150)}px"
					style:transform={tip.left ? 'translateX(-100%)' : null}
				>
					<div class="font-semibold leading-snug">{L.song[hi]}</div>
					<div class="opacity-70">{L.mapper[hi]}</div>
					<div class="mt-1 flex items-center gap-1.5">
						<span class="h-0.5 w-3 rounded-full" style:background="var(--diff-{DIFF_KEYS[L.diff[hi]]})"></span>
						{DIFFS[L.diff[hi]]}{#if L.ssStars[hi] != null} · SS {stars(L.ssStars[hi])}★{/if}{#if L.blStars[hi] != null} · BL {stars(L.blStars[hi])}★{/if}
					</div>
					<dl class="mt-1.5 grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5 tabular-nums">
						{#if R.acc[hi] != null}<dt class="opacity-70">SS accuracy</dt><dd class="text-right font-semibold">{pct(R.acc[hi])}</dd>{/if}
						{#if RB?.acc[hi] != null}<dt class="opacity-70">BL accuracy</dt><dd class="text-right font-semibold">{pct(RB.acc[hi])}</dd>{/if}
						{#if R.pp[hi] != null}<dt class="opacity-70">SS PP</dt><dd class="text-right font-semibold">{pp(R.pp[hi])}</dd>{/if}
						{#if RB?.pp[hi] != null}<dt class="opacity-70">BL PP</dt><dd class="text-right font-semibold">{pp(RB.pp[hi])}</dd>{/if}
						<dt class="opacity-70">Misses</dt><dd class="text-right font-semibold">{num(hp?.misses[hi])}</dd>
						<dt class="opacity-70">Combo</dt>
						<dd class="text-right font-semibold">{hp?.fc[hi] ? 'FC' : `${num(hp?.combo[hi])}/${num(L.notes[hi])}`}</dd>
						{#if xKey === 'nps'}<dt class="opacity-70">Notes/sec</dt><dd class="text-right font-semibold">{(L.notes[hi] / L.duration[hi]).toFixed(2)}</dd>{/if}
						{#if xKey === 'duration'}<dt class="opacity-70">Length</dt><dd class="text-right font-semibold">{duration(L.duration[hi])}</dd>{/if}
					</dl>
					<div class="mt-1.5 opacity-60">{touch ? 'Tap again to open the map' : 'Click to open the map'}</div>
				</div>
			{/if}
		{/if}
	</div>

	<div class="text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
		{#if yKey === 'misses'}<span>Dots are spread a little vertically so equal miss counts don't stack.</span>{/if}
		{#if clipped > 0 || (fullRange && Y.cap === 1)}
			<span>
				{#if !fullRange}{num(clipped)} {clipped === 1 ? 'play' : 'plays'} below {Y.tick(ys.lo, ys.step)} sit on the bottom edge.{/if}
			</span>
			<Toggle size="sm" variant="outline" bind:pressed={fullRange} aria-label="Show the full accuracy range">Full range</Toggle>
		{/if}
	</div>
</div>

<style>
	.plot {
		touch-action: pan-y pinch-zoom;
	}
	.grid {
		stroke: var(--border);
	}
	.axis {
		fill: var(--muted-foreground);
		font-size: 11px;
		font-family: var(--font-mono);
	}
	.axis.title {
		font-family: inherit;
	}
	.ring {
		fill: none;
		stroke: var(--c);
		stroke-width: 2;
	}
</style>
