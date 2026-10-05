// Shared map filtering + sorting for the level tables.

/**
 * `ranked` narrows to the levels ranked on a board ('ss' / 'bl': ranked there,
 * 'both', 'ssonly' / 'blonly': ranked there and not on the other board).
 * The star range uses that board's stars: BeatLeader's for 'bl' / 'blonly',
 * else ScoreSaber's (the scales differ and are never mixed), so a star range
 * leaves out the levels the scale's board does not rank.
 * @typedef {'any' | 'ss' | 'bl' | 'both' | 'ssonly' | 'blonly'} Ranked
 * @typedef {{q: string, diff: number, ranked: Ranked, minStars: number | null, maxStars: number | null}} LevelFilter
 */

/** @returns {LevelFilter} */
export const emptyFilter = () => ({ q: '', diff: -1, ranked: 'any', minStars: null, maxStars: null });

/** the star scale a filter's star range uses @param {LevelFilter} f @returns {'ss' | 'bl'} */
export const starScale = (f) => (f.ranked === 'bl' || f.ranked === 'blonly' ? 'bl' : 'ss');

/**
 * Indices of levels matching the filter.
 * @param {import('./data.js').Levels} L
 * @param {LevelFilter} f
 * @param {(i: number) => boolean} [extra]
 */
export function filterLevels(L, f, extra) {
	const q = f.q.trim().toLowerCase();
	const stars = starScale(f) === 'bl' ? L.blStars : L.ssStars;
	const useStars = f.minStars != null || f.maxStars != null;
	const out = [];
	for (let i = 0; i < L.hash.length; i++) {
		if (f.diff >= 0 && L.diff[i] !== f.diff) continue;
		const ss = L.ssId[i] != null,
			bl = L.blId[i] != null;
		const r = f.ranked;
		if (r !== 'any' && !(r === 'ss' ? ss : r === 'bl' ? bl : r === 'both' ? ss && bl : r === 'ssonly' ? ss && !bl : bl && !ss)) continue;
		if (useStars) {
			const s = stars[i];
			if (s == null) continue;
			if (f.minStars != null && s < f.minStars) continue;
			if (f.maxStars != null && s > f.maxStars) continue;
		}
		if (q && !L.song[i].toLowerCase().includes(q) && !L.mapper[i].toLowerCase().includes(q)) continue;
		if (extra && !extra(i)) continue;
		out.push(i);
	}
	return out;
}

/**
 * Sort indices in place by a numeric or string column accessor; nulls last.
 * @param {number[]} idx
 * @param {(i: number) => number | string | null} get
 * @param {boolean} desc
 */
export function sortIdx(idx, get, desc) {
	const s = desc ? -1 : 1;
	return idx.sort((a, b) => {
		const va = get(a),
			vb = get(b);
		if (va == null) return vb == null ? 0 : 1;
		if (vb == null) return -1;
		return (typeof va === 'string' ? va.localeCompare(/** @type {string} */ (vb)) : va - /** @type {number} */ (vb)) * s;
	});
}

/**
 * A sorted copy of `rows` (nulls last), for the small tables.
 * @template T
 * @param {T[]} rows
 * @param {(r: T) => number | string | null} get
 * @param {boolean} desc
 */
export function sortRows(rows, get, desc) {
	const idx = sortIdx(rows.map((_, i) => i), (i) => get(rows[i]), desc);
	return idx.map((i) => rows[i]);
}
