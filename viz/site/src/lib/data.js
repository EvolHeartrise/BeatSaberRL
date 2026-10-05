// Client-side access to the static dataset built by scripts/build-data.mjs.
// Every file is fetched once per session and shared between pages.
// levels.json is the unified catalog: one row per level across ScoreSaber
// and BeatLeader; each model's SS results (results/<hash>.json) and BL
// results (results/<hash>b.json, BL-benched models only) are aligned to it.

/**
 * @typedef {{lo: number, hi: number | null, n: number, acc: number | null, fcRate: number | null}} StarBucket
 * @typedef {{
 *   resultsFile: string, totalPp: number, unweightedPp: number,
 *   levels: number, fcs: number, meanAcc: number, goodCutRate: number,
 *   misses: number, badCuts: number, bombsHit: number, stars: StarBucket[], tau?: number
 * }} BoardStats
 * @typedef {BoardStats & {
 *   id: string, modelHash: string, weightsHash?: string, tickHz: number, actionRepeat: number, updated: string,
 *   color: string,
 *   replays?: Record<string, string>,
 *   bl?: BoardStats & {failed: number, coverage: {played: number, catalog: number}}
 * }} Model
 * @typedef {{ssId: (number | null)[], blId: (string | null)[], hash: string[], song: string[], mapper: string[],
 *   diff: number[], chars: string[], char: number[], duration: number[], notes: number[],
 *   ssStars: (number | null)[], maxPp: (number | null)[], maxScore: (number | null)[],
 *   blStars: (number | null)[], passRating: (number | null)[], accRating: (number | null)[], techRating: (number | null)[],
 *   predAcc: (number | null)[], blMaxScore: (number | null)[]}} Levels
 * @typedef {{score: number[], acc: number[], pp: number[], fc: number[],
 *   misses: number[], bad: number[], bombs: number[], combo: number[],
 *   passPp?: number[], accPp?: number[], techPp?: number[], failed?: number[], tau?: (number | null)[]}} Results
 */

import { PALETTE } from './palette.js';

/** @type {Map<string, Promise<any>>} */
const cache = new Map();

// models.json is always revalidated; every other file is fetched under the
// dataset's `generated` stamp (?v=...), so a new dataset is a new URL and no
// browser or edge cache can hand an older (or a failed) copy of a file to a
// newer page. A non-JSON answer (the SPA shell served for a missing file) is
// reported as such instead of surfacing as a bare JSON parse error.
/** @param {typeof fetch} f @param {string} url @param {RequestInit} [init] */
function get(f, url, init) {
	let p = cache.get(url);
	if (!p) {
		p = f(url, init).then((r) => {
			if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
			const type = r.headers.get('content-type') ?? '';
			if (!type.includes('json')) throw new Error(`${url}: expected JSON, got ${type || 'no content type'}`);
			return r.json();
		});
		p.catch(() => cache.delete(url));
		cache.set(url, p);
	}
	return p;
}

/** @param {typeof fetch} f @returns {Promise<{generated: string, models: Model[]}>} */
export async function loadModels(f) {
	const d = await get(f, '/data/models.json', { cache: 'no-cache' });
	// build-data.mjs assigns each model its color; older datasets without
	// one fall back to the palette in rank order
	d.models.forEach((/** @type {Model} */ m, /** @type {number} */ k) => {
		m.color ??= PALETTE[k % PALETTE.length];
	});
	return d;
}

/** @param {typeof fetch} f @param {string} file */
async function getVersioned(f, file) {
	const { generated } = await loadModels(f);
	return get(f, `/data/${file}?v=${encodeURIComponent(generated)}`);
}

/** @param {typeof fetch} f @returns {Promise<Levels>} */
export const loadLevels = (f) => getVersioned(f, 'levels.json');

/** @param {typeof fetch} f @param {Model} model @returns {Promise<Results>} */
export const loadResults = (f, model) => getVersioned(f, `results/${encodeURIComponent(model.resultsFile)}.json`);

/** @param {typeof fetch} f @param {Model[]} models @returns {Promise<Results[]>} */
export const loadAllResults = (f, models) => Promise.all(models.map((m) => loadResults(f, m)));

// ---- BeatLeader --------------------------------------------------------------
// results/<hash>b.json (hex names, like the ScoreSaber files), aligned to the
// same levels.json rows; null for a model not benched on BeatLeader.

/** @param {typeof fetch} f @param {Model} model @returns {Promise<Results | null>} */
export const loadResultsBl = (f, model) =>
	model.bl ? getVersioned(f, `results/${encodeURIComponent(model.bl.resultsFile)}.json`) : Promise.resolve(null);

/** @param {typeof fetch} f @param {Model[]} models @returns {Promise<(Results | null)[]>} */
export const loadAllResultsBl = (f, models) => Promise.all(models.map((m) => loadResultsBl(f, m)));

/** The map page of a level: /maps/<ScoreSaber id> when ScoreSaber ranks it,
 * else /maps/bl/<BeatLeader id>. @param {Levels} L @param {number} i */
export const mapHref = (L, i) => (L.ssId[i] != null ? `/maps/${L.ssId[i]}` : `/maps/bl/${L.blId[i]}`);

/** A model's play on level `i` that the shared columns (score, misses, combo)
 * describe: its ScoreSaber play, else (a BeatLeader-only level) its BL play.
 * @param {Results} R @param {Results | null | undefined} RB @param {number} i
 * @returns {Results | null} */
export const playOf = (R, RB, i) => (R.score[i] != null ? R : RB && RB.score[i] != null ? RB : null);

// Replays: models.json `replays` maps a level index to the 16-hex id of its
// replay on replays.flee.systems (the hosted 3D viewer). Opaque on purpose:
// ad-block lists match URL substrings, so no model id rides in the path.
export const REPLAY_HOST = 'https://replays.flee.systems';

/** The viewer URL for model `m`'s play on level `i`, or null when it has no replay.
 * @param {Model} m @param {number} i */
export function watchUrl(m, i) {
	const id = m.replays?.[i];
	return id && /^[0-9a-f]{16}$/.test(id) ? `${REPLAY_HOST}/?replay=/r/${id}.rpl.gz` : null;
}
