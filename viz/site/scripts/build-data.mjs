// Compile ScoreSaber bench results (python/bench_scoresaber.py output) into
// the site's static dataset under static/data/.
//
//   node scripts/build-data.mjs [--src DIR ...] [--replays MANIFEST]
//
// Sources: the repo's tracked reference benches (python/bench/) plus every
// `*/bench/*.json` under the laptop run archive (~/beat-arena-volume-backup),
// when it exists. Extra --src dirs are scanned the same way (recursively).
//
// Only CURRENT-ERA benches are published — results across eras are different
// physics and must never share a table (README: "never merge across rules"):
//   - scoring_version 3 or 4 (real NJS speeds, cut-plane accuracy; v4,
//     2026-10-02, fixes arc-tail grants and arc-attached chain heads on
//     ~20% of levels: same physics, a corrected rule, so both are shown)
//   - arch.z_asinh > 0 (env v48 asinh z encoding; legacy-clamp checkpoints
//     were removed 2026-09-22)
//   - sim "rust" (the reference path; CUDA-sim benches are not
//     interchangeable with it)
//   - the whole ranked catalog evaluated (no partial star-range runs)
//   - the training control rate: tick_hz 30, action_repeat 1, no obs_window (exp/tick-rate
//     benches at other rates share the model hash and must never replace
//     or duplicate its published bench; files without the keys are 30 / 1)
// The same file often exists in several archive folders (campaign dir and
// repo-bench/); duplicates are keyed by basename and must carry the same
// model hash. Bench file names are NOT published (see publicId below).
//
// Output (all JSON, column-oriented to stay small):
//   levels.json            the UNIFIED catalog: one row per level (song hash +
//                          difficulty + characteristic) across both boards,
//                          every ScoreSaber-ranked level plus the BeatLeader-
//                          only ones. `ssId` / `blId` are the two boards'
//                          leaderboard ids (null = not ranked there); SS
//                          columns (ssStars, maxPp, maxScore) and BL columns
//                          (blStars, pass/acc/techRating, predAcc,
//                          blMaxScore) are kept apart: different scales,
//                          never mixed
//   models.json            one summary per bench (ranked by SS weighted PP)
//   results/<hash16>.json  per-bench ScoreSaber columns aligned to the
//                          levels.json rows (null where the bench has no SS
//                          play; file name = models.json resultsFile)
//
// BeatLeader. A `--catalog both` bench (header `catalog: "both"`) plays the
// union of the ScoreSaber and BeatLeader ranked catalogs once; each record
// carries the play plus `ss` and `bl` sub-results (null where that board
// does not rank the level). Its `ss` side feeds the ScoreSaber files above
// exactly like a ScoreSaber-only bench; its `bl` side also produces:
//   models.json `bl`        the model's BeatLeader totals and stats (absent
//                           = not benched on BeatLeader)
//   results/<hash16>b.json  BL columns, also aligned to the levels.json rows
// BL per-level play stats come from `bl_play` when the record has one (a
// composite that picked another run for the BL view), else the record's
// own play; BL accuracy is `bl.accuracy` (score / BeatLeader's maxScore).
//
// Composites (header `composite`): a per-level best-of over several runs of
// one checkpoint, published under its public id like any other model (user,
// 2026-10-01: "just called v61", no run / body details). The input runs and
// each level's source run (`ss_from` / `bl_from`) are used here only, to match
// replays to the run each pick was played in; models.json and the results
// files carry none of it. A model's `color` is assigned here (palette in
// rank order; composites from the palette's end, so adding one never
// recolors the others).
//
// Replays (the "Watch" links): the replay store's manifest.json, written by
// viz/replays-site/scripts/publish_replays.py (default
// ~/beat-arena-volume-backup/replays-site/manifest.json, or --replays PATH;
// absent = no links). A replay is linked from a published model only when it
// was made in that model's exact variant — same weights hash, tick_hz,
// action_repeat and obs_window — on a level the published bench played, with
// that bench entry's score, and its file is in the store. models.json then
// carries `replays: {<levels.json row>: <16-hex file id>}`, served by
// replays.flee.systems as /r/<id>.rpl.gz. A composite's replays (manifest
// entry `composite: true`, same pseudo model hash and weights hash) carry
// the run they were recorded in (tickHz / actionRepeat / obsWindow /
// tauScale) and are matched per level against the composite's PICKS: a
// replay links when its run is the picked input run (rate and body) and its
// score is that pick's. One link per row: the ScoreSaber pick's replay (the
// play the row's score, misses and combo describe), else the BeatLeader
// pick's.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import { PALETTE } from '../src/lib/palette.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, '..');
const repo = path.resolve(site, '../..');
const outDir = path.join(site, 'static/data');

const srcDirs = [path.join(repo, 'python/bench'), path.join(os.homedir(), 'beat-arena-volume-backup')];
let replayManifest = path.join(os.homedir(), 'beat-arena-volume-backup/replays-site/manifest.json');
for (let i = 2; i < process.argv.length; i++) {
	if (process.argv[i] === '--src') srcDirs.push(path.resolve(process.argv[++i]));
	else if (process.argv[i] === '--replays') replayManifest = path.resolve(process.argv[++i]);
}

function* benchFiles(dir) {
	if (!fs.existsSync(dir)) return;
	for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
		const p = path.join(dir, ent.name);
		if (ent.isDirectory()) {
			if (ent.name === 'node_modules' || ent.name.startsWith('.')) continue;
			yield* benchFiles(p);
		} else if (ent.name.endsWith('.json') && path.basename(dir) === 'bench') {
			yield p;
		}
	}
}

const DIFFS = ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus'];
const STAR_BUCKETS = [0, 2, 4, 6, 8, 10, 12];

// Public-name overrides (scripts/public-names.json): a bench basename maps
// to its published id, or to null to keep it off the site (e.g. the #9
// ablation arms). Anything not listed keeps the version-prefix rule below.
/** @type {Record<string, string | null>} */
const nameOverrides = JSON.parse(fs.readFileSync(path.join(here, 'public-names.json'), 'utf8'));

/** the bench ran the standard 30 Hz control rate, no action repeat, no obs window */
const defaultRate = (/** @type {any} */ d) =>
	(d.tick_hz ?? 30) === 30 && (d.action_repeat ?? 1) === 1 && !d.obs_window;

// A bench at another control rate (exp/tick-rate) is published only when
// public-names.json gives it an id of its own; it then lists as a separate
// model (same weights, different control rate), never merged with the
// 30 Hz entry.
const unpublishedPrefixes = /** @type {string[]} */ (/** @type {any} */ (nameOverrides)._unpublished_prefixes ?? []);

/** @param {any} d @param {string} id */
function currentEra(d, id) {
	if (unpublishedPrefixes.some((pre) => id.startsWith(pre))) return false;
	return (
		d &&
		Array.isArray(d.levels) &&
		(d.catalog ?? 'scoresaber') !== 'beatleader' &&
		(d.scoring_version === 3 || d.scoring_version === 4) &&
		(d.arch?.z_asinh ?? 0) > 0 &&
		d.sim === 'rust' &&
		d.n_skipped === 0 &&
		(defaultRate(d) || typeof nameOverrides[id] === 'string')
	);
}

/** one published entry per (weights, control rate, repeat, obs window) */
const variantKey = (/** @type {any} */ d) =>
	`${d.model_hash}|${d.tick_hz ?? 30}|${d.action_repeat ?? 1}|${d.obs_window ? 1 : 0}`;

// ---- collect -------------------------------------------------------------
/** @type {Map<string, {file: string, d: any}>} */
const benches = new Map();
let scanned = 0;
for (const dir of srcDirs) {
	for (const file of benchFiles(dir)) {
		scanned++;
		// the header sits before the levels array; skip old-era files cheaply
		const raw = fs.readFileSync(file, 'utf8');
		if (!/"z_asinh":\s*[1-9]/.test(raw.slice(0, 4000))) continue;
		let d;
		try {
			d = JSON.parse(raw);
		} catch {
			console.warn(`skip (bad json): ${file}`);
			continue;
		}
		const id = path.basename(file, '.json');
		if (!currentEra(d, id)) continue;
		const prev = benches.get(id);
		if (prev) {
			if (prev.d.model_hash !== d.model_hash)
				throw new Error(`${id}: two files with different models:\n  ${prev.file}\n  ${file}`);
			if ((d.updated ?? '') <= (prev.d.updated ?? '')) continue;
		}
		benches.set(id, { file, d });
	}
}
if (!benches.size) throw new Error('no current-era bench files found');

for (const id of [...benches.keys()]) {
	if (nameOverrides[id] === null) {
		console.log(`  (unpublished by public-names.json: ${id})`);
		benches.delete(id);
	}
}

// One entry per model: a checkpoint benched under several body presets
// (same weights hash) publishes only its best bench by weighted PP.
/** @type {Map<string, string>} */
const bestOfModel = new Map();
for (const [file, { d }] of benches) {
	const cur = bestOfModel.get(variantKey(d));
	if (!cur || d.total_weighted_pp > /** @type {any} */ (benches.get(cur)).d.total_weighted_pp)
		bestOfModel.set(variantKey(d), file);
}
for (const [file, { d }] of [...benches]) {
	if (bestOfModel.get(variantKey(d)) !== file) {
		console.log(`  (dropped ${file}: same model as ${bestOfModel.get(variantKey(d))}, lower PP)`);
		benches.delete(file);
	}
}

// ---- per-board views of each bench ------------------------------------------
const isBoth = (/** @type {any} */ d) => d.catalog === 'both';

/** The bench's ScoreSaber-ranked plays in the ScoreSaber-only record shape.
 * @param {any} d @returns {any[]} */
function ssLevelsOf(d) {
	if (!isBoth(d)) return d.levels;
	return d.levels
		.filter((/** @type {any} */ l) => l.ss)
		.map((/** @type {any} */ l) => ({
			...l,
			leaderboard_id: l.ss.leaderboard_id,
			stars: l.ss.stars,
			max_pp: l.ss.max_pp,
			api_max_score: l.ss.api_max_score,
			accuracy: l.ss.accuracy ?? l.accuracy,
			pp: l.ss.pp
		}));
}

/** The bench's BeatLeader-ranked plays: level fields, `bl` and the BL play.
 * @param {any} d @returns {any[]} */
function blLevelsOf(d) {
	if (!isBoth(d)) return [];
	return d.levels.filter((/** @type {any} */ l) => l.bl).map((/** @type {any} */ l) => ({ ...l, play: l.bl_play ?? l }));
}


// BeatLeader view of ScoreSaber-only benches (python/tools/bench_add_bl.py,
// 2026-10-01): a sidecar <blDerivedDir>/<bench basename>.json prices the
// bench's own plays on the maps both leaderboards rank by BeatLeader's rules
// (same play: score, misses, combo stay the SS record's). Such a model never
// played the BeatLeader-only maps, so its BL totals cover the shared maps
// only; models.json `bl.coverage` = {played, catalog} says so.
const blDerivedDir = path.join(os.homedir(), 'beat-arena-volume-backup/bl-derived');

/** the BL plays of an SS-only bench, from its sidecar (null = none / mismatched)
 * @param {string} id bench basename @param {any} d */
function derivedBl(id, d) {
	const f = path.join(blDerivedDir, `${id}.json`);
	if (!fs.existsSync(f)) return null;
	const x = JSON.parse(fs.readFileSync(f, 'utf8'));
	const same =
		x.model_hash === d.model_hash &&
		(x.tick_hz ?? 30) === (d.tick_hz ?? 30) &&
		(x.action_repeat ?? 1) === (d.action_repeat ?? 1) &&
		!!x.obs_window === !!d.obs_window;
	if (!same) {
		console.warn(`  (BL sidecar ${f} does not match the bench ${id} (hash / rate): skipped)`);
		return null;
	}
	const bySs = new Map(d.levels.filter((/** @type {any} */ l) => !l.skipped).map((/** @type {any} */ l) => [l.leaderboard_id, l]));
	const out = [];
	for (const s of x.levels) {
		const l = bySs.get(s.ss_leaderboard_id);
		if (!l) throw new Error(`${f}: SS level ${s.ss_leaderboard_id} not played in ${id}`);
		out.push({ ...l, ss: { leaderboard_id: l.leaderboard_id }, bl: s.bl, play: l });
	}
	return { levels: out, weighted: x.bl_total_weighted_pp, unweighted: x.bl_total_unweighted_pp };
}

for (const [id, b] of benches) {
	/** @type {any} */ (b).ss = ssLevelsOf(b.d);
	/** @type {any} */ (b).bl = blLevelsOf(b.d);
	if (!isBoth(b.d)) {
		const x = derivedBl(id, b.d);
		if (x) {
			/** @type {any} */ (b).bl = x.levels;
			/** @type {any} */ (b).blTotals = x;
		}
	}
}

// ---- catalog: the union of every bench's levels ---------------------------
// ScoreSaber-only benches first, so a level's catalog row comes from the same
// record as before catalog-both benches existed
const catalogOrder = [...benches.values()].sort((a, b) => Number(isBoth(a.d)) - Number(isBoth(b.d)));
/** @type {Map<number, any>} */
const catalog = new Map();
for (const b of catalogOrder) {
	for (const l of /** @type {any} */ (b).ss) {
		if (!catalog.has(l.leaderboard_id)) catalog.set(l.leaderboard_id, l);
	}
}
// ---- BeatLeader catalog --------------------------------------------------------
/** @type {Map<string, any>} */
const blCatalog = new Map();
// catalog-both benches first: BL level facts come from the bench's own BL
// catalog record, not from a derived (sidecar) one
for (const b of [...catalogOrder].reverse()) {
	for (const l of /** @type {any} */ (b).bl) {
		if (!blCatalog.has(l.bl.leaderboard_id)) blCatalog.set(l.bl.leaderboard_id, l);
	}
}

// ---- the unified catalog (levels.json) ----------------------------------------
// One row per level (song hash + difficulty + characteristic) across both
// boards: every ScoreSaber-ranked level, plus the BeatLeader-ranked levels
// ScoreSaber does not rank. A BL record's `ss` sub-result names the SS row of
// the same level. SS fields (stars, max PP, max score, song facts) come from
// the ScoreSaber catalog record exactly as before; BL fields from the BL one.
/** @type {{ss: any | null, bl: any | null}[]} */
const rows = [...catalog.values()].map((l) => ({ ss: l, bl: null }));
/** @type {Map<number, {ss: any | null, bl: any | null}>} */
const rowOfSs = new Map(rows.map((r) => [r.ss.leaderboard_id, r]));
for (const l of blCatalog.values()) {
	const r = l.ss ? rowOfSs.get(l.ss.leaderboard_id) : undefined;
	if (r) {
		if (r.bl) throw new Error(`ScoreSaber level ${l.ss.leaderboard_id} linked to two BeatLeader levels`);
		r.bl = l;
	} else rows.push({ ss: null, bl: l });
}
const facts = (/** @type {{ss: any | null, bl: any | null}} */ r) => r.ss ?? r.bl;
const charOf = (/** @type {any} */ l) => l.characteristic ?? 'Standard';
rows.sort((x, y) => {
	const a = facts(x),
		b = facts(y);
	return (
		a.song.localeCompare(b.song) ||
		a.song_hash.localeCompare(b.song_hash) ||
		DIFFS.indexOf(a.difficulty) - DIFFS.indexOf(b.difficulty) ||
		charOf(a).localeCompare(charOf(b))
	);
});
/** ScoreSaber leaderboard id -> unified row */
const index = new Map(rows.flatMap((r, i) => (r.ss ? [[r.ss.leaderboard_id, i]] : [])));
/** BeatLeader leaderboard id -> unified row */
const blIndex = new Map(rows.flatMap((r, i) => (r.bl ? [[r.bl.bl.leaderboard_id, i]] : [])));
const nLevels = rows.length;
const r4 = (/** @type {number | null | undefined} */ x) => (x == null ? null : Math.round(x * 1e4) / 1e4);
const chars = [...new Set(rows.map((r) => charOf(facts(r))))].sort();
const levelCols = {
	ssId: rows.map((r) => r.ss?.leaderboard_id ?? null),
	blId: rows.map((r) => r.bl?.bl.leaderboard_id ?? null),
	hash: rows.map((r) => facts(r).song_hash),
	song: rows.map((r) => facts(r).song),
	mapper: rows.map((r) => facts(r).mapper),
	diff: rows.map((r) => DIFFS.indexOf(facts(r).difficulty)),
	chars,
	char: rows.map((r) => chars.indexOf(charOf(facts(r)))),
	duration: rows.map((r) => facts(r).duration_s),
	notes: rows.map((r) => facts(r).n_notes),
	// ScoreSaber (null = not ranked on ScoreSaber)
	ssStars: rows.map((r) => r.ss?.stars ?? null),
	maxPp: rows.map((r) => r.ss?.max_pp ?? null),
	maxScore: rows.map((r) => r.ss?.max_score ?? null),
	// BeatLeader (null = not ranked on BeatLeader); a different star scale
	blStars: rows.map((r) => r4(r.bl?.bl.stars)),
	passRating: rows.map((r) => r4(r.bl?.bl.pass_rating)),
	accRating: rows.map((r) => r4(r.bl?.bl.acc_rating)),
	techRating: rows.map((r) => r4(r.bl?.bl.tech_rating)),
	predAcc: rows.map((r) => r4(r.bl?.bl.predicted_acc)),
	blMaxScore: rows.map((r) => (r.bl ? (r.bl.bl.api_max_score ?? r.bl.max_score) : null))
};

// ---- per-bench results + summaries ---------------------------------------
const round = (x, n) => Math.round(x * 10 ** n) / 10 ** n;

// the published colors, read before the output dir is cleared (sticky
// colors — see the color block below)
/** @type {Record<string, string>} */
const priorColors = {};
try {
	for (const m of JSON.parse(fs.readFileSync(path.join(outDir, 'models.json'), 'utf8')).models)
		if (m.color) priorColors[m.id] = m.color;
} catch {
	// no published models.json yet: every model is new
}
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(path.join(outDir, 'results'), { recursive: true });

// Public names hide the bench file name and the run's full name (its suffix
// is the evaluation body preset, its middle the run's recipe; neither is
// published): only the run's version prefix, e.g. "v56" from
// "v56_6blk_g965_nstuned_cd400", plus "-N" when several published benches
// share a prefix (numbered in bench-file-name order).
/** @type {Map<string, string[]>} */
const byName = new Map();
for (const [file, { d }] of [...benches].sort(([a], [b]) => a.localeCompare(b))) {
	const name = nameOverrides[file] ?? d.checkpoint_name.split('/')[0].split('_')[0];
	byName.set(name, [...(byName.get(name) ?? []), file]);
}
/** @type {Map<string, string>} */
const publicId = new Map();
for (const [name, files] of byName) {
	files.forEach((file, k) => publicId.set(file, files.length > 1 ? `${name}-${k + 1}` : name));
}

// ---- replays (see header) ---------------------------------------------------
// replays.flee.systems serves the replays from R2 (2026-10-03): a file is
// linkable when the store still has it or the uploader's ledger records it
// (viz/replays-site/scripts/upload_r2.mjs --prune drops local copies)
const r2Ledger = (() => {
	const p = path.join(path.dirname(replayManifest), 'r2-ledger.json');
	return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : {};
})();
/** @param {string} file */
const replayServed = (file) => !!r2Ledger[file] || fs.existsSync(path.join(path.dirname(replayManifest), file));
/** @type {{models: Record<string, any>}} */
const replayStore = fs.existsSync(replayManifest)
	? JSON.parse(fs.readFileSync(replayManifest, 'utf8'))
	: (console.warn(`  (no replay manifest at ${replayManifest}: no Watch links)`), { models: {} });
const REPLAY_FILE = /^r\/([0-9a-f]{16})\.rpl\.gz$/;

/** the same body: two tau_scale lists (absent = all 1.0)
 * @param {number[] | null | undefined} a @param {number[] | null | undefined} b */
const sameBody = (a, b) => {
	const x = a ?? new Array(16).fill(1),
		y = b ?? new Array(16).fill(1);
	return x.length === y.length && x.every((v, k) => Math.abs(v - y[k]) <= 1e-6);
};

/** A composite's replays, matched against its per-level picks.
 * @param {any} d the composite bench @param {string} id @param {string} name @param {any} m the manifest entry */
function compositeReplays(d, id, name, m) {
	/** @type {Record<string, string>} */
	const ss = {},
		bl = {};
	const runs = new Map(d.composite.inputs.map((/** @type {any} */ r) => [r.file, r]));
	const recs = new Map(d.levels.filter((/** @type {any} */ l) => l.level_key && !l.skipped).map((/** @type {any} */ l) => [l.level_key, l]));
	for (const [fid, r] of Object.entries(m.replays ?? {})) {
		const l = recs.get(r.levelKey);
		const f = REPLAY_FILE.exec(r.file);
		/** the replay is this board's pick: recorded in the picked run, with its score @param {'ss' | 'bl'} board */
		const isPick = (board) => {
			const run = l?.[board] ? runs.get(l[`${board}_from`]) : null;
			const play = board === 'ss' ? l : (l?.bl_play ?? l);
			return (
				!!run &&
				(run.tick_hz ?? 30) === (r.tickHz ?? 30) &&
				(r.actionRepeat ?? 1) === 1 &&
				!r.obsWindow &&
				sameBody(run.tau_scale, r.tauScale) &&
				play.score === r.score
			);
		};
		const boards = /** @type {('ss' | 'bl')[]} */ (['ss', 'bl']).filter(isPick);
		const why = !l
			? 'level not in the published bench'
			: !boards.length
				? `recorded at ${r.tickHz ?? 30} Hz / score ${r.score}: neither pick's run and score`
				: !f
					? `bad file name ${r.file}`
					: !replayServed(r.file)
						? `${r.file} neither in the store nor in R2`
						: null;
		if (why) {
			console.warn(`  (no Watch link: ${id} ${fid} ${r.song}: ${why})`);
			continue;
		}
		const row = l.ss ? index.get(l.ss.leaderboard_id) : blIndex.get(l.bl.leaderboard_id);
		for (const b of boards) (b === 'ss' ? ss : bl)[String(row)] = /** @type {RegExpExecArray} */ (f)[1];
	}
	if (name !== id) console.warn(`  (replays stored as "${name}" match the bench published as "${id}")`);
	return { ...bl, ...ss };
}

/** @param {any} d the published bench @param {string} id its public id */
function replaysOf(d, id) {
	/** @type {Record<string, string>} */
	const out = {};
	for (const [name, m] of Object.entries(replayStore.models)) {
		if (m.composite || d.composite) {
			if (m.composite && d.composite && m.modelHash === d.model_hash && m.weightsHash === d.weights_hash)
				Object.assign(out, compositeReplays(d, id, name, m));
			continue;
		}
		const variant = `${m.modelHash}|${m.tickHz ?? 30}|${m.actionRepeat ?? 1}|${m.obsWindow ? 1 : 0}`;
		if (variant !== variantKey(d)) continue;
		if (name !== id) console.warn(`  (replays stored as "${name}" match the bench published as "${id}")`);
		const played = new Map(ssLevelsOf(d).map((/** @type {any} */ l) => [l.leaderboard_id, l]));
		for (const [lb, r] of Object.entries(m.replays ?? {})) {
			const l = played.get(Number(lb));
			const f = REPLAY_FILE.exec(r.file);
			const why = !l
				? 'level not in the published bench'
				: l.score !== r.score
					? `score ${r.score} != bench ${l.score}`
					: !f
						? `bad file name ${r.file}`
						: !replayServed(r.file)
							? `${r.file} neither in the store nor in R2`
							: null;
			if (why) {
				console.warn(`  (no Watch link: ${id} ${lb} ${r.song}: ${why})`);
				continue;
			}
			out[String(index.get(Number(lb)))] = /** @type {RegExpExecArray} */ (f)[1];
		}
	}
	return out;
}

/** a play's star bucket
 * @param {any[]} buckets @param {number} s */
const bucketOf = (buckets, s) => buckets.findLast((b) => s >= b.lo);
const newBuckets = () => STAR_BUCKETS.map((lo, i) => ({ lo, hi: STAR_BUCKETS[i + 1] ?? null, n: 0, acc: 0, fc: 0 }));
const bucketSummary = (/** @type {any[]} */ buckets) =>
	buckets.map((b) => ({
		lo: b.lo,
		hi: b.hi,
		n: b.n,
		acc: b.n ? round(b.acc / b.n, 5) : null,
		fcRate: b.n ? round(b.fc / b.n, 4) : null
	}));

/** BeatLeader columns + summary of a catalog-both bench
 * @param {any} d @param {any[]} bl the bench's BL plays @param {string} resultsFile
 */
function blOf(d, bl, resultsFile, totals) {
	const n = nLevels;
	const col = () => new Array(n).fill(null);
	/** @type {Record<string, any[]>} */
	const cols = {
		score: col(),
		acc: col(),
		pp: col(),
		fc: col(),
		misses: col(),
		bad: col(),
		bombs: col(),
		combo: col(),
		passPp: col(),
		accPp: col(),
		techPp: col(),
		failed: col(),
		tau: col()
	};
	const buckets = newBuckets();
	let fcs = 0,
		accSum = 0,
		misses = 0,
		bad = 0,
		bombs = 0,
		notes = 0,
		good = 0,
		failed = 0,
		tauSum = 0,
		tauN = 0;
	for (const l of bl) {
		const i = /** @type {number} */ (blIndex.get(l.bl.leaderboard_id));
		const p = l.play;
		cols.score[i] = p.score;
		cols.acc[i] = round(l.bl.accuracy, 5);
		cols.pp[i] = round(l.bl.pp, 2);
		cols.fc[i] = p.full_combo ? 1 : 0;
		cols.misses[i] = p.misses;
		cols.bad[i] = p.bad_cuts;
		cols.bombs[i] = p.bombs_hit;
		cols.combo[i] = p.highest_combo;
		cols.passPp[i] = round(l.bl.pass_pp ?? 0, 2);
		cols.accPp[i] = round(l.bl.acc_pp ?? 0, 2);
		cols.techPp[i] = round(l.bl.tech_pp ?? 0, 2);
		cols.failed[i] = l.bl.failed ? 1 : 0;
		if (p.torque_nms != null) {
			cols.tau[i] = round(p.torque_nms, 1);
			tauSum += p.torque_nms;
			tauN++;
		}
		fcs += p.full_combo ? 1 : 0;
		accSum += l.bl.accuracy;
		misses += p.misses;
		bad += p.bad_cuts;
		bombs += p.bombs_hit;
		notes += l.n_notes;
		good += p.good_cuts;
		failed += l.bl.failed ? 1 : 0;
		const b = bucketOf(buckets, l.bl.stars);
		b.n++;
		b.acc += l.bl.accuracy;
		b.fc += p.full_combo ? 1 : 0;
	}
	fs.writeFileSync(path.join(outDir, 'results', `${resultsFile}b.json`), JSON.stringify(cols));
	return {
		resultsFile: `${resultsFile}b`,
		totalPp: round((totals ? totals.weighted : d.bl_total_weighted_pp) ?? 0, 2),
		unweightedPp: round((totals ? totals.unweighted : d.bl_total_unweighted_pp) ?? 0, 2),
		levels: bl.length,
		// the BL maps this model played, of all BeatLeader-ranked ones
		coverage: { played: bl.length, catalog: blIndex.size },
		fcs,
		failed,
		meanAcc: round(accSum / bl.length, 5),
		goodCutRate: round(good / notes, 5),
		misses,
		badCuts: bad,
		bombsHit: bombs,
		// total applied torque, Σ|τ| N·m·s, over every BL play (only when
		// the bench recorded torque for all of them)
		...(tauN && tauN === bl.length ? { tau: round(tauSum, 0) } : {}),
		stars: bucketSummary(buckets)
	};
}

const models = [];
for (const [file, b] of benches) {
	const { d } = b;
	const ssLevels = /** @type {any} */ (b).ss;
	const id = /** @type {string} */ (publicId.get(file));
	const n = nLevels;
	const cols = {
		score: new Array(n).fill(null),
		acc: new Array(n).fill(null),
		pp: new Array(n).fill(null),
		fc: new Array(n).fill(null),
		misses: new Array(n).fill(null),
		bad: new Array(n).fill(null),
		bombs: new Array(n).fill(null),
		combo: new Array(n).fill(null),
		tau: new Array(n).fill(null)
	};
	const buckets = STAR_BUCKETS.map((lo, i) => ({
		lo,
		hi: STAR_BUCKETS[i + 1] ?? null,
		n: 0,
		acc: 0,
		fc: 0
	}));
	let fcs = 0,
		accSum = 0,
		misses = 0,
		bad = 0,
		bombs = 0,
		notes = 0,
		good = 0,
		tauSum = 0,
		tauN = 0;
	for (const l of ssLevels) {
		const i = index.get(l.leaderboard_id);
		cols.score[i] = l.score;
		cols.acc[i] = round(l.accuracy, 5);
		cols.pp[i] = round(l.pp, 2);
		cols.fc[i] = l.full_combo ? 1 : 0;
		cols.misses[i] = l.misses;
		cols.bad[i] = l.bad_cuts;
		cols.bombs[i] = l.bombs_hit;
		cols.combo[i] = l.highest_combo;
		if (l.torque_nms != null) {
			cols.tau[i] = round(l.torque_nms, 1);
			tauSum += l.torque_nms;
			tauN++;
		}
		fcs += l.full_combo ? 1 : 0;
		accSum += l.accuracy;
		misses += l.misses;
		bad += l.bad_cuts;
		bombs += l.bombs_hit;
		notes += l.n_notes;
		good += l.good_cuts;
		const b = buckets.findLast((b) => l.stars >= b.lo);
		b.n++;
		b.acc += l.accuracy;
		b.fc += l.full_combo ? 1 : 0;
	}
	// named by weights hash, not by public id: ad/tracker filter lists match
	// substrings of any request URL, and EasyPrivacy's `/v60.js` (Nielsen's
	// imrworldwide tracker) blocked results/v60.json under uBlock Origin.
	// A hex hash carries no word or version-like token for a rule to hit.
	// a non-default control rate shares the weights hash, so it gets the
	// hash of its variant key instead (still plain hex)
	const resultsFile = defaultRate(d)
		? d.model_hash.replace('sha256:', '').slice(0, 16)
		: crypto.createHash('sha256').update(variantKey(d)).digest('hex').slice(0, 16);
	fs.writeFileSync(path.join(outDir, 'results', `${resultsFile}.json`), JSON.stringify(cols));

	const replays = replaysOf(d, id);
	const bl = /** @type {any} */ (b).bl.length ? blOf(d, /** @type {any} */ (b).bl, resultsFile, /** @type {any} */ (b).blTotals) : null;
	models.push({
		id,
		...(Object.keys(replays).length ? { replays } : {}),
		modelHash: d.model_hash,
		...(d.weights_hash && d.weights_hash !== d.model_hash ? { weightsHash: d.weights_hash } : {}),
		resultsFile,
		tickHz: d.tick_hz ?? 30,
		actionRepeat: d.action_repeat ?? 1,
		updated: d.updated,
		totalPp: round(d.total_weighted_pp, 2),
		unweightedPp: round(d.total_unweighted_pp ?? 0, 2),
		levels: ssLevels.length,
		fcs,
		meanAcc: round(accSum / ssLevels.length, 5),
		goodCutRate: round(good / notes, 5),
		misses,
		badCuts: bad,
		bombsHit: bombs,
		// total applied torque, Σ|τ| N·m·s, over every SS play (benches since
		// 2026-10-01 record it per level; older ones have none)
		...(tauN && tauN === ssLevels.length ? { tau: round(tauSum, 0) } : {}),
		stars: buckets.map((b) => ({
			lo: b.lo,
			hi: b.hi,
			n: b.n,
			acc: b.n ? round(b.acc / b.n, 5) : null,
			fcRate: b.n ? round(b.fc / b.n, 4) : null
		})),
		...(bl ? { bl } : {})
	});
}
models.sort((a, b) => b.totalPp - a.totalPp);
// colors are STICKY (2026-10-05): a model keeps the color the published
// models.json already gives it, so a new model never recolors the others
// even when it ranks above them. A new model takes the first palette color
// no kept model uses — single runs from the palette's start, composites
// from its end (kept from when composites were shown apart) — and the old
// rank-order rule only once the palette is exhausted.
{
	const compIds = new Set([...benches].filter(([, b]) => b.d.composite).map(([file]) => publicId.get(file)));
	const prior = priorColors;
	const used = new Set(models.map((m) => prior[m.id]).filter(Boolean));
	let run = 0,
		comp = 0;
	for (const m of models) {
		const isComp = compIds.has(m.id);
		const order = isComp ? [...PALETTE].reverse() : PALETTE;
		const free = prior[m.id] ?? order.find((c) => !used.has(c));
		const color = free ?? (isComp ? PALETTE[PALETTE.length - 1 - (comp % PALETTE.length)] : PALETTE[run % PALETTE.length]);
		if (isComp) comp++;
		else run++;
		used.add(color);
		/** @type {any} */ (m).color = color;
	}
}

fs.writeFileSync(path.join(outDir, 'levels.json'), JSON.stringify(levelCols));
fs.writeFileSync(
	path.join(outDir, 'models.json'),
	JSON.stringify({ generated: new Date().toISOString(), models })
);

console.log(
	`scanned ${scanned} bench files -> ${models.length} current-era models, ${nLevels} levels (${index.size} ScoreSaber, ${blIndex.size} BeatLeader, ${rows.filter((r) => r.ss && r.bl).length} on both)`
);
for (const [file, { d }] of benches)
	console.log(
		`  ${d.total_weighted_pp.toFixed(2).padStart(9)} pp  ${publicId.get(file)?.padEnd(12)} <- ${file}` +
			(isBoth(d) ? `  (BL ${d.bl_total_weighted_pp.toFixed(2)} pp)` : /** @type {any} */ (benches.get(file)).blTotals ? `  (BL ${/** @type {any} */ (benches.get(file)).blTotals.weighted.toFixed(2)} pp, derived, shared maps)` : '') +
			(d.composite ? '  [composite]' : '')
	);
for (const m of models) if (m.replays) console.log(`  ${m.id}: ${Object.keys(m.replays).length} replays linked`);
