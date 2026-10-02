// Assemble replays.flee.systems' deploy dir (dist/) — a Cloudflare Worker
// serving static assets only (wrangler.jsonc):
//
//   dist/index.html   the 3D replay viewer: viz/app's SITE build
//                     (viz/app/dist-site/index.html, `npm run build:site`),
//                     not viz/replay.html. It opens ?replay=/r/<file> and is
//                     watch-only: no Flux WebXR game, no Flux replays (they
//                     are refused with a message), no pose mode and no
//                     file-picker button — see viz/app/src/lib/build.js.
//   dist/r/*.jsonl.gz the published replays, pulled from the local replay
//                     store (NOT in git) that scripts/publish_replays.py
//                     writes — only the files its manifest lists
//   dist/_headers     caching
//
//   (cd ../app && npm run build:site)
//   node scripts/build.mjs [--src STORE]   (default: $REPLAYS_STORE, else
//                                           ~/beat-arena-volume-backup/replays-site)
//   npx wrangler deploy
//
// Adding a model's replays later:
//   1. record them (python/make_replays_batched.py — or make_top10_replays.py
//      one at a time — in the bench's regime; each must verify EXACT against
//      the bench, or be written back into it with --write-back, and carry
//      header.map);
//   2. python viz/replays-site/scripts/publish_replays.py --model <public id>
//        --bench <its published bench.json> --ckpt <its checkpoint> <jsonl...>
//   3. (cd ../app && npm run build:site) && node scripts/build.mjs
//        && npx wrangler deploy                              (this dir)
//   4. viz/site: npm run data && npm run deploy   (Watch links from the manifest)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, '..');
const viewer = path.resolve(site, '../app/dist-site/index.html');
const dist = path.join(site, 'dist');
const MAX_BYTES = 25 * 1024 * 1024; // Cloudflare static assets, per file
const NAME_RE = /^r\/[0-9a-f]{16}\.jsonl\.gz$/;

let src = process.env.REPLAYS_STORE || path.join(os.homedir(), 'beat-arena-volume-backup/replays-site');
for (let i = 2; i < process.argv.length; i++) {
	if (process.argv[i] === '--src') src = path.resolve(process.argv[++i]);
}

const manifest = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
if (!fs.existsSync(viewer)) throw new Error(`${viewer} missing: run \`npm run build:site\` in viz/app first`);
const html = fs.readFileSync(viewer, 'utf8');
if (!html.includes('under /r/ on this site'))
	throw new Error(`${viewer} predates URL loading (?replay=): rebuild viz/app`);
// the site ships the watch-only viewer: a FULL build (viz/replay.html, or
// dist/ from a plain `npm run build`) still has these
for (const [mark, what] of [['play Flux', 'the Flux WebXR game'], ['🧍 pose', 'pose mode'],
                            ['click to choose', 'the file-picker button']])
	if (html.includes(mark))
		throw new Error(`${viewer} still has ${what} ("${mark}"): use \`npm run build:site\`, not a full build`);
if (!html.includes("Flux replays aren't supported on this site"))
	throw new Error(`${viewer} does not refuse Flux replays: use \`npm run build:site\``);

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(path.join(dist, 'r'), { recursive: true });
// the viewer names no icon, so every visit would log a /favicon.ico 404
const page = /rel=["']?icon/.test(html) ? html : html.replace('<head>', '<head><link rel="icon" href="data:,">');
fs.writeFileSync(path.join(dist, 'index.html'), page);

let n = 0,
	bytes = 0;
for (const [model, m] of Object.entries(manifest.models)) {
	for (const [lb, r] of Object.entries(m.replays ?? {})) {
		if (!NAME_RE.test(r.file)) throw new Error(`${model} ${lb}: bad file name ${r.file}`);
		const from = path.join(src, r.file);
		const size = fs.statSync(from).size;
		const magic = Buffer.alloc(2);
		const fd = fs.openSync(from, 'r');
		fs.readSync(fd, magic, 0, 2, 0);
		fs.closeSync(fd);
		if (magic[0] !== 0x1f || magic[1] !== 0x8b) throw new Error(`${from}: not gzip`);
		if (size >= MAX_BYTES) throw new Error(`${from}: ${size} bytes, over 25 MiB`);
		// hard link, not a copy (2026-10-02): the store's files are written
		// once and replaced (never edited in place) by publish_replays.py, so
		// dist can share their inodes; a copy only across filesystems
		const to = path.join(dist, r.file);
		try {
			fs.linkSync(from, to);
		} catch (e) {
			if (e.code !== 'EXDEV' && e.code !== 'EPERM' && e.code !== 'ENOTSUP') throw e;
			fs.copyFileSync(from, to);
		}
		n++;
		bytes += size;
		console.log(`  ${r.file}  ${(size / 1048576).toFixed(2).padStart(5)} MiB  ${model}  ${r.song} [${r.difficulty}]`);
	}
}

// the viewer is revalidated so a redeploy reaches open-ended visitors at
// once; replay files are named by (weights, rate, level) and deterministic,
// so they can sit in caches for a day
fs.writeFileSync(
	path.join(dist, '_headers'),
	['/', '  Cache-Control: no-cache', '/index.html', '  Cache-Control: no-cache',
	 '/r/*', '  Cache-Control: public, max-age=86400', '  X-Content-Type-Options: nosniff', ''].join('\n')
);
console.log(`dist: viewer + ${n} replays (${(bytes / 1048576).toFixed(1)} MiB) from ${src}`);
