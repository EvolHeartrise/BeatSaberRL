// Assemble replays.flee.systems' deploy dir (dist/) — a Cloudflare Worker
// (wrangler.jsonc): the viewer as a static asset, the replays as objects in
// the R2 bucket beat-arena-replays served by src/worker.js (2026-10-03):
//
//   dist/index.html   the 3D replay viewer: viz/app's SITE build
//                     (viz/app/dist-site/index.html, `npm run build:site`),
//                     not viz/replay.html. It opens ?replay=/r/<file> and is
//                     watch-only: no Flux WebXR game, no Flux replays (they
//                     are refused with a message), no pose mode and no
//                     file-picker button — see viz/app/src/lib/build.js.
//   dist/_headers     caching
//
// The replays (r/<16 hex>.rpl.gz) are listed by the local replay store's
// manifest.json (NOT in git; written by scripts/publish_replays.py) and must
// all be uploaded (scripts/upload_r2.mjs, recorded in r2-ledger.json) before
// a deploy.
//
//   (cd ../app && npm run build:site)
//   node scripts/upload_r2.mjs [--src STORE] [--prune]
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
//        --bench <its published bench.json> --ckpt <its checkpoint> <rpl.gz...>
//   3. node scripts/upload_r2.mjs [--prune]                 (this dir)
//   4. (cd ../app && npm run build:site) && node scripts/build.mjs
//        && npx wrangler deploy                              (this dir)
//   5. viz/site: npm run data && npm run deploy   (Watch links from the manifest)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, '..');
const viewer = path.resolve(site, '../app/dist-site/index.html');
const dist = path.join(site, 'dist');
const NAME_RE = /^r\/[0-9a-f]{16}\.rpl\.gz$/;

let src = process.env.REPLAYS_STORE || path.join(os.homedir(), 'beat-arena-volume-backup/replays-site');
for (let i = 2; i < process.argv.length; i++) {
	if (process.argv[i] === '--src') src = path.resolve(process.argv[++i]);
}

const manifest = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
if (!fs.existsSync(viewer)) throw new Error(`${viewer} missing: run \`npm run build:site\` in viz/app first`);
const html = fs.readFileSync(viewer, 'utf8');
if (!html.includes('under /r/ on this site'))
	throw new Error(`${viewer} predates URL loading (?replay=): rebuild viz/app`);
if (!html.includes('like /r/0123456789abcdef.rpl.gz'))
	throw new Error(`${viewer} predates replay format 2 (.rpl.gz): rebuild viz/app (npm run build:site)`);
// a viewer older than any app source is a stale build: deploying it shipped a
// viewer that refused every .rpl.gz link (2026-10-02)
const newest = (dir) => {
	let t = 0;
	for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
		const p = path.join(dir, e.name);
		t = Math.max(t, e.isDirectory() ? newest(p) : fs.statSync(p).mtimeMs);
	}
	return t;
};
if (fs.statSync(viewer).mtimeMs < newest(path.resolve(site, '../app/src')))
	throw new Error(`${viewer} is older than viz/app/src: rebuild it (cd ../app && npm run build:site)`);
// the site ships the watch-only viewer: a FULL build (viz/replay.html, or
// dist/ from a plain `npm run build`) still has these
for (const [mark, what] of [['play Flux', 'the Flux WebXR game'], ['🧍 pose', 'pose mode'],
                            ['click to choose', 'the file-picker button']])
	if (html.includes(mark))
		throw new Error(`${viewer} still has ${what} ("${mark}"): use \`npm run build:site\`, not a full build`);
if (!html.includes("Flux replays aren't supported on this site"))
	throw new Error(`${viewer} does not refuse Flux replays: use \`npm run build:site\``);

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });
// the viewer names no icon, so every visit would log a /favicon.ico 404
const page = /rel=["']?icon/.test(html) ? html : html.replace('<head>', '<head><link rel="icon" href="data:,">');
fs.writeFileSync(path.join(dist, 'index.html'), page);

// the replays are R2 objects (src/worker.js, scripts/upload_r2.mjs), not
// assets: dist is the viewer alone. Every file the manifest lists must be in
// the bucket already (r2-ledger.json), so a deploy never links a replay the
// Worker cannot serve
const lpath = path.join(src, 'r2-ledger.json');
const ledger = fs.existsSync(lpath) ? JSON.parse(fs.readFileSync(lpath, 'utf8')) : {};
let n = 0,
	bytes = 0;
const notUp = [];
for (const [model, m] of Object.entries(manifest.models)) {
	for (const [lb, r] of Object.entries(m.replays ?? {})) {
		if (!NAME_RE.test(r.file)) throw new Error(`${model} ${lb}: bad file name ${r.file}`);
		if (!ledger[r.file]) notUp.push(r.file);
		n++;
		bytes += ledger[r.file]?.bytes ?? 0;
	}
}
if (notUp.length)
	throw new Error(`${notUp.length} replays are not in R2 yet (e.g. ${notUp[0]}): run node scripts/upload_r2.mjs first`);

// the viewer is revalidated so a redeploy reaches open-ended visitors at
// once (the replays' caching is set by src/worker.js)
fs.writeFileSync(
	path.join(dist, '_headers'),
	['/', '  Cache-Control: no-cache', '/index.html', '  Cache-Control: no-cache',
	 ''].join('\n')
);
console.log(`dist: viewer; ${n} replays (${(bytes / 1048576).toFixed(1)} MiB) listed by ${src}, all in R2`);
