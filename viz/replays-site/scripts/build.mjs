// Assemble replays.flee.systems' deploy dir (dist/) — a Cloudflare Worker
// serving static assets only (wrangler.jsonc):
//
//   dist/index.html   the 3D replay viewer (viz/replay.html, the single-file
//                     build of viz/app; it opens ?replay=/r/<file>)
//   dist/r/*.jsonl.gz the published replays, pulled from the local replay
//                     store (NOT in git) that scripts/publish_replays.py
//                     writes — only the files its manifest lists
//   dist/_headers     caching
//
//   node scripts/build.mjs [--src STORE]   (default: $REPLAYS_STORE, else
//                                           ~/beat-arena-volume-backup/replays-site)
//   npx wrangler deploy
//
// Adding a model's replays later:
//   1. record them (python/make_top10_replays.py, in the bench's regime; each
//      must verify EXACT against the bench, and carry header.map);
//   2. python viz/replays-site/scripts/publish_replays.py --model <public id>
//        --bench <its published bench.json> --ckpt <its checkpoint> <jsonl...>
//   3. node scripts/build.mjs && npx wrangler deploy        (this dir)
//   4. viz/site: npm run data && npm run deploy   (Watch links from the manifest)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const site = path.resolve(here, '..');
const viewer = path.resolve(site, '../replay.html');
const dist = path.join(site, 'dist');
const MAX_BYTES = 25 * 1024 * 1024; // Cloudflare static assets, per file
const NAME_RE = /^r\/[0-9a-f]{16}\.jsonl\.gz$/;

let src = process.env.REPLAYS_STORE || path.join(os.homedir(), 'beat-arena-volume-backup/replays-site');
for (let i = 2; i < process.argv.length; i++) {
	if (process.argv[i] === '--src') src = path.resolve(process.argv[++i]);
}

const manifest = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
if (!fs.existsSync(viewer)) throw new Error(`${viewer} missing: build viz/app first`);
const html = fs.readFileSync(viewer, 'utf8');
if (!html.includes('under /r/ on this site'))
	throw new Error(`${viewer} predates URL loading (?replay=): rebuild viz/app`);

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
		const buf = fs.readFileSync(from);
		if (buf[0] !== 0x1f || buf[1] !== 0x8b) throw new Error(`${from}: not gzip`);
		if (buf.length >= MAX_BYTES) throw new Error(`${from}: ${buf.length} bytes, over 25 MiB`);
		fs.writeFileSync(path.join(dist, r.file), buf);
		n++;
		bytes += buf.length;
		console.log(`  ${r.file}  ${(buf.length / 1048576).toFixed(2).padStart(5)} MiB  ${model}  ${r.song} [${r.difficulty}]`);
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
