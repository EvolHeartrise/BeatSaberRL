// Upload the replay store's files to the R2 bucket replays.flee.systems
// serves them from (src/worker.js; 2026-10-03).
//
//   node scripts/upload_r2.mjs [--src STORE] [--jobs N] [--prune]
//        (default STORE: $REPLAYS_STORE, else ~/beat-arena-volume-backup/replays-site)
//
// Every file the store's manifest.json lists (r/<16 hex>.rpl.gz, written by
// scripts/publish_replays.py) is put to beat-arena-replays/<file> with
// `wrangler r2 object put --remote`, unless <store>/r2-ledger.json already
// records it with the same sha256. The ledger is the record of what the
// bucket holds: build.mjs refuses to deploy while a manifest file is not in
// it, and viz/site's build-data.mjs links a replay when its file is either
// in the store or in the ledger.
//
// --prune deletes each local copy once the ledger has it, so an
// every-level replay set never has to fit on the laptop: publish a chunk,
// upload with --prune, repeat. A pruned file that the manifest still lists
// is fine (the ledger has it); one that is in neither is an error.

import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const BUCKET = 'beat-arena-replays';
const NAME_RE = /^r\/[0-9a-f]{16}\.rpl\.gz$/;
const MAX_BYTES = 25 * 1024 * 1024; // kept from the Static Assets era: the viewer loads the whole file

let src = process.env.REPLAYS_STORE || path.join(os.homedir(), 'beat-arena-volume-backup/replays-site');
let jobs = 12,
	prune = false;
for (let i = 2; i < process.argv.length; i++) {
	const a = process.argv[i];
	if (a === '--src') src = path.resolve(process.argv[++i]);
	else if (a === '--jobs') jobs = Math.max(1, +process.argv[++i]);
	else if (a === '--prune') prune = true;
	else throw new Error(`unknown argument ${a}`);
}
const wrangler = process.env.WRANGLER ? [process.env.WRANGLER] : ['npx', '--yes', 'wrangler'];

const manifest = JSON.parse(fs.readFileSync(path.join(src, 'manifest.json'), 'utf8'));
const lpath = path.join(src, 'r2-ledger.json');
/** @type {Record<string, {bytes: number, sha256: string}>} */
const ledger = fs.existsSync(lpath) ? JSON.parse(fs.readFileSync(lpath, 'utf8')) : {};
const saveLedger = () => {
	const tmp = lpath + '.tmp';
	fs.writeFileSync(tmp, JSON.stringify(ledger, null, 1));
	fs.renameSync(tmp, lpath);
};

const files = new Set();
for (const [model, m] of Object.entries(manifest.models))
	for (const [k, r] of Object.entries(m.replays ?? {})) {
		if (!NAME_RE.test(r.file)) throw new Error(`${model} ${k}: bad file name ${r.file}`);
		files.add(r.file);
	}

const todo = [];
let have = 0,
	pruned = 0;
const missing = [];
for (const f of files) {
	const local = path.join(src, f);
	if (!fs.existsSync(local)) {
		if (ledger[f]) have++;
		else missing.push(f);
		continue;
	}
	const buf = fs.readFileSync(local);
	if (buf[0] !== 0x1f || buf[1] !== 0x8b) throw new Error(`${local}: not gzip`);
	if (buf.length >= MAX_BYTES) throw new Error(`${local}: ${buf.length} bytes, over 25 MiB`);
	const sha256 = crypto.createHash('sha256').update(buf).digest('hex');
	if (ledger[f]?.sha256 === sha256) {
		have++;
		if (prune) (fs.unlinkSync(local), pruned++);
	} else todo.push({ f, local, bytes: buf.length, sha256 });
}
if (missing.length)
	throw new Error(`${missing.length} manifest files are neither in the store nor uploaded, e.g. ${missing.slice(0, 3).join(', ')}`);
console.log(`${files.size} manifest files: ${have} already in R2, ${todo.length} to upload (${jobs} at a time)`);

/** @param {string[]} args */
const run = (args) =>
	new Promise((resolve) => {
		const p = spawn(wrangler[0], [...wrangler.slice(1), ...args], { stdio: ['ignore', 'ignore', 'pipe'] });
		let err = '';
		p.stderr.on('data', (d) => (err += d));
		p.on('close', (code) => resolve({ code, err }));
	});

let done = 0,
	failed = 0,
	upBytes = 0,
	sinceSave = 0;
const t0 = Date.now();
async function worker() {
	for (;;) {
		const job = todo.shift();
		if (!job) return;
		let r = { code: 1, err: '' };
		for (let attempt = 0; attempt < 3 && r.code !== 0; attempt++)
			r = await run(['r2', 'object', 'put', `${BUCKET}/${job.f}`, '--file', job.local,
			               '--content-type', 'application/gzip', '--remote']);
		if (r.code !== 0) {
			failed++;
			console.error(`FAILED ${job.f}: ${r.err.trim().split('\n').slice(-3).join(' | ')}`);
			continue;
		}
		ledger[job.f] = { bytes: job.bytes, sha256: job.sha256 };
		if (prune) (fs.unlinkSync(job.local), pruned++);
		done++;
		upBytes += job.bytes;
		if (++sinceSave >= 25) (saveLedger(), (sinceSave = 0));
		if (done % 100 === 0)
			console.log(`  ${done} uploaded (${(upBytes / 1048576).toFixed(0)} MiB, ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
	}
}
await Promise.all(Array.from({ length: jobs }, worker));
saveLedger();
console.log(`uploaded ${done} (${(upBytes / 1048576).toFixed(1)} MiB) in ${((Date.now() - t0) / 1000).toFixed(0)} s` +
            (prune ? `; pruned ${pruned} local copies` : '') + (failed ? `; ${failed} FAILED` : ''));
if (failed) process.exit(1);
