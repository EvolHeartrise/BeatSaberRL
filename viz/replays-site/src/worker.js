// replays.flee.systems: the viewer is a static asset (dist/index.html);
// the replays live in the R2 bucket beat-arena-replays (2026-10-03, the
// every-level replay sets outgrew Static Assets' file cap and the laptop
// disk). This Worker runs only for /r/* (wrangler.jsonc run_worker_first)
// and answers from the bucket, so replay links keep their same-origin
// /r/<16 hex>.rpl.gz form and the viewer's path check is unchanged.
//
// Headers match what the static store sent: application/gzip with NO
// Content-Encoding (the viewer inflates the bytes itself), a day of
// caching (names are per weights, rate, level and body, so a file only
// changes when a write-back re-records the same play), nosniff. A missing
// replay is a plain-text 404, never the viewer page, so the viewer can
// say "this replay does not exist".

const NAME_RE = /^\/r\/[0-9a-f]{16}\.rpl\.gz$/;
const CACHE = 'public, max-age=86400';

export default {
	async fetch(request, env, ctx) {
		const url = new URL(request.url);
		if (request.method !== 'GET' && request.method !== 'HEAD')
			return new Response('method not allowed', { status: 405, headers: { Allow: 'GET, HEAD' } });
		if (!NAME_RE.test(url.pathname)) return notFound();

		// the edge cache first: R2 reads are free up to a quota, but a
		// popular replay should not cost one per view
		const cache = caches.default;
		const key = new Request(url.origin + url.pathname, { method: 'GET' });
		let res = await cache.match(key);
		if (!res) {
			const obj = await env.REPLAYS.get(url.pathname.slice(1));
			if (!obj) return notFound();
			res = new Response(obj.body, {
				headers: {
					'Content-Type': 'application/gzip',
					'Content-Length': String(obj.size),
					'Cache-Control': CACHE,
					'X-Content-Type-Options': 'nosniff',
					ETag: obj.httpEtag
				}
			});
			ctx.waitUntil(cache.put(key, res.clone()));
		}
		if (request.headers.get('If-None-Match') === res.headers.get('ETag'))
			return new Response(null, { status: 304, headers: { ETag: res.headers.get('ETag'), 'Cache-Control': CACHE } });
		return request.method === 'HEAD' ? new Response(null, { headers: res.headers }) : res;
	}
};

function notFound() {
	return new Response('not found', {
		status: 404,
		headers: { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' }
	});
}
