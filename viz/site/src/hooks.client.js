// A tab opened before a deploy still references the previous build's hashed
// chunks. They are gone after the deploy, and not_found_handling
// (wrangler.jsonc) answers with the HTML shell, so the route import fails
// and SvelteKit renders its 500 page. When a navigation fails that way,
// reload the target URL once from the server (sessionStorage guards
// against a reload loop if the failure is real).
const STALE = /dynamically imported module|Importing a module script failed|MIME type|Unable to preload CSS/i;

/** @type {import('@sveltejs/kit').HandleClientError} */
export function handleError({ error, event }) {
	const msg = String(/** @type {any} */ (error)?.message ?? error);
	// a static public site: show the real error instead of "Internal Error"
	if (!STALE.test(msg)) return { message: msg };
	const target = event.url.href;
	try {
		if (sessionStorage.getItem('stale-chunk-reload') === target) return { message: msg };
		sessionStorage.setItem('stale-chunk-reload', target);
	} catch {
		// storage blocked: still reload; a real failure then shows the error page
	}
	location.href = target;
	return { message: msg };
}
