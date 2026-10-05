import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Fully static: the dataset is JSON under static/data/, fetched by the
			// client. Map and model pages are client-routed; Cloudflare serves the
			// shell for them (wrangler.jsonc not_found_handling), and the
			// fallback page only satisfies adapter-static's strict check.
			adapter: adapter({ fallback: '200.html' }),
			// absolute asset URLs: the shell is served for nested paths too
			paths: { relative: false },
			// a deploy replaces every hashed chunk, and not_found_handling serves
			// the HTML shell for the old ones, so a tab opened before a deploy
			// cannot import its next route. Poll _app/version.json so the layout
			// (beforeNavigate) turns the next navigation into a full page load.
			version: { pollInterval: 60_000 }
		})
	]
});
