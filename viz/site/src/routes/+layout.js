import { loadModels } from '$lib/data.js';

// Everything renders client-side from the static JSON dataset; the two index
// pages prerender as shells, the parameterized pages are client-routed.
export const ssr = false;
export const prerender = true;
export const trailingSlash = 'never';

/** @type {import('./$types').LayoutLoad} */
export const load = ({ fetch }) => loadModels(fetch);
