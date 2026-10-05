import { loadMap } from '$lib/map-load.js';

export const prerender = false;

/** @type {import('./$types').PageLoad} */
export const load = ({ fetch, params }) => loadMap(fetch, 'bl', params.id);
