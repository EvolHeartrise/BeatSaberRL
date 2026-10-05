import { error } from '@sveltejs/kit';
import { loadLevels, loadModels, loadResults, loadResultsBl } from '$lib/data.js';

export const prerender = false;

/** @type {import('./$types').PageLoad} */
export async function load({ fetch, params }) {
	const { models } = await loadModels(fetch);
	const model = models.find((m) => m.id === params.id);
	if (!model) error(404, `No model "${params.id}"`);
	const [levels, results, resultsBl] = await Promise.all([loadLevels(fetch), loadResults(fetch, model), loadResultsBl(fetch, model)]);
	return { model, models, levels, results, resultsBl };
}
