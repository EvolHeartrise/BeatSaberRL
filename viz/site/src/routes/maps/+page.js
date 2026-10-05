import { loadAllResults, loadAllResultsBl, loadLevels, loadModels } from '$lib/data.js';

/** @type {import('./$types').PageLoad} */
export async function load({ fetch }) {
	const { models } = await loadModels(fetch);
	const [levels, results, resultsBl] = await Promise.all([loadLevels(fetch), loadAllResults(fetch, models), loadAllResultsBl(fetch, models)]);
	return { models, levels, results, resultsBl };
}
