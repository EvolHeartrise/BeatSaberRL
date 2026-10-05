// Loader of the map page. One unified page per level: /maps/<ScoreSaber id>
// for every level ScoreSaber ranks, /maps/bl/<BeatLeader id> for the
// BeatLeader-only ones. A BeatLeader id of a level ScoreSaber also ranks
// redirects to its /maps/<ScoreSaber id> page, so older /maps/bl links keep
// working.
import { error, redirect } from '@sveltejs/kit';
import { loadAllResults, loadAllResultsBl, loadLevels, loadModels } from '$lib/data.js';

/** @param {typeof fetch} fetch @param {'ss' | 'bl'} board the URL's id space @param {string} id */
export async function loadMap(fetch, board, id) {
	const { models } = await loadModels(fetch);
	const [levels, results, resultsBl] = await Promise.all([loadLevels(fetch), loadAllResults(fetch, models), loadAllResultsBl(fetch, models)]);
	let i;
	if (board === 'ss') {
		i = levels.ssId.indexOf(Number(id));
		if (i < 0) {
			// an all-digit BeatLeader id typed into the ScoreSaber path
			if (levels.blId.includes(id)) redirect(307, `/maps/bl/${id}`);
			error(404, `No ranked map with ScoreSaber leaderboard id ${id}`);
		}
	} else {
		i = levels.blId.indexOf(id);
		if (i < 0) error(404, `No ranked map with BeatLeader leaderboard id ${id}`);
		if (levels.ssId[i] != null) redirect(308, `/maps/${levels.ssId[i]}`);
	}
	return { i, models, levels, results, resultsBl };
}
