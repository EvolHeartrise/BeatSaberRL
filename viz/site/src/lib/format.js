export const DIFFS = ['Easy', 'Normal', 'Hard', 'Expert', 'Expert+'];
export const DIFF_KEYS = ['easy', 'normal', 'hard', 'expert', 'expertplus'];

const int = new Intl.NumberFormat('en-US');

/** @param {number | null | undefined} x */
export const pct = (x, d = 2) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`);
/** @param {number | null | undefined} x */
export const pp = (x) => (x == null ? '—' : x.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
/** Whole points, no decimals (SS UPP). @param {number | null | undefined} x */
export const ppWhole = (x) => (x == null ? '—' : int.format(Math.round(x)));
/** @param {number | null | undefined} x */
export const num = (x) => (x == null ? '—' : int.format(x));
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 });
/** Torque in N·m·s, compact (33.2K, 247.3M). @param {number | null | undefined} x */
export const nms = (x) => (x == null ? '—' : compact.format(x));
/** @param {number} x */
export const stars = (x) => x.toFixed(2);
/** @param {number} s */
export const duration = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
/** @param {string} iso */
export const date = (iso) =>
	new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });

/** @param {string} hash */
export const cover = (hash) => `https://cdn.beatsaver.com/${hash.toLowerCase()}.jpg`;
/** @param {number} id */
export const scoresaber = (id) => `https://scoresaber.com/leaderboard/${id}`;
/** @param {string} id */
export const beatleader = (id) => `https://beatleader.com/leaderboard/global/${id}`;
