// Hidden graphs (2026-10-06, user request): a card's hide button takes a
// panel off its tab, the header's "Hidden (n)" menu brings it back.
//
// One set per tab, keyed by the stable panel ids of series.js — Live:
// PANELS[].id; Compare: the small-multiple metric key (COMPARE_KEYS) and
// "diff:<key>" for the difference panels (DIFF_KEYS). Live and Compare
// show some of the same metrics but for different jobs (one run in depth
// vs many runs side by side), so hiding a panel on one tab leaves the
// other alone.
//
// Persisted in localStorage (dash.hidden.<tab>); data refreshes never
// touch it, so an auto-refresh cannot un-hide anything. Ids no panel has
// any more are pruned on load; panels added later default to visible.
import { PANELS, COMPARE_KEYS, DIFF_KEYS, label } from "./series.js";

export const HIDE_TABS = ["live", "compare"];

// every panel id a tab can show, with its menu title
export const PANEL_TITLES = {
  live: new Map(PANELS.map((p) => [p.id, p.title])),
  compare: new Map([
    ...COMPARE_KEYS.map((k) => [k, label(k)]),
    ...DIFF_KEYS.map((k) => ["diff:" + k, "Δ " + label(k)]),
  ]),
};

const KEY = (tab) => `dash.hidden.${tab}`;

function load(tab) {
  let ids = [];
  try {
    const v = JSON.parse(localStorage.getItem(KEY(tab)) || "[]");
    if (Array.isArray(v)) ids = v.filter((x) => typeof x === "string");
  } catch {
    /* no storage, or junk in it: nothing hidden */
  }
  const known = PANEL_TITLES[tab];
  const kept = [...new Set(ids.filter((id) => known.has(id)))];
  if (kept.length !== ids.length) save(tab, kept); // prune ids that no longer exist
  return kept;
}

function save(tab, ids) {
  try {
    localStorage.setItem(KEY(tab), JSON.stringify(ids));
  } catch {
    /* private window / blocked storage: hidden for this page only */
  }
}

export const hidden = $state(Object.fromEntries(HIDE_TABS.map((t) => [t, load(t)])));

export const isHidden = (tab, id) => hidden[tab]?.includes(id) ?? false;

export function hidePanel(tab, id) {
  if (!hidden[tab] || hidden[tab].includes(id)) return;
  hidden[tab] = [...hidden[tab], id];
  save(tab, hidden[tab]);
}

export function showPanel(tab, id) {
  if (!hidden[tab]?.includes(id)) return;
  hidden[tab] = hidden[tab].filter((x) => x !== id);
  save(tab, hidden[tab]);
}

export function showAll(tab) {
  if (!hidden[tab]?.length) return;
  hidden[tab] = [];
  save(tab, []);
}
