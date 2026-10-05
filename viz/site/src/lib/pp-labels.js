// Short column labels for leaderboard values, with their full wording.
// ScoreSaber and BeatLeader numbers are different scales: every label names
// its board.
const href = 'https://wiki.scoresaber.com/ranking-system';
export const WPP = { label: 'SS WPP', full: 'ScoreSaber Weighted Performance Points', href };
export const UPP = { label: 'SS UPP', full: 'ScoreSaber Unweighted Performance Points', href };
export const SSPP = { label: 'SS PP', full: 'ScoreSaber Performance Points', href };
export const SSSTARS = { label: 'SS ★', full: 'ScoreSaber Star Rating' };

export const BLWPP = { label: 'BL WPP', full: 'BeatLeader Weighted Performance Points (each map’s PP weighted 0.965^rank)' };
export const BLUPP = { label: 'BL UPP', full: 'BeatLeader Unweighted Performance Points' };
export const BLPP = { label: 'BL PP', full: 'BeatLeader Performance Points (pass + acc + tech)' };
export const BLSTARS = { label: 'BL ★', full: 'BeatLeader Star Rating (a different scale from ScoreSaber stars)' };
export const BLSPLIT = { label: 'Pass / Acc / Tech', full: 'BeatLeader PP split into its pass, accuracy and tech parts' };

// applied torque: Σ over ticks of Σ_j |τ_j| × tick length (N·m·s), the
// same at any control rate; benches since 2026-10-01 record it
export const TAU = { label: 'Σ|τ|', full: 'Applied torque: the sum over the level of every joint’s |torque| × time, in N·m·s (lower = less effort)' };
export const SSTAU = { label: 'SS Σ|τ|', full: 'Total applied torque over every ScoreSaber-ranked level played, in N·m·s' };
export const BLTAU = { label: 'BL Σ|τ|', full: 'Total applied torque over every BeatLeader-ranked level played, in N·m·s' };
