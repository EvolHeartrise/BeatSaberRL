//! ScoreSaber multiplied-score walk over an ordered event stream (the
//! bench's `combo_walk`). Judge-shared like geom.rs: compiled into the
//! env crate and the Flux WebXR game's wasm crate, so the game's
//! combo / multiplier / score is the bench's, byte for byte.

use crate::Real;

/// ScoreSaber multiplied-score combo walk (bench-only; the training gym
/// stays multiplier-free). Mirrors beat_arena_env/scoresaber.py's
/// ComboTracker exactly — pinned by a cross-implementation test:
/// promotion (1x/2x/4x/8x, bar 2/4/8 hits) applies to the hit that
/// fills the bar; a note's cut AND its later post points commit at its
/// cut-time multiplier; any break (bad cut / miss / bomb) zeroes the
/// combo, halves the multiplier, resets the bar.
pub(crate) struct ComboWalk {
    pub(crate) multiplier: u32,
    pub(crate) progress: u32,
    pub(crate) combo: u32,
    pub(crate) max_combo: u32,
    pub(crate) breaks: u32,
    pub(crate) score: Real,
    pub(crate) note_mult: Vec<u32>, // 0 = not cut yet
}

impl ComboWalk {
    pub(crate) fn new(n_objects: usize) -> Self {
        ComboWalk {
            multiplier: 1,
            progress: 0,
            combo: 0,
            max_combo: 0,
            breaks: 0,
            score: 0.0,
            note_mult: vec![0; n_objects],
        }
    }

    pub(crate) fn progress_needed(m: u32) -> Option<u32> {
        match m {
            1 => Some(2),
            2 => Some(4),
            4 => Some(8),
            _ => None, // 8x cap
        }
    }

    pub(crate) fn consume(&mut self, note: u32, code: u8, pts: f32) {
        let i = note as usize;
        match code {
            0 => {
                // good cut: promote-before-commit, pin the multiplier
                if let Some(need) = Self::progress_needed(self.multiplier) {
                    self.progress += 1;
                    if self.progress >= need {
                        self.multiplier *= 2;
                        self.progress = 0;
                    }
                }
                self.note_mult[i] = self.multiplier;
                self.combo += 1;
                self.max_combo = self.max_combo.max(self.combo);
                self.score += pts as Real * self.multiplier as Real;
            }
            5 => {
                // follow-through: the note's cut-time multiplier
                let m = if self.note_mult[i] == 0 {
                    self.multiplier
                } else {
                    self.note_mult[i]
                };
                self.score += pts as Real * m as Real;
            }
            7 => {}
            // 7 = head-in-wall drain tick: NO combo effect (spec §12b —
            // only the ENTRY breaks; without this arm the catch-all
            // below would have broken combo every tick inside a wall)
            _ => {
                // slow / angle / wrong_hand / miss / bomb / wall ENTRY
                // (8): combo break
                self.combo = 0;
                self.breaks += 1;
                self.multiplier = (self.multiplier / 2).max(1);
                self.progress = 0;
            }
        }
    }
}
