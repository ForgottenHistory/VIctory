//! Title memory that flights declare their shaders may write, but that the
//! card is seen never to change (`VICTORY_LEARN_UNWRITTEN=1`; off by
//! default).
//!
//! The world's compute passes bind whole buffers for unordered access, so a
//! flight declares all of the title's 174 MB heap as written, and every read
//! of it while preparing a submission waits for the flights in the air to
//! land: 1.5 settles a frame for a 480-byte fill of upload memory, 24 ms a
//! frame (Tracy capture 2026-11-26, with argument batches). Skipping that
//! wait altogether (`VICTORY_IGNORE_FILL_READS=1`) took 7.4 fps to 8.0 with
//! no glitch seen, but nothing said the bytes it read were not about to be
//! written.
//!
//! Here the compare on the card is the witness: for copies in the card's
//! memory every landing tells, per 64 KiB piece of title memory a flight
//! declared, whether its shaders changed it. A piece compared clean
//! [`AFTER`] times and never seen changed is taken as one shaders do not
//! write, and a fill or a buffer copy that reads only such pieces does not
//! wait for the flights. A piece is seen changed when the card's compare
//! flags it, when a range is declared but not compared (all of it), and
//! whenever landed bytes are put into title memory there by any path
//! ([`written`]). Seen changed once, a piece is never taken again.
//!
//! What is counted ([`report`]): reads that went ahead, and misses: pieces
//! taken as unwritten that a later landing found written. A miss means a
//! read that went ahead around then may have taken bytes from before the
//! shaders' writes; none means none did.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The unit learned: the card's compare piece, 64 KiB of title memory.
const PIECE: usize = 64 << 10;
/// Clean compares before a piece is taken as unwritten.
const AFTER: u32 = 64;

/// Whether the reads of preparing may go ahead of flights where the card
/// is seen never to write (`VICTORY_LEARN_UNWRITTEN=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_LEARN_UNWRITTEN").is_some_and(|v| v == "1");
        if on {
            tracing::warn!("reads of preparing go ahead of flights where the card is seen never to write (VICTORY_LEARN_UNWRITTEN=1)");
        }
        on
    })
}

#[derive(Default)]
struct Learned {
    /// Title piece → clean compares, for pieces never seen changed.
    clean: HashMap<usize, u32>,
    /// Pieces seen changed.
    changed: HashSet<usize>,
}

static LEARNED: Mutex<Option<Learned>> = Mutex::new(None);

/// Since the last [`report`]: reads that went ahead, misses, pieces seen
/// changed for the first time, clean compares.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Since the start: misses.
static MISSES: AtomicU64 = AtomicU64::new(0);

fn with<T>(f: impl FnOnce(&mut Learned) -> T) -> T {
    let mut l = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
    f(l.get_or_insert_with(Learned::default))
}

fn mark_changed(l: &mut Learned, piece: usize) {
    if l.changed.insert(piece) {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        if l.clean.remove(&piece).is_some_and(|n| n >= AFTER) {
            STATS[1].fetch_add(1, Ordering::Relaxed);
            let all = MISSES.fetch_add(1, Ordering::Relaxed) + 1;
            if all <= 16 || all.is_power_of_two() {
                tracing::warn!(
                    piece = format_args!("{:#x}", piece * PIECE),
                    misses = all,
                    "a piece taken as unwritten by shaders was written by them (VICTORY_LEARN_UNWRITTEN)"
                );
            }
        }
    }
}

/// A landing's compare on the card: `pieces` (title piece, whether it was
/// changed) of the ranges it compared, and `uncompared` title ranges
/// declared but not compared (taken as changed).
pub(crate) fn landed(pieces: &[(usize, bool)], uncompared: &[(usize, usize)]) {
    if !on() {
        return;
    }
    with(|l| {
        for &(piece, changed) in pieces {
            if changed {
                mark_changed(l, piece);
            } else if !l.changed.contains(&piece) {
                let n = l.clean.entry(piece).or_insert(0);
                *n = n.saturating_add(1);
                STATS[3].fetch_add(1, Ordering::Relaxed);
            }
        }
        for &(from, to) in uncompared {
            for piece in from / PIECE..to.div_ceil(PIECE) {
                mark_changed(l, piece);
            }
        }
    });
}

/// Landed bytes were put into title memory at `[address, address + len)`.
pub(crate) fn written(address: usize, len: usize) {
    if !on() || len == 0 {
        return;
    }
    with(|l| {
        for piece in address / PIECE..address.saturating_add(len).div_ceil(PIECE) {
            mark_changed(l, piece);
        }
    });
}

/// Whether every piece of `[from, to)` is taken as one shaders do not write.
pub(crate) fn unwritten(from: usize, to: usize) -> bool {
    if !on() || from >= to {
        return false;
    }
    with(|l| {
        (from / PIECE..to.div_ceil(PIECE))
            .all(|piece| !l.changed.contains(&piece) && l.clean.get(&piece).is_some_and(|&n| n >= AFTER))
    })
}

/// A read went ahead of the flights because of [`unwritten`].
pub(crate) fn went_ahead() {
    STATS[0].fetch_add(1, Ordering::Relaxed);
}

/// For the frame-rate line: per frame since the last one.
pub(crate) fn report(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let (taken, changed) = with(|l| (l.clean.values().filter(|&&n| n >= AFTER).count(), l.changed.len()));
    tracing::info!(
        "unwritten by shaders (VICTORY_LEARN_UNWRITTEN): per frame {:.1} reads went ahead of flights, {:.2} misses \
         ({} since the start), {:.0} pieces compared clean, {:.1} first seen changed; {} pieces ({} MB) taken as \
         unwritten, {} seen changed",
        s[0],
        s[1],
        MISSES.load(Ordering::Relaxed),
        s[3],
        s[2],
        taken,
        taken * PIECE >> 20,
        changed
    );
}
