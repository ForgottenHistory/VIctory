//! Steps of a submission that read title memory flights in the air may
//! still write, done on the card in the flights' order instead of on the
//! CPU after a wait for them (`VICTORY_CARD_STEPS`, see [`stage`]). After
//! avatarmanz/VIctory-7900XTX cc03ecb, fitted to this tree.
//!
//! Why: the world's compute passes bind the title's 174 MB heap whole for
//! unordered access, so every flight declares all of it written, and the
//! title's ring of constants lies in the same heap. A constant buffer of a
//! descriptor table (480 bytes) and a title buffer copy read from it made
//! the submission wait for the flights in the air to land: 1-1.5 times a
//! frame, 20-24 ms a frame of the submitting thread (Tracy capture
//! 2026-11-26, with argument batches). Skipping the wait
//! (`VICTORY_IGNORE_FILL_READS=1`) gave 7.4 -> 8.0 fps but may read bytes
//! from before the shaders' writes; learning where shaders do not write
//! (`VICTORY_LEARN_UNWRITTEN`) was wrong 689 times while the world loaded.
//!
//! What is done: a list of ours goes to the card in front of the
//! submission's lists, in the same call. It copies the bytes read from the
//! host copy that the flights write into a buffer of ours, and from there
//! to the destination's host copy (a copy), or the fill's descriptor slot
//! is pointed at that buffer (a fill; the slot is the fill's alone: a table
//! made afresh for the draw, whose list has not gone to the card).
//!
//! Why that is the value the CPU would have read after the wait. After the
//! wait title memory holds, for every byte read, what the title wrote or
//! what the last of the flights that wrote it wrote (the landings put it
//! there, in order). The host copy holds the same when our list runs: every
//! flight that may write one of these bytes writes this same host copy
//! ([`holder`]; a texture read-back or a value written behind a flight
//! there, or a write through another host copy, and the old wait is taken),
//! they are all ahead of our list on the queue, and the title's own bytes
//! are brought up to date by a refresh of our list, prepared like any other
//! list's (also under flights' declared writes: `stale_under_flights`).
//! Nothing of this submission writes these bytes earlier than our list: the
//! steps are done in front of all its lists, and [`plan`] refuses when
//! another step of the submission writes what is read. Where a page is not
//! mapped the host copy may hold other bytes than the CPU's copy would have
//! left: refused.
//!
//! For a copy's destination our list declares a shader write, exactly the
//! bytes copied, after a refresh of them: from there on it is what any
//! shader's write is. The relay compares, and puts what changed into title
//! memory before the title is told anything of this flight (its
//! bottom-of-pipe writes and fences come after, in order); CPU steps of
//! later submissions that read it find it declared and wait, or are done
//! here. Refused: another step of the submission reads or writes the
//! destination on the CPU, or a refresh brings another host copy of it up
//! to date ([`plan`], [`elsewhere`]). While such a copy is in the air a
//! submission that refreshes another host copy of its destination, or reads
//! it in a step of ours, waits for it ([`copied_read`]); with the copy done
//! on the CPU it was there at once.
//!
//! What stage 2 does not see: code of ours that reads title memory away
//! from a submission's steps (the pass over textures picked by index, while
//! a list is recorded). With the CPU's copy such a read of the destination
//! found the bytes at once; now it finds them when the flight has landed,
//! and the landing's write into title memory is seen by the write watch and
//! the guards, so a texture made from older bytes is made again.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use super::*;
use crate::shadow::Shadow;

/// The stage (`VICTORY_CARD_STEPS`; off by default): `1` constant buffers
/// of descriptor tables (fills with a slot of their own), `2` the title's
/// buffer copies too. Only with the relay's exact waits (`relay_exact`).
pub(super) fn stage() -> u32 {
    static STAGE: OnceLock<u32> = OnceLock::new();
    *STAGE.get_or_init(|| match std::env::var("VICTORY_CARD_STEPS").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

pub(super) fn on() -> bool {
    stage() > 0
}

/// The bytes of our buffer between reading and writing: what one submission
/// may do this way.
const BETWEEN: u64 = 1 << 20;

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to, and whether it is the destination of a copy done here.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
    copy: bool,
}

impl Writer {
    fn again(&self) -> Writer {
        Writer { shadow: self.shadow.clone(), ..*self }
    }
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer { shadow: Some(shadow.clone()), from, to, copy: false }));
                }
                out.extend(flight.texture_writes.iter().map(|&(from, to)| Writer { shadow: None, from, to, copy: false }));
                out.extend(flight.own.iter().flat_map(Taken::copied));
            }
            Late::Write(at, _, bytes) => out.push(Writer { shadow: None, from: *at, to: *at + *bytes, copy: false }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// The destinations of copies done here that have not landed.
pub(super) fn copies(line: &Line) -> Vec<Writer> {
    if stage() < 2 {
        return Vec::new();
    }
    let flying = line.line.iter().filter_map(|l| match l {
        Late::Run(flight) => flight.own.as_ref(),
        _ => None,
    });
    let landing = line.landing.iter().filter(|w| w.copy).map(Writer::again);
    flying.flat_map(Taken::copied).chain(landing).collect()
}

/// Whether `runs` refresh another host copy of a destination in `copies`,
/// or read it in a step of ours (`Step::Host`): that reads title memory,
/// where the copy is not yet.
pub(super) fn copied_read(copies: &[Writer], runs: &[&HostList]) -> bool {
    if copies.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            Step::Refresh(shadow, at, len) => copies
                .iter()
                .any(|w| w.from < *at + *len && *at < w.to && w.shadow.as_ref().is_none_or(|s| !Arc::ptr_eq(s, shadow))),
            Step::Host(_, (from, to)) => copies.iter().any(|w| w.from < *to && *from < w.to),
            _ => false,
        })
    })
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    Fill { src: usize, len: usize, asked: bool, slot: Option<(usize, u32)>, dst: usize },
    /// Another read on the CPU (a texture, rectangles' vertices).
    Read { from: usize, to: usize, asked: bool },
    /// A refresh of a host copy (named by its address).
    Refresh { shadow: usize, from: usize, to: usize },
    /// Work of ours that may write title memory anywhere.
    Run,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The bytes `d` reads and writes as a copy or a fill (an empty range for
/// what it does not write).
fn reads_writes(d: &Does) -> Option<((usize, usize), (usize, usize))> {
    match *d {
        Does::Copy { dst, src, len, .. } => Some(((src, src + len), (dst, dst + len))),
        Does::Fill { src, len, .. } => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

fn touched(ranges: &[(usize, usize)], r: (usize, usize)) -> bool {
    touches(ranges, r.0, r.1)
}

/// The most steps one submission's list of ours does.
const MOST: usize = 4096;

/// The steps of `does` (by place) to be done on the card, in order, each
/// with its level; or why not. All that are asked for, and every copy and
/// fill that is in the way of one that is done there: it writes what that
/// one reads or writes, or reads what it writes, whichever comes first. So
/// what is left on the CPU writes none of their bytes, and reads of them
/// only what a taken copy has not been told to write yet: bytes a step
/// ahead of it reads, in title memory as it is. (After avatarmanz 007fe55.)
///
/// Steps of a level are done together, every read before every write; a
/// step that writes what an earlier one reads or writes, or reads what it
/// writes, is of a higher level. Our list does them level by level.
fn plan(does: &[Does], stage: u32) -> Result<Vec<(usize, u32)>, &'static str> {
    let takes = |d: &Does| match *d {
        Does::Copy { .. } if stage < 2 => Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
        Does::Fill { slot: None, .. } => Err("a fill without a descriptor slot of its own"),
        _ => Ok(()),
    };
    let mut taken = vec![false; does.len()];
    let (mut read, mut written) = (Vec::new(), Vec::new());
    let mut more = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Copy { asked: true, .. } | Does::Fill { asked: true, .. } => more.push(i),
            _ => {}
        }
    }
    while !more.is_empty() {
        for i in std::mem::take(&mut more) {
            takes(&does[i])?;
            taken[i] = true;
            if let Some((r, w)) = reads_writes(&does[i]) {
                if w.0 < w.1 && meet(r, w) {
                    return Err("a copy onto itself");
                }
                join_ranges(&mut read, [r]);
                join_ranges(&mut written, [w]);
            }
        }
        if taken.iter().filter(|t| **t).count() > MOST {
            return Err("too many steps depend on it");
        }
        for (i, d) in does.iter().enumerate() {
            // A fill that cannot be taken is looked at below.
            if taken[i] || matches!(d, Does::Fill { slot: None, .. }) {
                continue;
            }
            if let Some((r, w)) = reads_writes(d)
                && (touched(&written, r) || touched(&written, w) || touched(&read, w))
            {
                more.push(i);
            }
        }
    }
    let order: Vec<usize> = (0..does.len()).filter(|&i| taken[i]).collect();
    // What stays on the CPU and reads what an earlier taken copy writes
    // would read it too early.
    let mut behind: Vec<(usize, usize)> = Vec::new();
    for (i, d) in does.iter().enumerate() {
        if taken[i] {
            if let Some((_, w)) = reads_writes(d) {
                join_ranges(&mut behind, [w]);
            }
            continue;
        }
        match *d {
            Does::Read { from, to, .. } if touched(&behind, (from, to)) => {
                return Err("a texture or expanded vertices read what it writes");
            }
            Does::Fill { src, len, .. } if touched(&behind, (src, src + len)) => {
                return Err("a fill without a descriptor slot of its own reads what it writes");
            }
            _ => {}
        }
    }
    let mut out: Vec<(usize, u32)> = Vec::with_capacity(order.len());
    // The copies among them, which alone put others behind them or have to
    // stand behind others.
    let mut copies: Vec<usize> = Vec::new();
    for &i in &order {
        let Some((r, w)) = reads_writes(&does[i]) else {
            continue;
        };
        let writes = w.0 < w.1;
        let mut level = 0;
        // Behind what writes what this reads or writes, and, if this
        // writes, behind what reads it.
        let ahead: Box<dyn Iterator<Item = usize>> =
            if writes { Box::new(0..out.len()) } else { Box::new(copies.iter().copied()) };
        for k in ahead {
            let (j, at): (usize, u32) = out[k];
            let Some((jr, jw)) = reads_writes(&does[j]) else {
                continue;
            };
            if at >= level && (meet(jw, r) || meet(jw, w) || meet(jr, w)) {
                level = at + 1;
            }
        }
        if writes {
            copies.push(out.len());
        }
        out.push((i, level));
    }
    Ok(out)
}

/// Whether a refresh of `does` brings a host copy other than `shadow` up to
/// date in `[from, to)`.
fn elsewhere(does: &[Does], shadow: usize, from: usize, to: usize) -> bool {
    does.iter().any(|d| match *d {
        Does::Refresh { shadow: s, from: a, to: b } => s != shadow && meet((a, b), (from, to)),
        _ => false,
    })
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through, or without any the
/// one shaders would be given.
fn holder(writers: &[Writer], from: usize, to: usize) -> Result<Arc<Shadow>, &'static str> {
    let mut found: Option<&Arc<Shadow>> = None;
    for w in writers.iter().filter(|w| meet((w.from, w.to), (from, to))) {
        match (&w.shadow, found) {
            (None, _) => return Err("a read-back or a value behind a flight is written there"),
            (Some(s), Some(f)) if !Arc::ptr_eq(s, f) => return Err("flights write it through two host copies"),
            (Some(s), _) => found = Some(s),
        }
    }
    let shadow = match found {
        Some(s) => s.clone(),
        None => crate::shadow::find(from as u64, (to - from) as u64).ok_or("no host copy of it")?,
    };
    let (start, len) = shadow.span();
    if shadow.in_place() || (from as u64) < start || to as u64 > start + len {
        return Err("no host copy holds all of it");
    }
    Ok(shadow)
}

/// Whether every page of `[from, to)` can be read.
fn mapped(from: usize, to: usize) -> bool {
    let mut bytes = vec![0u8; to - from];
    read_into(from as *const u8, &mut bytes)
}

/// A step done on the card.
struct Op {
    from: (Arc<Shadow>, usize),
    len: usize,
    /// Its place in the buffer between.
    at: u64,
    /// Its level ([`plan`]).
    level: u32,
    to: To,
}

enum To {
    /// A host copy, and the title address; the step, as the CPU has it.
    Copy(Arc<Shadow>, usize),
    /// A descriptor slot (title heap, slot), and the fill's destination.
    Slot(usize, u32, usize),
}

/// The list in front of a submission's, and its buffer ([`BETWEEN`] bytes).
struct Own {
    list: HostList,
    between: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded, and the list is touched by one
// thread at a time (the submitting one until it is sent, then nobody until
// its flight is done with and it is taken out of `POOL`).
unsafe impl Send for Own {}

/// Those whose flight is done with: closed, their steps dropped.
static POOL: Mutex<Vec<Own>> = Mutex::new(Vec::new());

/// Since the last frame-rate line: fills and copies done on the card, their
/// bytes, submissions that waited instead, refused by a check.
static DONE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Why the last refusals were, since the last frame-rate line.
static REFUSED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

fn refused(why: &'static str) {
    DONE[3].fetch_add(1, Ordering::Relaxed);
    let mut all = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    match all.iter_mut().find(|(w, _)| *w == why) {
        Some(e) => e.1 += 1,
        None => all.push((why, 1)),
    }
}

/// The steps of a submission done on the card, that the CPU leaves out:
/// fills by their destination in upload memory, copies as they stand
/// (destination, source, bytes).
#[derive(Default)]
pub(super) struct Skips {
    fills: std::collections::HashSet<usize>,
    copies: std::collections::HashSet<(usize, usize, usize)>,
}

impl Skips {
    pub(super) fn fill(&self, dst: usize) -> bool {
        self.fills.contains(&dst)
    }

    pub(super) fn copy(&self, step: (usize, usize, usize)) -> bool {
        self.copies.contains(&step)
    }
}

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, the steps the CPU leaves out, and the
/// copies' destinations. Its list goes back to the pool when it is dropped:
/// with the flight it went with, once that has landed.
pub(super) struct Taken {
    own: Option<Own>,
    skips: Skips,
    copied: Vec<(Arc<Shadow>, usize, usize)>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.as_ref().expect("a taken list").list
    }

    pub(super) fn skips(&self) -> &Skips {
        &self.skips
    }

    fn copied(&self) -> impl Iterator<Item = Writer> {
        self.copied.iter().map(|(shadow, from, to)| Writer { shadow: Some(shadow.clone()), from: *from, to: *to, copy: true })
    }
}

impl Drop for Taken {
    fn drop(&mut self) {
        if let Some(own) = self.own.take() {
            // The host copies its steps held go; its commands stay until the
            // next use resets it.
            drop(std::mem::take(&mut *own.list.deferred()));
            POOL.lock().unwrap_or_else(|e| e.into_inner()).push(own);
        }
    }
}

impl Host {
    /// With flights in the air: if what keeps `runs` from being prepared
    /// now is only reads of title memory that can all be done on the card
    /// ([`stage`]), the list that does them. `None` if nothing is in the
    /// way or the old wait is due. Under the submission lock.
    pub(super) fn card_steps(&self, runs: &[&HostList]) -> Option<Taken> {
        if !on() || !relay_exact() {
            return None;
        }
        let (written, copies) = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            (line.written.clone(), copies(&line))
        };
        blocked_by(&written, runs, false, None)?;
        if copied_read(&copies, runs) {
            refused("a copy done on the card is read through another host copy");
            return None;
        }
        match self.take_steps(&written, runs) {
            Ok(Some(taken)) => {
                // Nothing else of the submission waits for the flights.
                if blocked_by(&written, runs, false, Some(taken.skips())).is_some() {
                    refused("another step of the submission waits too");
                    return None;
                }
                Some(taken)
            }
            Ok(None) => None,
            Err(why) => {
                refused(why);
                None
            }
        }
    }

    fn take_steps(&self, written: &PendingWrites, runs: &[&HostList]) -> Result<Option<Taken>, &'static str> {
        let mut does = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                match step {
                    &Step::Copy(dst, src, len) => does.push(Does::Copy {
                        dst,
                        src,
                        len,
                        asked: written.touches(src, src + len) || written.touches(dst, dst + len),
                    }),
                    Step::Fill(_) | Step::Fills(_) => {
                        for f in step.fills() {
                            does.push(Does::Fill {
                                src: f.src,
                                len: f.len,
                                asked: written.touches(f.src, f.src.saturating_add(f.len)),
                                slot: f.slot,
                                dst: f.dst,
                            });
                        }
                    }
                    Step::Texture(staging) => {
                        let (at, len) = staging.memory();
                        does.push(Does::Read { from: at, to: at + len, asked: written.touches(at, at + len) });
                    }
                    &Step::Host(_, (from, to)) if from < to => does.push(Does::Read { from, to, asked: false }),
                    Step::Refresh(shadow, at, len) => {
                        does.push(Does::Refresh { shadow: Arc::as_ptr(shadow) as usize, from: *at, to: *at + *len });
                    }
                    Step::Run(_) => does.push(Does::Run),
                    _ => {}
                }
            }
        }
        let chosen = plan(&does, stage())?;
        if chosen.is_empty() {
            return Ok(None);
        }
        let mut writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(Writer::again));
            all
        };
        let mut ops: Vec<Op> = Vec::with_capacity(chosen.len());
        // The places read (host copy, address, bytes, level): where in the
        // buffer between their bytes are.
        let mut places = std::collections::HashMap::new();
        let mut end = 0u64;
        for &(i, level) in &chosen {
            let Some(((src, _), (dst, _))) = reads_writes(&does[i]) else {
                continue;
            };
            let (Does::Copy { len, .. } | Does::Fill { len, .. }) = does[i] else {
                continue;
            };
            if len == 0 {
                return Err("nothing to read");
            }
            let from = holder(&writers, src, src + len)?;
            let place = (Arc::as_ptr(&from) as usize, src, len, level);
            let at = match places.get(&place) {
                Some(&at) => at,
                None => {
                    if end + len as u64 > BETWEEN {
                        return Err("too much at once");
                    }
                    if !mapped(src, src + len) {
                        return Err("not all of what it reads is mapped");
                    }
                    let at = end;
                    end = (end + len as u64).next_multiple_of(256);
                    places.insert(place, at);
                    at
                }
            };
            let to = match does[i] {
                Does::Copy { .. } => {
                    let to = holder(&writers, dst, dst + len)?;
                    if elsewhere(&does, Arc::as_ptr(&to) as usize, dst, dst + len) {
                        return Err("another host copy of what it writes is read");
                    }
                    if !mapped(dst, dst + len) {
                        return Err("not all of what it writes is mapped");
                    }
                    // What the steps behind it read of it is in that host
                    // copy, or nowhere.
                    writers.push(Writer { shadow: Some(to.clone()), from: dst, to: dst + len, copy: true });
                    To::Copy(to, dst)
                }
                Does::Fill { slot: Some((heap, index)), dst, .. } => To::Slot(heap, index, dst),
                _ => continue,
            };
            ops.push(Op { from: (from, src), len, at, level, to });
        }
        let Some(own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        let mut copied = Vec::new();
        let mut skips = Skips::default();
        // First every refresh, then the writes: the bytes compared with are
        // the title's.
        for op in &ops {
            op.from.0.refresh(&own.list, op.from.1 as u64, op.len as u64);
            if let To::Copy(to, dst) = &op.to {
                to.refresh(&own.list, *dst as u64, op.len as u64);
            }
        }
        for op in &ops {
            if let To::Copy(to, dst) = &op.to {
                to.gpu_writes(&own.list, *dst as u64, op.len as u64, "card steps");
                copied.push((to.clone(), *dst, *dst + op.len));
                skips.copies.insert((*dst, op.from.1, op.len));
            }
        }
        #[cfg(feature = "diagnostics")]
        own.list.note_op(|| format!("{} steps of title memory done on the card, {end} bytes (VICTORY_CARD_STEPS)", ops.len()));
        let levels = ops.iter().map(|op| op.level).max().unwrap_or(0);
        for level in 0..=levels {
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for op in ops.iter().filter(|op| op.level == level) {
                let len = op.len as u64;
                let read = (&op.from.0.buffer, op.from.0.offset(op.from.1 as u64), op.at, len);
                // (Each place once.)
                if places.remove(&(Arc::as_ptr(&op.from.0) as usize, op.from.1, op.len, level)).is_some() {
                    reads.push(read);
                }
                if let To::Copy(to, dst) = &op.to {
                    writes.push((&to.buffer, to.offset(*dst as u64), op.at, len));
                }
            }
            record(level == 0, &own.list.list, &own.between, &reads, &writes);
        }
        DONE[4].fetch_add(u64::from(levels) + 1, Ordering::Relaxed);
        own.list.close();
        // SAFETY: a live buffer.
        let gpu = unsafe { own.between.GetGPUVirtualAddress() };
        for op in &ops {
            DONE[2].fetch_add(op.len as u64, Ordering::Relaxed);
            match &op.to {
                To::Slot(heap, index, dst) => {
                    DONE[0].fetch_add(1, Ordering::Relaxed);
                    // (A heap that is gone: nothing can read the slot.)
                    crate::descriptor::constant_slot(*heap, *index, gpu + op.at, op.len as u32);
                    skips.fills.insert(*dst);
                }
                To::Copy(..) => {
                    DONE[1].fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        Ok(Some(Taken { own: Some(own), skips, copied }))
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(own) = pooled {
            own.list.reset();
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "steps of title memory done on the card".into());
        let list = self.new_list().ok()?;
        Some(Own { list, between })
    }
}

/// Records a level of what [`stage`] describes into `list` (open, the
/// first of its call): `reads` are copied into `between` (source, offset in
/// it, offset in `between`, bytes), then `writes` out of it (destination,
/// offset in it, offset in `between`, bytes). `between` is left readable as
/// constants (and must be so, unless this is the `first` level), the
/// others in the common state, as the submission's lists expect them;
/// buffers decay to the common state when the call has run.
fn record(
    first: bool,
    list: &ID3D12GraphicsCommandList,
    between: &ID3D12Resource,
    reads: &[(&ID3D12Resource, u64, u64, u64)],
    writes: &[(&ID3D12Resource, u64, u64, u64)],
) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    let mut dests: Vec<&ID3D12Resource> = Vec::new();
    for (all, list) in [(reads, &mut sources), (writes, &mut dests)] {
        for &(buffer, ..) in all {
            if !list.iter().any(|b| b.as_raw() == buffer.as_raw()) {
                list.push(buffer);
            }
        }
    }
    let both = |b: &ID3D12Resource| dests.iter().any(|d| d.as_raw() == b.as_raw());
    let constants = source | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    let mut barriers = vec![transition(between, dest, constants)];
    barriers.extend(sources.iter().map(|b| transition(b, source, if both(b) { dest } else { common })));
    let done: Vec<D3D12_RESOURCE_BARRIER> = dests.iter().map(|b| transition(b, dest, common)).collect();
    // SAFETY: an open list; live buffers, and the callers' ranges lie in
    // them. The buffers start in the common state (the list is the first of
    // its call, and every level leaves them so) and are promoted by their
    // first use.
    unsafe {
        if first {
            // What earlier lists' shaders wrote, first.
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
        } else {
            list.ResourceBarrier(&[transition(between, constants, dest)]);
        }
        for &(buffer, offset, at, len) in reads {
            list.CopyBufferRegion(between, at, buffer, offset, len);
        }
        list.ResourceBarrier(&barriers);
        for &(buffer, offset, at, len) in writes {
            list.CopyBufferRegion(buffer, offset, between, at, len);
        }
        if !done.is_empty() {
            list.ResourceBarrier(&done);
        }
    }
}

/// For the frame-rate line.
pub(super) fn log(frames: f64) {
    if !on() {
        return;
    }
    let d: Vec<f64> = DONE.iter().map(|d| d.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let why: Vec<String> = std::mem::take(&mut *REFUSED.lock().unwrap_or_else(|e| e.into_inner()))
        .into_iter()
        .map(|(w, n)| format!("{w} {:.2}", n as f64 / frames))
        .collect();
    tracing::info!(
        "steps done on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS={}): {:.2} fills, {:.2} buffer \
         copies, {:.1} KB, {:.2} levels; {:.2} submissions refused and waited{}{}",
        stage(),
        d[0],
        d[1],
        d[2] / 1024.0,
        d[4],
        d[3],
        if why.is_empty() { "" } else { ": " },
        why.join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(dst: usize, src: usize, len: usize, asked: bool) -> Does {
        Does::Copy { dst, src, len, asked }
    }

    fn fill(src: usize, len: usize, asked: bool) -> Does {
        Does::Fill { src, len, asked, slot: Some((1, 2)), dst: src + 1_000_000 }
    }

    /// A fill without a slot of its own: the card cannot be given its bytes.
    fn stuck(src: usize, len: usize) -> Does {
        Does::Fill { src, len, asked: false, slot: None, dst: src + 2_000_000 }
    }

    /// What is done on the card, as (place, level), in stage 2.
    fn planned(does: &[Does]) -> Vec<(usize, u32)> {
        plan(does, 2).unwrap()
    }

    /// Only what is asked for is taken, all of it, and a step that stands
    /// in none of their ways stays where it is.
    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
            stuck(3100, 5),
        ];
        assert_eq!(planned(&does), [(1, 0), (3, 0)]);
        assert_eq!(plan(&[fill(0, 64, true), fill(32, 64, true)], 1), Ok(vec![(0, 0), (1, 0)]));
        assert_eq!(plan(&[fill(0, 64, false)], 1), Ok(vec![]));
        let does = [
            copy(1000, 5000, 100, true),
            copy(900, 9000, 100, false),
            copy(1100, 9000, 5, false),
            copy(4900, 9000, 100, false),
        ];
        assert_eq!(planned(&does), [(0, 0)]);
    }

    /// What depends on a taken step, or it on it, is taken too, on the level
    /// the order of the steps asks for.
    #[test]
    fn steps_that_depend_on_one_another_are_taken_in_order() {
        let taken = copy(1000, 5000, 100, true);
        assert_eq!(planned(&[taken, fill(1099, 64, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[fill(1099, 64, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(7000, 1050, 10, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[copy(5099, 9000, 1, false), taken]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(5099, 9000, 1, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, copy(999, 9000, 2, false)]), [(0, 0), (1, 1)]);
        assert_eq!(planned(&[taken, taken]), [(0, 0), (1, 1)]);
        let does = [
            copy(8000, 7000, 50, false),
            taken,
            copy(2000, 1010, 20, false),
            fill(90_000, 10, false),
            fill(2000, 20, false),
            fill(2010, 20, true),
            copy(1000, 2005, 5, false),
            fill(1000, 100, false),
        ];
        assert_eq!(planned(&does), [(1, 0), (2, 1), (4, 2), (5, 2), (6, 2), (7, 3)]);
        assert_eq!(planned(&[taken, fill(5000, 100, false)]), [(0, 0)]);
        assert_eq!(planned(&[copy(7000, 5000, 10, false), taken]), [(1, 0)]);
    }

    /// What cannot be done in the order of the steps is refused.
    #[test]
    fn steps_that_cannot_be_ordered_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        assert!(plan(&[taken, Does::Read { from: 1050, to: 1060, asked: false }], 2).is_err());
        assert!(plan(&[taken, stuck(1050, 4)], 2).is_err());
        assert!(plan(&[Does::Run, taken], 2).is_err());
        assert!(plan(&[taken], 1).is_err());
        assert!(plan(&[copy(1000, 1050, 100, true)], 2).is_err());
        assert!(plan(&[Does::Fill { src: 1, len: 1, asked: true, slot: None, dst: 7 }], 2).is_err());
        // A read on the CPU ahead of the copy reads the bytes as they are.
        assert_eq!(planned(&[Does::Read { from: 1050, to: 1060, asked: false }, taken]), [(1, 0)]);
    }

    /// Whatever the steps, done level by level (in a level every read
    /// before every write) they read and leave what they read and leave one
    /// after the other, the CPU's steps ahead of all of them.
    #[test]
    fn levels_do_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        let (mut done, mut refused) = (0, 0);
        for _ in 0..4000 {
            let does: Vec<Does> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (random(60) as usize, random(60) as usize, 1 + random(8) as usize);
                    let asked = random(4) == 0;
                    match random(8) {
                        0..4 => copy(at, other, len, asked),
                        4..6 => fill(at, len, asked),
                        6 => stuck(at, len),
                        _ => Does::Read { from: at, to: at + len, asked: false },
                    }
                })
                .collect();
            let Ok(chosen) = plan(&does, 2) else {
                refused += 1;
                continue;
            };
            done += 1;
            let start: Vec<u32> = (0..80).collect();
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for d in &does {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                wanted.push(memory[from..to].to_vec());
                if let Does::Copy { dst, src, len, .. } = *d {
                    memory.copy_within(src..src + len, dst);
                }
            }
            let mut cpu = start.clone();
            let taken = |i: usize| chosen.iter().any(|c| c.0 == i);
            for (i, d) in does.iter().enumerate().filter(|&(i, _)| !taken(i)) {
                let (from, to) = match *d {
                    Does::Read { from, to, .. } => (from, to),
                    _ => reads_writes(d).unwrap().0,
                };
                assert_eq!(cpu[from..to], wanted[i][..], "step {i} of {does:?}");
                if let Does::Copy { dst, src, len, .. } = *d {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu.clone();
            for level in 0..=chosen.iter().map(|c| c.1).max().unwrap_or(0) {
                let of: Vec<usize> = chosen.iter().filter(|c| c.1 == level).map(|c| c.0).collect();
                let before = card.clone();
                for i in of {
                    let ((from, to), (dst, _)) = reads_writes(&does[i]).unwrap();
                    assert_eq!(before[from..to], wanted[i][..], "step {i} of {does:?}: {chosen:?}");
                    if matches!(does[i], Does::Copy { .. }) {
                        card[dst..dst + to - from].copy_from_slice(&before[from..to]);
                    }
                }
            }
            assert_eq!(card, memory, "{does:?}: {chosen:?}");
        }
        assert!(done > 1000 && refused > 100, "done {done}, refused {refused}");
    }

    #[test]
    fn a_refresh_of_another_copy_of_the_destination_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 0, to: 100 }, Does::Refresh { shadow: 8, from: 200, to: 300 }];
        assert!(elsewhere(&does, 8, 50, 60));
        assert!(!elsewhere(&does, 7, 50, 60));
        assert!(!elsewhere(&does, 8, 100, 200));
    }
}
