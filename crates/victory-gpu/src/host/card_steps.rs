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

/// The steps of `does` (by place) to be done on the card: all that are
/// asked for, or why not. Our list does them all in front of the
/// submission's, every read before every write, so that is right only if no
/// step, asked for or not, writes what one of them reads or writes, and no
/// CPU step reads what one of them writes.
fn plan(does: &[Does], stage: u32) -> Result<Vec<usize>, &'static str> {
    let mut asked = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Read { asked: true, .. } => return Err("a texture reads it"),
            Does::Fill { asked: true, slot: None, .. } => return Err("a fill without a descriptor slot of its own"),
            Does::Copy { asked: true, .. } if stage < 2 => return Err("a buffer copy (VICTORY_CARD_STEPS=2 does it on the card)"),
            Does::Copy { asked: true, .. } | Does::Fill { asked: true, .. } => asked.push(i),
            _ => {}
        }
    }
    for &i in &asked {
        let (src, dst) = match does[i] {
            Does::Copy { dst, src, len, .. } => ((src, src + len), Some((dst, dst + len))),
            Does::Fill { src, len, .. } => ((src, src + len), None),
            _ => continue,
        };
        for (k, d) in does.iter().enumerate() {
            let (read, written) = match *d {
                Does::Copy { dst, src, len, .. } => (Some((src, src + len)), Some((dst, dst + len))),
                Does::Fill { src, len, .. } => (Some((src, src + len)), None),
                Does::Read { from, to, .. } => (Some((from, to)), None),
                Does::Refresh { .. } | Does::Run => (None, None),
            };
            if written.is_some_and(|w| meet(w, src)) {
                return Err("a copy of the submission writes what it reads");
            }
            let Some(dst) = dst else {
                continue;
            };
            if k != i && written.is_some_and(|w| meet(w, dst)) {
                return Err("another copy of the submission writes what it writes");
            }
            if read.is_some_and(|r| meet(r, dst)) {
                return Err("a step of the submission reads what it writes");
            }
        }
    }
    Ok(asked)
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
static DONE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

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
        let writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(Writer::again));
            all
        };
        let mut ops: Vec<Op> = Vec::with_capacity(chosen.len());
        let mut at = 0u64;
        for &i in &chosen {
            let (src, len) = match does[i] {
                Does::Copy { src, len, .. } | Does::Fill { src, len, .. } => (src, len),
                _ => continue,
            };
            if len == 0 {
                return Err("nothing to read");
            }
            if at + len as u64 > BETWEEN {
                return Err("too much at once");
            }
            let from = holder(&writers, src, src + len)?;
            if !mapped(src, src + len) {
                return Err("not all of what it reads is mapped");
            }
            let to = match does[i] {
                Does::Copy { dst, .. } => {
                    let to = holder(&writers, dst, dst + len)?;
                    if elsewhere(&does, Arc::as_ptr(&to) as usize, dst, dst + len) {
                        return Err("another host copy of what it writes is read");
                    }
                    if !mapped(dst, dst + len) {
                        return Err("not all of what it writes is mapped");
                    }
                    To::Copy(to, dst)
                }
                Does::Fill { slot: Some((heap, index)), dst, .. } => To::Slot(heap, index, dst),
                _ => continue,
            };
            ops.push(Op { from: (from, src), len, at, to });
            at = (at + len as u64).next_multiple_of(256);
        }
        let Some(own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut copied = Vec::new();
        let mut skips = Skips::default();
        // First every refresh, then the write: the bytes compared with are
        // the title's.
        for op in &ops {
            op.from.0.refresh(&own.list, op.from.1 as u64, op.len as u64);
            reads.push((&op.from.0.buffer, op.from.0.offset(op.from.1 as u64), op.at, op.len as u64));
            if let To::Copy(to, dst) = &op.to {
                to.refresh(&own.list, *dst as u64, op.len as u64);
                writes.push((&to.buffer, to.offset(*dst as u64), op.at, op.len as u64));
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
        own.list.note_op(|| format!("{} steps of title memory done on the card, {at} bytes (VICTORY_CARD_STEPS)", ops.len()));
        record(&own.list.list, &own.between, &reads, &writes);
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

/// Records into `list` (open, the first of its call): `reads` copied into
/// `between` (source, offset in it, offset in `between`, bytes), then
/// `writes` out of it (destination, offset in it, offset in `between`,
/// bytes). `between` is left readable as constants, the others in the
/// common state, as the submission's lists expect them; buffers decay to
/// the common state when the call has run.
fn record(
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
    let mut barriers = vec![transition(between, dest, source | D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)];
    barriers.extend(sources.iter().map(|b| transition(b, source, if both(b) { dest } else { common })));
    let done: Vec<D3D12_RESOURCE_BARRIER> = dests.iter().map(|b| transition(b, dest, common)).collect();
    // SAFETY: an open list; live buffers, and the callers' ranges lie in
    // them. The buffers start in the common state (the list is the first of
    // its call) and are promoted by their first use.
    unsafe {
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
         copies, {:.1} KB; {:.2} submissions refused and waited{}{}",
        stage(),
        d[0],
        d[1],
        d[2] / 1024.0,
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

    /// Only what is asked for is taken, all of it, and a step that stands
    /// in none of their ways changes nothing.
    #[test]
    fn steps_apart_from_one_another_are_taken() {
        let does = [
            copy(1000, 5000, 100, false),
            fill(2000, 64, true),
            Does::Refresh { shadow: 1, from: 0, to: 9000 },
            copy(3000, 6000, 100, true),
            fill(1000, 64, false),
            Does::Read { from: 3100, to: 4000, asked: false },
        ];
        assert_eq!(plan(&does, 2), Ok(vec![1, 3]));
        assert_eq!(plan(&[fill(0, 64, true), fill(32, 64, true)], 1), Ok(vec![0, 1]));
        assert_eq!(plan(&[fill(0, 64, false)], 1), Ok(vec![]));
    }

    /// Everything that could make the card's order another than the CPU's
    /// refuses the lot, wherever in the submission it stands.
    #[test]
    fn steps_in_one_anothers_way_are_refused() {
        let taken = copy(1000, 5000, 100, true);
        let refused = |other: Does| {
            assert!(plan(&[taken, other], 2).is_err(), "{other:?} after");
            assert!(plan(&[other, taken], 2).is_err(), "{other:?} before");
        };
        for asked in [false, true] {
            refused(copy(5099, 9000, 1, asked));
            refused(copy(999, 9000, 2, asked));
            refused(copy(7000, 1050, 10, asked));
            refused(fill(1099, 64, asked));
            refused(Does::Read { from: 900, to: 1001, asked });
        }
        refused(Does::Run);
        refused(Does::Read { from: 1, to: 2, asked: true });
        refused(Does::Fill { src: 1, len: 1, asked: true, slot: None, dst: 7 });
        assert!(plan(&[copy(1000, 1050, 100, true)], 2).is_err());
        assert!(plan(&[taken, taken], 2).is_err());
        assert!(plan(&[fill(1000, 4, true), copy(1000, 5000, 100, false)], 2).is_err());
        assert!(plan(&[taken], 1).is_err());
        assert!(plan(&[taken, copy(900, 9000, 100, false), copy(1100, 9000, 5, false), copy(4900, 9000, 100, false)], 2).is_ok());
    }

    #[test]
    fn a_refresh_of_another_copy_of_the_destination_is_seen() {
        let does = [Does::Refresh { shadow: 7, from: 0, to: 100 }, Does::Refresh { shadow: 8, from: 200, to: 300 }];
        assert!(elsewhere(&does, 8, 50, 60));
        assert!(!elsewhere(&does, 7, 50, 60));
        assert!(!elsewhere(&does, 8, 100, 200));
    }
}
