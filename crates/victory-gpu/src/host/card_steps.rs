//! Constant buffers of a submission that read title memory flights in the
//! air may still write, filled on the card in the flights' order instead of
//! on the CPU after a wait for them (`VICTORY_CARD_STEPS=1`; off by
//! default). After avatarmanz/VIctory-7900XTX cc03ecb (its first stage),
//! fitted to this tree.
//!
//! Why: the world's compute passes bind the title's 174 MB heap whole for
//! unordered access, so every flight declares all of it written, and the
//! title's ring of constants lies in the same heap. A constant buffer of a
//! descriptor table (480 bytes) read from it made the submission wait for
//! the flights in the air to land: 1.5 times a frame, 24 ms a frame of the
//! submitting thread (Tracy capture 2026-11-26, with argument batches).
//! Skipping the wait (`VICTORY_IGNORE_FILL_READS=1`) gave 7.4 -> 8.0 fps but
//! may read bytes from before the shaders' writes; learning where shaders
//! do not write (`VICTORY_LEARN_UNWRITTEN`) was wrong 689 times while the
//! world loaded.
//!
//! What is done: a list of ours goes to the card in front of the
//! submission's lists, in the same call. It copies the bytes the fill reads
//! from the host copy that the flights write into a buffer of ours, and the
//! fill's descriptor slot is pointed at that buffer instead of the upload
//! memory the CPU would have filled (the slot is the fill's alone: a table
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
//! Nothing of the submission writes these bytes before our list does its
//! copy on the CPU's side: [`plan`] refuses when a step of the submission
//! writes what a fill taken reads, or runs work of ours that may write
//! anything. Where a page is not mapped the host copy may hold other bytes
//! than the CPU's copy would have left: refused.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use super::*;
use crate::shadow::Shadow;

/// Whether fills are done on the card (`VICTORY_CARD_STEPS=1`). Only with
/// the relay's exact waits (`relay_exact`).
pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_STEPS").is_some_and(|v| v == "1"))
}

/// The bytes of our buffer between reading and using: what one submission
/// may do this way.
const BETWEEN: u64 = 1 << 20;

/// Title memory something in the air writes: the host copy it goes through
/// (`None`: none, the bytes reach title memory when the flight lands), from,
/// to.
pub(super) struct Writer {
    shadow: Option<Arc<Shadow>>,
    from: usize,
    to: usize,
}

/// What `items` of the line write.
pub(super) fn writers(items: &[Late]) -> Vec<Writer> {
    let mut out = Vec::new();
    for item in items {
        match item {
            Late::Run(flight) => {
                for (shadow, ranges) in &flight.back {
                    out.extend(ranges.iter().map(|&(from, to)| Writer { shadow: Some(shadow.clone()), from, to }));
                }
                out.extend(flight.texture_writes.iter().map(|&(from, to)| Writer { shadow: None, from, to }));
            }
            Late::Write(at, _, bytes) => out.push(Writer { shadow: None, from: *at, to: *at + *bytes }),
            Late::Signal(..) => {}
        }
    }
    out
}

/// What a step of a submission does to title memory, as the plan sees it.
/// `asked`: flights in the air stop it from being done on the CPU now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Does {
    /// A title buffer copy: written on the CPU.
    Copy { dst: usize, src: usize, len: usize, asked: bool },
    Fill { src: usize, len: usize, asked: bool, slot: Option<(usize, u32)>, dst: usize },
    /// Work of ours that may write title memory anywhere.
    Run,
}

fn meet(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// The fills of `does` (by place) to be done on the card: all that are
/// asked for, or why not. Our list does them in front of the submission's,
/// so that is right only if no step of the submission writes what one of
/// them reads.
fn plan(does: &[Does]) -> Result<Vec<usize>, &'static str> {
    let mut asked = Vec::new();
    for (i, d) in does.iter().enumerate() {
        match *d {
            Does::Run => return Err("a step of ours runs"),
            Does::Fill { asked: true, slot: None, .. } => return Err("a fill without a descriptor slot of its own"),
            Does::Copy { asked: true, .. } => return Err("a buffer copy waits too"),
            Does::Fill { asked: true, .. } => asked.push(i),
            _ => {}
        }
    }
    for &i in &asked {
        let Does::Fill { src, len, .. } = does[i] else { continue };
        if does.iter().any(|d| matches!(*d, Does::Copy { dst, len: n, .. } if meet((dst, dst + n), (src, src + len)))) {
            return Err("a copy of the submission writes what it reads");
        }
    }
    Ok(asked)
}

/// The host copy that holds `[from, to)` on the card once what `writers`
/// write is there: the one they all write it through.
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

/// Since the last frame-rate line: fills done on the card, their bytes,
/// submissions that waited instead, refused by a check.
static DONE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Why the last refusals were, since the last frame-rate line.
static REFUSED: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

fn refused(why: &'static str) {
    DONE[2].fetch_add(1, Ordering::Relaxed);
    let mut all = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    match all.iter_mut().find(|(w, _)| *w == why) {
        Some(e) => e.1 += 1,
        None => all.push((why, 1)),
    }
}

/// What [`Host::card_steps`] took off a submission: the list that does it,
/// for the submission to run first, and the fills the CPU leaves out (by
/// their place in upload memory). Its list goes back to the pool when it is
/// dropped: with the flight it went with, once that has landed.
pub(super) struct Taken {
    own: Option<Own>,
    fills: std::collections::HashSet<usize>,
}

impl Taken {
    pub(super) fn list(&self) -> &HostList {
        &self.own.as_ref().expect("a taken list").list
    }

    /// The fills done by this, by their destination in upload memory.
    pub(super) fn fills(&self) -> &std::collections::HashSet<usize> {
        &self.fills
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
    /// now is only fills that can all be done on the card ([`on`]), the
    /// list that does them. `None` if nothing is in the way or the old wait
    /// is due. Under the submission lock.
    pub(super) fn card_steps(&self, runs: &[&HostList]) -> Option<Taken> {
        if !on() || !relay_exact() {
            return None;
        }
        let written = {
            let line = line();
            if line.written.is_empty() {
                return None;
            }
            line.written.clone()
        };
        blocked_by(&written, runs, false, None)?;
        match self.take_fills(&written, runs) {
            Ok(Some(taken)) => {
                // Nothing else of the submission waits for the flights.
                if let Some(why) = blocked_by(&written, runs, false, Some(taken.fills())) {
                    let _ = why;
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

    fn take_fills(&self, written: &PendingWrites, runs: &[&HostList]) -> Result<Option<Taken>, &'static str> {
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
                    Step::Run(_) => does.push(Does::Run),
                    _ => {}
                }
            }
        }
        let chosen = plan(&does)?;
        if chosen.is_empty() {
            return Ok(None);
        }
        let writers = {
            let line = line();
            let mut all = writers(&line.line);
            all.extend(line.landing.iter().map(|w| Writer { shadow: w.shadow.clone(), from: w.from, to: w.to }));
            all
        };
        // (holder, title address, bytes, place in the buffer, slot, fill's destination)
        let mut ops: Vec<(Arc<Shadow>, usize, usize, u64, (usize, u32), usize)> = Vec::with_capacity(chosen.len());
        let mut at = 0u64;
        for &i in &chosen {
            let Does::Fill { src, len, slot: Some(slot), dst, .. } = does[i] else { continue };
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
            ops.push((from, src, len, at, slot, dst));
            at = (at + len as u64).next_multiple_of(256);
        }
        let Some(own) = self.own_list() else {
            return Err("no host list or buffer for it");
        };
        // The title's own bytes first: a refresh of our list.
        for (shadow, src, len, ..) in &ops {
            shadow.refresh(&own.list, *src as u64, *len as u64);
        }
        #[cfg(feature = "diagnostics")]
        own.list.note_op(|| format!("{} constant buffers filled on the card, {at} bytes (VICTORY_CARD_STEPS)", ops.len()));
        let reads: Vec<(&ID3D12Resource, u64, u64, u64)> =
            ops.iter().map(|(s, src, len, at, ..)| (&s.buffer, s.offset(*src as u64), *at, *len as u64)).collect();
        record(&own.list.list, &own.between, &reads);
        own.list.close();
        // SAFETY: a live buffer.
        let gpu = unsafe { own.between.GetGPUVirtualAddress() };
        let mut fills = std::collections::HashSet::new();
        for (_, _, len, at, (heap, index), dst) in &ops {
            DONE[0].fetch_add(1, Ordering::Relaxed);
            DONE[1].fetch_add(*len as u64, Ordering::Relaxed);
            // (A heap that is gone: nothing can read the slot.)
            crate::descriptor::constant_slot(*heap, *index, gpu + at, *len as u32);
            fills.insert(*dst);
        }
        Ok(Some(Taken { own: Some(own), fills }))
    }

    /// A list of ours open for recording, and its buffer.
    fn own_list(&self) -> Option<Own> {
        let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).pop();
        if let Some(own) = pooled {
            own.list.reset();
            return Some(own);
        }
        let between = self.uav_buffer(BETWEEN)?;
        name(&between, || "constant buffers filled on the card".into());
        let list = self.new_list().ok()?;
        Some(Own { list, between })
    }
}

/// Records into `list` (open, the first of its call): `reads` copied into
/// `between` (source, offset in it, offset in `between`, bytes). `between`
/// is left readable as constants, the sources in the common state, as the
/// submission's lists expect them; buffers decay to the common state when
/// the call has run.
fn record(list: &ID3D12GraphicsCommandList, between: &ID3D12Resource, reads: &[(&ID3D12Resource, u64, u64, u64)]) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let (source, dest) = (D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut sources: Vec<&ID3D12Resource> = Vec::new();
    for &(buffer, ..) in reads {
        if !sources.iter().any(|b| b.as_raw() == buffer.as_raw()) {
            sources.push(buffer);
        }
    }
    let mut barriers = vec![transition(between, dest, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)];
    barriers.extend(sources.iter().map(|b| transition(b, source, common)));
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
        "constant buffers filled on the card instead of waiting for flights, per frame (VICTORY_CARD_STEPS=1): {:.2} fills, \
         {:.1} KB; {:.2} submissions refused and waited{}{}",
        d[0],
        d[1] / 1024.0,
        d[2],
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

    #[test]
    fn fills_apart_from_other_steps_are_taken() {
        let does = [copy(1000, 5000, 100, false), fill(2000, 64, true), fill(3000, 64, false), fill(2032, 64, true)];
        assert_eq!(plan(&does), Ok(vec![1, 3]));
        assert_eq!(plan(&[fill(0, 64, false)]), Ok(vec![]));
    }

    #[test]
    fn fills_in_the_way_of_others_are_refused() {
        // A copy of the submission writes what the fill reads, before or after.
        assert!(plan(&[copy(2010, 9000, 4, false), fill(2000, 64, true)]).is_err());
        assert!(plan(&[fill(2000, 64, true), copy(2060, 9000, 8, false)]).is_err());
        // Ends that touch are apart.
        assert_eq!(plan(&[copy(1900, 9000, 100, false), fill(2000, 64, true)]), Ok(vec![1]));
        assert!(plan(&[Does::Run, fill(2000, 64, true)]).is_err());
        assert!(plan(&[copy(1, 2, 3, true), fill(2000, 64, true)]).is_err());
        assert!(plan(&[Does::Fill { src: 1, len: 1, asked: true, slot: None, dst: 7 }]).is_err());
    }
}
