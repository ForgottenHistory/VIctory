//! Timeline profiling: zones, frame marks, plots and thread names for the
//! Tracy profiler (protocol 0.14.1).
//!
//! Built with the `profile` feature (`cargo build -p victory --features
//! profile`), the runtime streams to a `tracy-profiler` or `tracy-capture`
//! on this machine once one connects (on demand: nothing is collected
//! before). Without the feature every hook expands to nothing.
//!
//! Every hook is a no-op until [`start`] has run, so threads the title
//! starts early, or code shared with tools that never start the client,
//! need no care.
//!
//! ```ignore
//! fn submit() {
//!     victory_profile::zone!("submit");
//!     victory_profile::plot!("copied MB", mb);
//! }
//! ```

#[cfg(feature = "profile")]
#[doc(hidden)]
pub use tracy_client as __tracy;

/// Starts the client when `VICTORY_TRACY=1` (once per process; later calls
/// do nothing). Without it a profiling build runs as a plain one.
pub fn start() {
    #[cfg(feature = "profile")]
    if std::env::var_os("VICTORY_TRACY").is_some_and(|v| v == "1") {
        let _ = tracy_client::Client::start();
    }
}

/// Whether profiling is compiled in.
pub const ENABLED: bool = cfg!(feature = "profile");

/// Names the calling thread on the timeline.
pub fn thread_name(name: &str) {
    #[cfg(feature = "profile")]
    if let Some(c) = tracy_client::Client::running() {
        c.set_thread_name(name);
    }
    #[cfg(not(feature = "profile"))]
    let _ = name;
}

/// A text message on the timeline (shown in the messages list and as a
/// marker at the time it was sent).
pub fn message(text: &str) {
    #[cfg(feature = "profile")]
    if let Some(c) = tracy_client::Client::running() {
        c.message(text, 0);
    }
    #[cfg(not(feature = "profile"))]
    let _ = text;
}

/// The end of a frame (one per present).
pub fn frame() {
    #[cfg(feature = "profile")]
    if let Some(c) = tracy_client::Client::running() {
        c.frame_mark();
    }
}

/// A zone from here to the end of the enclosing block. `zone!("name")`, or
/// `zone!("name", text)` with a text the zone carries (evaluated only while
/// a viewer collects).
#[cfg(feature = "profile")]
#[macro_export]
macro_rules! zone {
    ($name:literal) => {
        let _victory_zone = $crate::__tracy::Client::running()
            .map(|c| c.span($crate::__tracy::span_location!($name), 0));
    };
    ($name:literal, $text:expr) => {
        let _victory_zone = $crate::__tracy::Client::running().map(|c| {
            let z = c.span($crate::__tracy::span_location!($name), 0);
            z.emit_text(&$text);
            z
        });
    };
}

#[cfg(not(feature = "profile"))]
#[macro_export]
macro_rules! zone {
    ($name:literal) => {};
    ($name:literal, $text:expr) => {};
}

/// A value plotted over time under `name` (a literal).
#[cfg(feature = "profile")]
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {
        if let Some(c) = $crate::__tracy::Client::running() {
            c.plot($crate::__tracy::plot_name!($name), ($value) as f64);
        }
    };
}

#[cfg(not(feature = "profile"))]
#[macro_export]
macro_rules! plot {
    // (the value is not evaluated: the closure is never called, it only
    // keeps what it uses from counting as unused)
    ($name:literal, $value:expr) => {
        let _ = || $value;
    };
}
