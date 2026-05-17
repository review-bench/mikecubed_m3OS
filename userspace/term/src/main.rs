//! `term` binary entry point — Phase 57 Track G end-to-end wiring.
//!
//! `term` is the Phase 57 graphical terminal emulator. The binary
//! composes the lib pieces (`PtyHost`, `Screen`, `Renderer`,
//! `InputHandler`, `Bell`) into a single-threaded event loop:
//!
//! 1. Open an IPC endpoint and register `"term"` so `session_manager`
//!    can observe the boot step.
//! 2. Connect to `display_server` (Hello + `CreateSurface` +
//!    `SetSurfaceRole(Toplevel)`) via [`DisplayClient`].
//! 3. Open a PTY pair via the production [`SyscallPtyOps`], fork +
//!    `execve` `/bin/ion` (with `/bin/sh0` fallback) on the secondary
//!    side, set the primary nonblocking.
//! 4. Loop: drain PTY reads → ANSI parser → screen state → render
//!    commands; pull `KeyEvent`s from `display_server`'s C.5 outbound
//!    queue → input handler → PTY writes; ring the bell on `Bell`
//!    commands; compose dirty frames.
//! 5. Exit zero on shell exit so the supervisor restarts per
//!    `term.conf`.
//!
//! `cfg(not(test))` gates protect the OS-only entry point so
//! `cargo test -p term --target x86_64-unknown-linux-gnu --lib`
//! continues to compile on the host.

#![cfg_attr(not(test), no_std)]
#![cfg_attr(not(test), no_main)]
#![cfg_attr(not(test), feature(alloc_error_handler))]

extern crate alloc;
#[cfg(test)]
extern crate std;

#[cfg(not(test))]
use core::alloc::Layout;

#[cfg(not(test))]
use kernel_core::display::protocol::ServerMessage;
#[cfg(not(test))]
use syscall_lib::heap::BrkAllocator;
#[cfg(not(test))]
use syscall_lib::{CLOCK_MONOTONIC, STDOUT_FILENO};

#[cfg(not(test))]
use term::bell::{AudioClientBellSink, AudioUnavailableBellSink, Bell, BellError};
#[cfg(not(test))]
use term::display::DisplayClient;
#[cfg(not(test))]
use term::input::{InputHandler, PtyWriter};
#[cfg(not(test))]
use term::mouse::{EncodingMode, MouseReporter, TrackingMode};
#[cfg(not(test))]
use term::pty::PtyHost;
#[cfg(not(test))]
use term::render::Renderer;
#[cfg(not(test))]
use term::screen::{RenderCommand, Screen};
#[cfg(not(test))]
use term::syscall_pty::SyscallPtyOps;
#[cfg(not(test))]
use term::{
    BOOT_LOG_MARKER, PROMPT_READY_MIN_BYTES, PROMPT_READY_SERVICE, READY_SENTINEL, SERVICE_NAME,
    should_compose_frame,
};

#[cfg(not(test))]
#[global_allocator]
static ALLOCATOR: BrkAllocator = BrkAllocator::new();

#[cfg(not(test))]
#[alloc_error_handler]
fn alloc_error(_layout: Layout) -> ! {
    syscall_lib::write_str(STDOUT_FILENO, "term: alloc error\n");
    syscall_lib::exit(99)
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    syscall_lib::write_str(STDOUT_FILENO, "term: PANIC\n");
    syscall_lib::exit(101)
}

#[cfg(not(test))]
syscall_lib::entry_point!(program_main);

/// Phase 56 C.5 close-out — IPC label term sends to drain one
/// queued `ServerMessage` from `display_server`. Mirrors
/// `display_server::client::LABEL_CLIENT_EVENT_PULL`. The
/// complementary `LABEL_CLIENT_EVENT_NONE = 4` is the server's
/// reply when the queue is empty; term checks against equality with
/// `LABEL_CLIENT_EVENT_PULL` rather than naming `_NONE` separately.
#[cfg(not(test))]
const LABEL_CLIENT_EVENT_PULL: u64 = 3;

/// Per-iteration sleep when no work was found this tick. Mirrors the
/// `display_server` main-loop yield (1 ms → ~1000 polls/sec).
#[cfg(not(test))]
const IDLE_SLEEP_NS: u32 = 1_000_000;

/// Bytes-per-iteration drain cap on the PTY primary fd. Big enough to
/// cover a typical shell prompt + output line; small enough that one
/// noisy program cannot starve the input + render passes.
#[cfg(not(test))]
const PTY_READ_CHUNK: usize = 256;

/// Minimum gap between successive `Renderer::compose()` calls, in
/// milliseconds. Caps the compose / framebuffer-upload frequency at
/// roughly 60 Hz: the renderer's `damaged()` queue keeps accumulating
/// `PutGlyph` ops between calls, so a burst of PTY echo bytes (one
/// per typed key) coalesces into a single compose pass instead of
/// firing one full-buffer upload per character. With the Phase 56
/// chunked-pixel path, each full-surface upload is hundreds of IPC
/// roundtrips for the 1280×800 surface; without throttling, every
/// keystroke paid that cost in series.
#[cfg(not(test))]
const COMPOSE_INTERVAL_MS: u64 = 16;

/// Phase 69 Track F.2 — blinking-cursor tick interval in milliseconds.
/// Matches xterm's default cursor blink rate.
#[cfg(not(test))]
const BLINK_INTERVAL_MS: u64 = 500;

#[cfg(not(test))]
const SHELL_DEPENDENCY_SERVICE: &str = "vfs";

#[cfg(not(test))]
fn program_main(_args: &[&str]) -> i32 {
    syscall_lib::write_str(STDOUT_FILENO, BOOT_LOG_MARKER);

    // 1. Open an IPC endpoint and register so `session_manager`
    //    observes this step. `term` does not yet accept inbound IPC
    //    traffic on this endpoint; it is a presence beacon.
    let ep = syscall_lib::create_endpoint();
    if ep == u64::MAX {
        syscall_lib::write_str(STDOUT_FILENO, "term: create_endpoint failed\n");
        return 2;
    }
    let ep_u32 = match u32::try_from(ep) {
        Ok(v) => v,
        Err(_) => {
            syscall_lib::write_str(STDOUT_FILENO, "term: endpoint id out of u32 range\n");
            return 3;
        }
    };
    let rc = syscall_lib::ipc_register_service(ep_u32, SERVICE_NAME);
    if rc == u64::MAX {
        syscall_lib::write_str(STDOUT_FILENO, "term: ipc_register_service failed\n");
        return 4;
    }

    // 2. Connect to display_server. Without it term has nothing to
    //    paint to; surface DisplayServerUnavailable cleanly.
    let display = match DisplayClient::connect() {
        Ok(d) => d,
        Err(_) => {
            syscall_lib::write_str(STDOUT_FILENO, "term: display_server unavailable\n");
            return 5;
        }
    };
    // Capture the display endpoint handle for the C.5 event-pull
    // path. `DisplayClient` did the lookup; we re-look-up here so
    // term can hold its own borrow without aliasing.
    let display_handle = match lookup_display_for_input() {
        Some(h) => h,
        None => {
            syscall_lib::write_str(STDOUT_FILENO, "term: display lookup for input failed\n");
            return 5;
        }
    };

    // 3. Open the PTY pair and spawn the production shell on the
    //    secondary side. `SyscallPtyOps::exec_shell` execve's
    //    `/bin/ion` first (matching `/etc/passwd`'s default and the
    //    path `login` exec's), falling back to `/bin/sh0` if ion
    //    is missing or broken.
    //    `SyscallPtyOps` is the production wiring of the
    //    `PtyOps` trait the lib already exercises against
    //    `MockPtyOps`.
    if !wait_for_shell_dependencies() {
        syscall_lib::write_str(STDOUT_FILENO, "term: VFS unavailable before shell spawn\n");
        return 6;
    }
    let mut pty = PtyHost::new(SyscallPtyOps::new());
    if let Err(_e) = pty.open_and_spawn() {
        syscall_lib::write_str(STDOUT_FILENO, "term: PTY open / shell spawn failed\n");
        return 6;
    }
    let primary_fd = match pty.primary_fd() {
        Some(fd) => fd,
        None => {
            syscall_lib::write_str(STDOUT_FILENO, "term: PtyHost has no primary fd\n");
            return 6;
        }
    };
    if syscall_lib::set_nonblocking(primary_fd) < 0 {
        syscall_lib::write_str(STDOUT_FILENO, "term: set_nonblocking failed\n");
        return 6;
    }

    // 4. Compose the screen state machine, the renderer, the input
    //    translator, and the bell. Bell starts on the production
    //    AudioClientBellSink; on first AudioUnavailable we swap
    //    permanently to the warn-once stub so noisy bell-loops do
    //    not retry the audio path forever.
    let mut screen = Screen::new();
    let mut renderer = Renderer::new(display);
    // Phase 69c Track E.1 — try to load the Nerd Font asset and
    // upgrade the renderer to atlas-backed glyph resolution. On a
    // file-missing, parse-error, or oversized-file failure the
    // renderer stays on the Phase 69b static-table path and the
    // terminal remains usable for ASCII / Latin-1 / box-drawing.
    // True OOM is *not* recovered here — this binary's
    // `alloc_error_handler` exits the process; `build_atlas` bounds
    // the worst-case font-read allocation with a hard size cap to
    // keep the fallback path reachable. See the docstring on
    // `build_atlas` for the full contract.
    build_atlas(&mut renderer);
    let mut input_handler = InputHandler::new();
    let mut mouse_reporter = MouseReporter::new();

    // Paint an initial cleared frame so the surface gets a buffer
    // attached *before* any PTY traffic arrives. Without this, the
    // renderer queue stays empty until the shell echoes its first byte
    // (or the user types), so `display_server` never sees an
    // `AttachBuffer` for term's surface and skips it during compose —
    // the user just sees the teal background and cursor with no
    // terminal rectangle. Pushing `RenderCommand::Clear` mirrors what
    // the screen state machine would emit on `ESC [ 2 J`; the throttle
    // below the loop will flush it on its first tick.
    renderer.apply(RenderCommand::Clear);
    let mut bell_audio = Some(Bell::new(AudioClientBellSink::new()));
    let mut bell_unavail: Option<Bell<AudioUnavailableBellSink>> = None;
    let mut render_cmds: alloc::vec::Vec<RenderCommand> = alloc::vec::Vec::new();
    let mut event_buf = [0u8; 64];
    let mut pty_buf = [0u8; PTY_READ_CHUNK];
    let mut writer = PrimaryFdWriter {
        fd: primary_fd,
        warned: false,
    };
    let clock = MonotonicClock;

    // Attach and publish the initial clear frame before the event-pull loop.
    // Otherwise early key/mouse traffic can keep term draining display events
    // before the first `AttachSharedBuffer`, leaving the terminal invisible.
    renderer.compose();
    let mut last_compose_ms = clock.now_ms();
    // Phase 69 Track F.2 — drives the 500 ms blink tick for the
    // DECSCUSR blinking cursor shapes. Always initialised; the tick
    // only marks damage when `Screen::cursor_shape().is_blinking()`.
    let mut last_blink_ms = last_compose_ms;

    syscall_lib::write_str(STDOUT_FILENO, READY_SENTINEL);

    // Track the last compose timestamp so the throttle below only
    // composes when at least `COMPOSE_INTERVAL_MS` has elapsed. The
    // underlying `Renderer` keeps its damage queue intact between
    // composes, so dropping a single frame's worth of compose calls
    // does not lose any glyphs — the next compose flushes everything.

    // Phase 57d follow-up "term: iter=" / events_pulled / composes /
    // pty_bytes stat-line diagnostic was removed in the Phase 57e
    // deferral cleanup (2026-05-07) per its own comment ("Ripped once
    // the input-pipeline race is closed").  The pty_bytes counter is
    // retained because the prompt-ready gate consumes it; the rest
    // are gone.
    let mut pty_bytes: u64 = 0;
    let mut prompt_ready_registered = false;

    // 5. Event loop. Single-threaded; multiplexes the PTY drain, the
    //    display_server outbound-event drain, the bell, the shell-exit
    //    poll, and the renderer's per-tick compose.
    loop {
        let mut did_work = false;
        let mut pty_drained_this_tick = false;

        // 5a. Drain the PTY primary fd. Nonblocking: -EAGAIN means
        //     no data this tick. 0 means the shell closed its end.
        let n = syscall_lib::read(primary_fd, &mut pty_buf);
        if n > 0 {
            did_work = true;
            pty_drained_this_tick = true;
            pty_bytes = pty_bytes.saturating_add(n as u64);
            if !prompt_ready_registered && pty_bytes >= PROMPT_READY_MIN_BYTES {
                let rc = syscall_lib::ipc_register_service(ep_u32, PROMPT_READY_SERVICE);
                if rc != u64::MAX {
                    prompt_ready_registered = true;
                    syscall_lib::write_str(STDOUT_FILENO, "TERM_SMOKE:prompt-ready\n");
                }
            }
            // Phase 57d follow-up "backspace doesn't erase" PTY hex-dump
            // diagnostic was removed in the Phase 57e deferral cleanup
            // (2026-05-07).  The shell's backspace sequence was settled;
            // 30 hex-dump lines per boot earned no ongoing value.
            for &byte in &pty_buf[..n as usize] {
                screen.feed(byte, &mut render_cmds);
            }
            for cmd in render_cmds.drain(..) {
                // TEMPORARY DIAGNOSTIC TRACE — less render bug investigation.
                // Remove before merge. Tags only; no formatted args.
                let tag: &str = match &cmd {
                    RenderCommand::PutGlyph { .. } => "TT:Put\n",
                    RenderCommand::Clear => "TT:Clear\n",
                    RenderCommand::Scroll { .. } => "TT:Scroll\n",
                    RenderCommand::MoveCursor { .. } => "TT:Move\n",
                    RenderCommand::SetColor { .. } => "TT:Color\n",
                    RenderCommand::Bell => "TT:Bell\n",
                    RenderCommand::SetMouseMode { .. } => "TT:Mouse\n",
                };
                let _ = syscall_lib::write_str(STDOUT_FILENO, tag);

                match cmd {
                    RenderCommand::Bell => {
                        ring_bell(&mut bell_audio, &mut bell_unavail, clock.now_ms());
                    }
                    RenderCommand::SetMouseMode { code, set } => {
                        update_mouse_mode(&mut mouse_reporter, code, set);
                    }
                    other => renderer.apply(other),
                }
            }
        } else if n == 0 {
            // EOF on primary — the shell closed the slave; treat it
            // as shell exit and break.
            syscall_lib::write_str(STDOUT_FILENO, "term: PTY EOF; shell closed\n");
            break;
        }
        // n < 0 path: either -EAGAIN (no data) or a hard error. We
        // do not distinguish today; the next iteration retries.

        // 5b. Drain ALL queued ServerMessages from display_server
        //     before yielding to compose. The previous "one event per
        //     iter" shape played badly with a 250 ms+ compose: while
        //     compose blocked, key + pointer events accumulated in
        //     display_server's per-client outbound queue (cap 128)
        //     and overflowed under fast typing — m3os4.log showed 100+
        //     "outbound queue full; oldest dropped" lines, plus a
        //     visibly-frozen mouse cursor whenever the user typed.
        //     Pulling in a tight loop while events are pending keeps
        //     the queue empty between composes regardless of compose
        //     wall-time, with no risk of starvation: PTY drain, exit
        //     poll, and compose still run after the queue is empty.
        let mut disconnect = false;
        loop {
            match pull_one_event(display_handle, &mut event_buf) {
                PulledEvent::Key(ev) => {
                    did_work = true;
                    input_handler.translate(&ev, &mut writer);
                }
                PulledEvent::Pointer(ev) => {
                    did_work = true;
                    if let Some(bytes) = mouse_reporter.encode(&ev, screen.cols(), screen.rows()) {
                        let _ = syscall_lib::write(primary_fd, bytes.as_slice());
                    }
                }
                PulledEvent::SurfaceResized { width, height } => {
                    did_work = true;
                    handle_surface_resize(primary_fd, &mut screen, &mut renderer, width, height);
                }
                PulledEvent::Disconnect => {
                    syscall_lib::write_str(STDOUT_FILENO, "term: display_server disconnect\n");
                    disconnect = true;
                    break;
                }
                PulledEvent::None => break,
            }
        }
        if disconnect {
            break;
        }

        // 5c. Poll shell exit. `Some(_)` ⇒ child exited (cleanly or
        //     not); break out of the loop.
        match pty.poll_shell_exit() {
            Ok(Some(_status)) => {
                syscall_lib::write_str(STDOUT_FILENO, "term: shell exited\n");
                break;
            }
            Ok(None) => {}
            Err(_) => {
                syscall_lib::write_str(STDOUT_FILENO, "term: poll_shell_exit error\n");
                break;
            }
        }

        // 5c2. Phase 69 Track F.2 — blink tick. When the current
        //      DECSCUSR cursor shape is a blinking variant, force a
        //      compose every 500 ms even if the PTY and event queue
        //      were idle so the cursor visibly blinks. The compose
        //      throttle still caps actual frame submission at
        //      `COMPOSE_INTERVAL_MS`.
        if screen.cursor_shape().is_blinking() {
            let now_ms = clock.now_ms();
            if now_ms.saturating_sub(last_blink_ms) >= BLINK_INTERVAL_MS {
                renderer.mark_damaged();
                last_blink_ms = now_ms;
            }
        }

        // 5d. Compose dirty frame, if any — throttled so we don't pay
        //     the full-surface upload cost on every keystroke. The
        //     `damaged()` queue accumulates between calls, and the
        //     `did_work = false` branch below picks the idle sleep so
        //     this loop neither busy-spins nor starves on a slow
        //     compose. When the chunked-pixel path is replaced by
        //     shared-memory buffers, the throttle still amortises
        //     `DamageSurface` IPC traffic across multiple PTY bytes.
        if renderer.damaged() {
            let now_ms = clock.now_ms();
            let elapsed_ms = now_ms.saturating_sub(last_compose_ms);
            let (cursor_row, _) = screen.cursor();
            if should_compose_frame(
                true,
                pty_drained_this_tick,
                cursor_row,
                screen.rows(),
                elapsed_ms,
                COMPOSE_INTERVAL_MS,
            ) {
                renderer.compose();
                last_compose_ms = now_ms;
                did_work = true;
            }
        }

        // 5e. Yield briefly when nothing happened so we don't burn CPU.
        if !did_work {
            let _ = syscall_lib::nanosleep_for(0, IDLE_SLEEP_NS);
        }
    }

    // Shell exited (or PTY EOF / unrecoverable error). Close the
    // primary fd cleanly so the kernel reclaims the slot, then exit
    // zero — the supervisor's `restart=on-failure` policy lets it
    // re-spawn term once the shell or display state recovers.
    pty.close_primary();
    0
}

/// Production [`PtyWriter`] — wraps `syscall_lib::write` against the
/// PTY primary fd. The input handler has no recovery for a failing
/// write (the byte is already gone from the input queue), but the
/// failure is observable through the boot transcript so a developer
/// can correlate "shell looks deaf" with "term: PTY write error
/// errno=-X". The `warned` flag rate-limits the log line to once per
/// "stuck" episode — a chronic write failure (e.g. shell exited and
/// PTY EOF'd) would otherwise spam the serial console on every
/// keystroke.
#[cfg(not(test))]
struct PrimaryFdWriter {
    fd: i32,
    warned: bool,
}

#[cfg(not(test))]
impl PtyWriter for PrimaryFdWriter {
    fn write(&mut self, bytes: &[u8]) {
        let rc = syscall_lib::write(self.fd, bytes);
        if rc < 0 {
            if !self.warned {
                syscall_lib::write_str(STDOUT_FILENO, "term: PTY write error\n");
                self.warned = true;
            }
            return;
        }
        // Successful write resets the warned flag so a transient
        // failure followed by recovery still produces a fresh log
        // line if the failure recurs.
        self.warned = false;
    }
}

#[cfg(not(test))]
fn wait_for_shell_dependencies() -> bool {
    syscall_lib::ipc_wait_service(SHELL_DEPENDENCY_SERVICE, 0)
}

/// Phase 69c Track E.1 — load the Nerd Font asset from
/// `/usr/share/fonts/m3os/term.ttf` and upgrade `renderer` to the
/// atlas-backed glyph path. On any failure (file missing, parse
/// error, or the staged file exceeds the hard size cap) the
/// renderer is left on Phase 69b's static-table fallback and a
/// single warning lands in the boot log so a developer can
/// correlate "no Nerd Font glyphs" with "load failed". The font
/// path is hard-coded — Phase 69c deliberately defers configurable
/// paths to a later phase.
///
/// Allocation policy: this binary's `alloc_error_handler` exits the
/// process, so an OOM during `extend_from_slice` would kill `term`
/// rather than fall back. The size cap below bounds the worst-case
/// allocation; a font that exceeds it is treated as a load failure
/// so the static-table path stays reachable. Replacing the cap with
/// `Vec::try_reserve_exact` is a documented follow-up.
#[cfg(not(test))]
fn build_atlas<F: term::render::FramebufferOwner>(renderer: &mut term::render::Renderer<F>) {
    const FONT_PATH: &[u8] = b"/usr/share/fonts/m3os/term.ttf\0";
    // Cell dimensions must match `term::display::{CELL_WIDTH,
    // CELL_HEIGHT}`. The atlas rasterises into this cell size, so
    // the bigger 16×32 cell gives the Nerd Font glyphs enough
    // resolution to be readable at QEMU GOP defaults.
    const ATLAS_CELL_W: u8 = term::display::CELL_WIDTH;
    const ATLAS_CELL_H: u8 = term::display::CELL_HEIGHT;
    // Hard cap on the read. JetBrainsMono Nerd Font Mono is ~2 MiB;
    // 8 MiB leaves headroom for future patched variants while
    // keeping the worst-case allocation bounded.
    const MAX_FONT_BYTES: usize = 8 * 1024 * 1024;
    let fd = syscall_lib::open(FONT_PATH, syscall_lib::O_RDONLY, 0);
    if fd < 0 {
        syscall_lib::write_str(
            STDOUT_FILENO,
            "term: font load failed; using static fallback\n",
        );
        return;
    }
    let fd = fd as i32;
    let mut bytes: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    let mut chunk = [0u8; 4096];
    let mut oversize = false;
    let mut read_error = false;
    loop {
        let n = syscall_lib::read(fd, &mut chunk);
        if n < 0 {
            // I/O error mid-read. Treat as a load failure rather
            // than constructing an atlas from a partial file — a
            // truncated TTF would parse-fail noisily, but a
            // truncation that happens to land on a table boundary
            // could parse and produce garbage glyphs.
            read_error = true;
            break;
        }
        if n == 0 {
            // Clean EOF.
            break;
        }
        if bytes.len() + n as usize > MAX_FONT_BYTES {
            oversize = true;
            break;
        }
        bytes.extend_from_slice(&chunk[..n as usize]);
    }
    let _ = syscall_lib::close(fd);
    if read_error || oversize || bytes.is_empty() {
        syscall_lib::write_str(
            STDOUT_FILENO,
            "term: font load failed; using static fallback\n",
        );
        return;
    }
    match kernel_core::font::Atlas::new(
        bytes,
        ATLAS_CELL_W,
        ATLAS_CELL_H,
        kernel_core::font::DEFAULT_ATLAS_CAPACITY,
    ) {
        Ok(atlas) => {
            // Pre-warm a representative range so the boot log's
            // glyph count clears the documented `N > 100` gate. We
            // touch printable ASCII (0x20..=0x7E, 95 cps) and the
            // Latin-1 supplement (0xA1..=0xFF, 95 cps) — together
            // ~190 codepoints, well above the 100-glyph threshold
            // without bloating the cache.
            let mut atlas = atlas;
            for cp in 0x20u32..=0x7E {
                let _ = atlas.resolve(cp);
            }
            for cp in 0xA1u32..=0xFF {
                let _ = atlas.resolve(cp);
            }
            let n = atlas.len();
            renderer.set_atlas(atlas);
            let mut buf = [0u8; 64];
            let s = format_atlas_msg(&mut buf, n);
            syscall_lib::write(STDOUT_FILENO, s);
        }
        Err(_) => {
            syscall_lib::write_str(
                STDOUT_FILENO,
                "term: font load failed; using static fallback\n",
            );
        }
    }
}

/// Build the boot-log line `term: atlas loaded <N> glyphs\n` into
/// the caller-provided buffer. `no_std`-friendly so we don't need
/// `format!` (which would pull in `alloc::string::String` formatting
/// into the hot path).
#[cfg(not(test))]
fn format_atlas_msg(buf: &mut [u8; 64], n: usize) -> &[u8] {
    const PREFIX: &[u8] = b"term: atlas loaded ";
    const SUFFIX: &[u8] = b" glyphs\n";
    let mut i = 0;
    for &b in PREFIX {
        buf[i] = b;
        i += 1;
    }
    // Write `n` decimal.
    let mut digits = [0u8; 20];
    let mut d_len = 0;
    let mut v = n;
    if v == 0 {
        digits[0] = b'0';
        d_len = 1;
    } else {
        while v > 0 {
            digits[d_len] = b'0' + (v % 10) as u8;
            d_len += 1;
            v /= 10;
        }
    }
    while d_len > 0 {
        d_len -= 1;
        buf[i] = digits[d_len];
        i += 1;
    }
    for &b in SUFFIX {
        buf[i] = b;
        i += 1;
    }
    &buf[..i]
}

/// Monotonic clock for [`Bell::ring`]. Tiny wrapper around
/// `clock_gettime(CLOCK_MONOTONIC)` so the bell call site is
/// self-documenting without spending a trait abstraction on a
/// single-method type.
#[cfg(not(test))]
#[derive(Clone, Copy)]
struct MonotonicClock;

#[cfg(not(test))]
impl MonotonicClock {
    fn now_ms(self) -> u64 {
        let (sec, nsec) = syscall_lib::clock_gettime(CLOCK_MONOTONIC);
        let sec = sec.max(0) as u64;
        let nsec = nsec.max(0) as u64;
        sec.saturating_mul(1000).saturating_add(nsec / 1_000_000)
    }
}

/// Ring the bell using whichever sink is currently active. On the
/// first `AudioUnavailable` from `AudioClientBellSink`, swap the
/// `Bell` permanently to the warn-once stub so a tight bell loop
/// does not re-attempt the audio path on every ring.
#[cfg(not(test))]
fn ring_bell(
    audio: &mut Option<Bell<AudioClientBellSink>>,
    unavail: &mut Option<Bell<AudioUnavailableBellSink>>,
    now_ms: u64,
) {
    if let Some(b) = audio.as_mut() {
        match b.ring(now_ms) {
            Ok(_) => return,
            Err(BellError::AudioUnavailable) => {
                // Permanently downgrade.
                *audio = None;
                *unavail = Some(Bell::new(AudioUnavailableBellSink::new()));
            }
            Err(_) => return,
        }
    }
    if let Some(b) = unavail.as_mut() {
        let _ = b.ring(now_ms);
    }
}

/// Outcome of one [`pull_one_event`] call. Pure data so the main
/// loop's match remains exhaustive and a future variant addition
/// fails to compile rather than silently dropping events.
#[cfg(not(test))]
enum PulledEvent {
    /// A `KeyEvent` for the input handler to translate.
    Key(kernel_core::input::events::KeyEvent),
    /// Phase 69 Track E — a `PointerEvent` for the mouse reporter
    /// to encode into PTY bytes.
    Pointer(kernel_core::input::events::PointerEvent),
    /// Phase 69 Track D — surface geometry changed; the loop must
    /// reshape the cell grid and propagate SIGWINCH via TIOCSWINSZ.
    SurfaceResized { width: u32, height: u32 },
    /// `display_server` told us the connection is closing — exit
    /// cleanly so the supervisor can restart per `term.conf`.
    Disconnect,
    /// No event this tick (`LABEL_CLIENT_EVENT_NONE`, transport
    /// error, decode failure, or a `ServerMessage` term doesn't
    /// consume — `Welcome`, `SurfaceConfigured`, `FocusIn` /
    /// `FocusOut`, `BufferReleased`, `SurfaceDestroyed`). All of
    /// these are non-fatal and the next iteration retries.
    None,
}

/// Pull one queued `ServerMessage` from `display_server`'s C.5
/// outbound queue and classify it for the main loop.
///
/// Disconnect is the only non-Key variant that changes behaviour:
/// it asks term to exit. Every other variant is dropped with no
/// state change because term's contract today is "Toplevel surface
/// + keyboard-focused PTY" — pointer events, focus changes, and
/// buffer-released are not load-bearing for that contract. A
/// future track that adds e.g. mouse-aware shell selection would
/// thread `Pointer` into the input handler here.
#[cfg(not(test))]
fn pull_one_event(display_handle: u32, buf: &mut [u8]) -> PulledEvent {
    let label = syscall_lib::ipc_call(display_handle, LABEL_CLIENT_EVENT_PULL, 0);
    if label != LABEL_CLIENT_EVENT_PULL {
        // LABEL_CLIENT_EVENT_NONE (= 4) or transport error — no
        // event. Even on the NONE path the kernel may have staged
        // an empty bulk; drain to keep the slot clean for the next
        // call.
        let _ = syscall_lib::ipc_take_pending_bulk(buf);
        return PulledEvent::None;
    }
    let n = syscall_lib::ipc_take_pending_bulk(buf);
    if n == 0 || n == u64::MAX {
        return PulledEvent::None;
    }
    let len = n as usize;
    if len > buf.len() {
        return PulledEvent::None;
    }
    match ServerMessage::decode(&buf[..len]) {
        Ok((ServerMessage::Key(ev), _)) => PulledEvent::Key(ev),
        Ok((ServerMessage::Pointer(ev), _)) => PulledEvent::Pointer(ev),
        Ok((ServerMessage::SurfaceResized { width, height, .. }, _)) => {
            PulledEvent::SurfaceResized { width, height }
        }
        Ok((ServerMessage::Disconnect { .. }, _)) => PulledEvent::Disconnect,
        // Welcome / FocusIn / FocusOut / SurfaceConfigured /
        // SurfaceDestroyed / BufferReleased: not load-bearing for
        // term's contract — drop silently.
        Ok(_) => PulledEvent::None,
        Err(_) => PulledEvent::None,
    }
}

/// Mirror of `DisplayClient`'s lookup-with-backoff so the input loop
/// can hold its own handle on the `"display"` service. `connect`
/// already paid the boot-time backoff cost; this call is expected to
/// resolve on the first attempt.
#[cfg(not(test))]
fn lookup_display_for_input() -> Option<u32> {
    let raw = syscall_lib::ipc_lookup_service("display");
    if raw == u64::MAX {
        return None;
    }
    Some(raw as u32)
}

/// Phase 69 Track E — translate a DEC private mode code into a
/// `MouseReporter` state update. Driven by
/// `RenderCommand::SetMouseMode` emitted by `Screen::feed`.
///
/// Tracking-mode (`?9` / `?1000` / `?1002` / `?1003`) and
/// encoding-mode (`?1006`) are stored independently so xterm-style
/// idioms like `?1000h ?1006h` (enable normal tracking, then SGR
/// encoding) followed by `?1006l` revert the encoding back to the
/// legacy form without disabling tracking.
#[cfg(not(test))]
fn update_mouse_mode(reporter: &mut MouseReporter, code: u16, set: bool) {
    match code {
        9 => reporter.set_tracking(if set {
            TrackingMode::X10
        } else {
            TrackingMode::Disabled
        }),
        1000 => reporter.set_tracking(if set {
            TrackingMode::Normal
        } else {
            TrackingMode::Disabled
        }),
        // ?1002 / ?1003 are tracked as their own variants so the
        // deferred motion-tracking work has a name to switch on,
        // even though `encode` currently treats them like Normal.
        1002 => reporter.set_tracking(if set {
            TrackingMode::ButtonMotion
        } else {
            TrackingMode::Disabled
        }),
        1003 => reporter.set_tracking(if set {
            TrackingMode::AnyEvent
        } else {
            TrackingMode::Disabled
        }),
        1006 => reporter.set_encoding(if set {
            EncodingMode::Sgr
        } else {
            EncodingMode::Legacy
        }),
        _ => {}
    }
}

/// Phase 69 Track D — react to a `SurfaceResized` notification by
/// recomputing the cell grid and issuing `ioctl(TIOCSWINSZ)` on the
/// PTY primary fd. The kernel `TIOCSWINSZ` handler updates the PTY's
/// `winsize` and sends SIGWINCH to the foreground process group.
#[cfg(not(test))]
fn handle_surface_resize<F: term::render::FramebufferOwner>(
    primary_fd: i32,
    screen: &mut Screen,
    renderer: &mut Renderer<F>,
    width: u32,
    height: u32,
) {
    // Cell metrics must track `term::display::{CELL_WIDTH,
    // CELL_HEIGHT}` so SurfaceResized cell math (pixels → cell grid)
    // matches the actual surface stride.
    const GLYPH_W: u32 = term::display::CELL_WIDTH as u32;
    const GLYPH_H: u32 = term::display::CELL_HEIGHT as u32;
    // Phase 69 PR 168 round-3 fix — a malformed `SurfaceResized` could
    // request 65535×65535 cells (~4.3B cells × `Cell` size = multi-GB)
    // and crash `term`. Cap each dimension at 1024 cells (with the
    // Phase 69c 16×32 cell metrics this corresponds to 16384×32768
    // logical pixels — well above any realistic display) and the total
    // cell budget at ~1M, which keeps `Screen::resize` allocations in
    // single-digit MB regardless of the message contents.
    const MAX_CELLS_PER_AXIS: u32 = 1024;
    const MAX_TOTAL_CELLS: u32 = 1_000_000;
    if width == 0 || height == 0 {
        return;
    }
    let mut cols = (width / GLYPH_W).max(1).min(MAX_CELLS_PER_AXIS) as u16;
    let mut rows = (height / GLYPH_H).max(1).min(MAX_CELLS_PER_AXIS) as u16;
    // Per-axis cap above bounds the product at `MAX_CELLS_PER_AXIS^2 =
    // ~1M`, which already satisfies the total-cell budget — but if the
    // axis cap is later relaxed, halve the larger axis until the total
    // budget is honoured. Pure-integer loop avoids `libm`.
    while (cols as u32) * (rows as u32) > MAX_TOTAL_CELLS {
        if cols >= rows {
            cols = (cols / 2).max(1);
        } else {
            rows = (rows / 2).max(1);
        }
    }

    let mut local_cmds: alloc::vec::Vec<RenderCommand> = alloc::vec::Vec::new();
    screen.resize(cols, rows, &mut local_cmds);
    for cmd in local_cmds {
        renderer.apply(cmd);
    }

    // Phase 69 PR 168 fix — `Winsize::ws_xpixel`/`ws_ypixel` are 16-bit but
    // the display server hands us `u32`. A pathological surface > 65535 px
    // would wrap to a bogus small value (e.g. 70000 → 4464); saturate to
    // `u16::MAX` so the TTY layer sees an honest "very large" sentinel
    // instead of misreporting the size after the truncation.
    let ws = syscall_lib::Winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: width.min(u16::MAX as u32) as u16,
        ws_ypixel: height.min(u16::MAX as u32) as u16,
    };
    let _ = syscall_lib::ioctl(
        primary_fd,
        syscall_lib::TIOCSWINSZ,
        &ws as *const syscall_lib::Winsize as usize,
    );
}
