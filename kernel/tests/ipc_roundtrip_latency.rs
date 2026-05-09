#![no_std]
#![no_main]
#![feature(custom_test_frameworks)]
#![test_runner(test_runner)]
#![reexport_test_harness_main = "test_main"]

//! IPC call/reply roundtrip latency regression test.
//!
//! Reproduces the latency surface that `vfs_server: slow req` warnings
//! observe at the userspace layer (50+ ms VFS reads), but in pure kernel
//! space so it can run unattended in `cargo xtask test`.
//!
//! ## What it measures
//!
//! Two pinned kernel tasks: a server on core 1 looping over
//! `recv_msg(REQ_EP) -> send(RESP_EP)`, and a client on core 0 looping over
//! `send(REQ_EP) -> recv_msg(RESP_EP)` for `ROUNDTRIPS` iterations. Each
//! roundtrip is timed with `_rdtsc` so resolution is sub-tick.
//!
//! ## Acceptance
//!
//! On a healthy kernel an unloaded cross-core IPC roundtrip is ~tens of
//! microseconds (single-digit ticks at most). The assertion budget is
//! deliberately loose (`MAX_AVG_US`, `MAX_P99_US`, `MAX_WORST_US`) so the
//! test only fires on a regression of two-orders-of-magnitude scale —
//! the kind of regression that produces `slow req` warnings.

extern crate alloc;

use alloc::vec::Vec;
use bootloader_api::{BootInfo, BootloaderConfig, config::Mapping, entry_point};
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut config = BootloaderConfig::new_default();
    config.mappings.physical_memory = Some(Mapping::Dynamic);
    config
};

entry_point!(ipc_roundtrip_latency_test, config = &BOOTLOADER_CONFIG);

fn ipc_roundtrip_latency_test(boot_info: &'static mut BootInfo) -> ! {
    kernel::test_prelude::init_minimal_smp(boot_info);
    kernel::test_prelude::boot_aps_if_available();
    kernel::task::spawn(test_runner_task, "ipc-rt-runner");
    kernel::test_prelude::spawn_idle();
    kernel::task::run()
}

fn test_runner_task() -> ! {
    test_main();
    kernel::test_prelude::qemu_exit_success()
}

trait Testable {
    fn run(&self);
}
impl<T: Fn()> Testable for T {
    fn run(&self) {
        self();
    }
}
fn test_runner(tests: &[&dyn Testable]) {
    for t in tests {
        t.run();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    if let Some(loc) = info.location() {
        kernel::serial_println!(
            "[ipc-rt-test] PANIC at {}:{}: {}",
            loc.file(),
            loc.line(),
            info.message()
        );
    } else {
        kernel::serial_println!("[ipc-rt-test] PANIC: {}", info.message());
    }
    kernel::test_prelude::qemu_exit_failure()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const ROUNDTRIPS: usize = 200;

/// Tag carried by every request/response so the server can sanity-check it
/// is replying to the call it just received.
const TAG: u64 = 0x5252_5252_5252_5252;

/// Wall-clock ceiling for the whole test (server + client). Avoids hanging CI
/// if scheduler gets stuck or one side never wakes.
const TEST_DEADLINE_TICKS: u64 = 30_000; // 30 s with 1 ms ticks.

// Loose acceptance budgets. On a healthy kernel a 200-iteration cross-core
// roundtrip series finishes in under 100 ms total (avg < 500 us). These
// thresholds are intentionally generous — they exist to flag a Phase 61-class
// regression where avg roundtrip jumps to multi-millisecond territory and
// p99/worst climb past 50 ms.
const MAX_AVG_US: u64 = 5_000; // 5 ms
const MAX_P99_US: u64 = 25_000; // 25 ms
const MAX_WORST_US: u64 = 100_000; // 100 ms

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

static REQ_EP: AtomicU64 = AtomicU64::new(u64::MAX);
static RESP_EP: AtomicU64 = AtomicU64::new(u64::MAX);

static SERVER_READY: AtomicBool = AtomicBool::new(false);
static SERVER_TASK_ID: AtomicU64 = AtomicU64::new(0);
static CLIENT_TASK_ID: AtomicU64 = AtomicU64::new(0);
static CLIENT_DONE: AtomicBool = AtomicBool::new(false);
static SERVER_BAD_MSG: AtomicBool = AtomicBool::new(false);

/// Number of completed iterations the server has serviced. Read by the
/// runner for diagnostic output if the test deadlines.
static SERVER_ITER: AtomicU64 = AtomicU64::new(0);

/// TSC samples written by the client, one per roundtrip. Sized exactly to
/// `ROUNDTRIPS`. Read by the runner after `CLIENT_DONE`.
static mut LATENCIES_TSC: [u64; ROUNDTRIPS] = [0u64; ROUNDTRIPS];

// ---------------------------------------------------------------------------
// Server task
// ---------------------------------------------------------------------------

fn server_task() -> ! {
    use kernel_core::ipc::message::Message;
    use kernel_core::types::EndpointId;

    let req_ep = EndpointId(REQ_EP.load(Ordering::Acquire) as u8);
    let resp_ep = EndpointId(RESP_EP.load(Ordering::Acquire) as u8);
    let task_id = kernel::task::scheduler::current_task_id().expect("server task id");
    SERVER_TASK_ID.store(task_id.0, Ordering::Release);
    SERVER_READY.store(true, Ordering::Release);

    loop {
        let msg = kernel::ipc::endpoint::recv_msg(task_id, req_ep);
        if msg.label != TAG {
            SERVER_BAD_MSG.store(true, Ordering::Release);
        }
        let reply = Message::new(TAG);
        let _ = kernel::ipc::endpoint::send(task_id, resp_ep, reply);
        SERVER_ITER.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Client task
// ---------------------------------------------------------------------------

fn client_task() -> ! {
    use kernel_core::ipc::message::Message;
    use kernel_core::types::EndpointId;

    let req_ep = EndpointId(REQ_EP.load(Ordering::Acquire) as u8);
    let resp_ep = EndpointId(RESP_EP.load(Ordering::Acquire) as u8);
    let task_id = kernel::task::scheduler::current_task_id().expect("client task id");
    CLIENT_TASK_ID.store(task_id.0, Ordering::Release);

    // Wait until the server is registered + parked in recv_msg. The server
    // sets SERVER_READY before its first recv_msg call; that is the earliest
    // safe moment to start sending without missing the first iteration.
    while !SERVER_READY.load(Ordering::Acquire) {
        kernel::task::yield_now();
    }

    for i in 0..ROUNDTRIPS {
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        let _ = kernel::ipc::endpoint::send(task_id, req_ep, Message::new(TAG));
        let _ = kernel::ipc::endpoint::recv_msg(task_id, resp_ep);
        let end = unsafe { core::arch::x86_64::_rdtsc() };
        // SAFETY: client is the sole writer of LATENCIES_TSC and writes each
        // index exactly once, in order. No other task reads it until
        // CLIENT_DONE is set Release below.
        unsafe { LATENCIES_TSC[i] = end.wrapping_sub(start) };
    }
    CLIENT_DONE.store(true, Ordering::Release);
    loop {
        kernel::task::yield_now();
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test_case]
fn cross_core_ipc_roundtrip_latency_within_budget() {
    let cores = kernel::smp::core_count() as usize;
    assert!(
        cores >= 2,
        "ipc roundtrip latency test requires at least 2 cores; got {cores}"
    );

    let req = kernel::ipc::endpoint::ENDPOINTS.lock().create();
    let resp = kernel::ipc::endpoint::ENDPOINTS.lock().create();
    REQ_EP.store(req.0 as u64, Ordering::Release);
    RESP_EP.store(resp.0 as u64, Ordering::Release);

    kernel::task::scheduler::spawn_on_core(server_task, "ipc-rt-server", 1);
    kernel::task::scheduler::spawn_on_core(client_task, "ipc-rt-client", 0);

    let deadline =
        kernel::arch::x86_64::interrupts::tick_count().saturating_add(TEST_DEADLINE_TICKS);
    while !CLIENT_DONE.load(Ordering::Acquire) {
        if kernel::arch::x86_64::interrupts::tick_count() >= deadline {
            panic!(
                "ipc roundtrip test deadlined at {} iterations / {} server iters",
                ROUNDTRIPS,
                SERVER_ITER.load(Ordering::Relaxed)
            );
        }
        kernel::task::yield_now();
    }

    assert!(
        !SERVER_BAD_MSG.load(Ordering::Acquire),
        "server saw a request with the wrong tag"
    );

    // Pull samples and convert to microseconds. `tsc_per_ms` was calibrated at
    // boot in `apic::init_timer`; if it is zero, fail loud rather than
    // dividing by zero.
    let tsc_per_ms = kernel::arch::x86_64::apic::tsc_per_ms();
    assert!(tsc_per_ms > 0, "tsc_per_ms not calibrated");
    let tsc_per_us = tsc_per_ms / 1_000;
    assert!(
        tsc_per_us > 0,
        "tsc_per_us underflowed (tsc_per_ms = {tsc_per_ms})"
    );

    // SAFETY: client is no longer writing (CLIENT_DONE was set Release before
    // we observed it Acquire above). No other task touches LATENCIES_TSC.
    let mut samples_us: Vec<u64> = (0..ROUNDTRIPS)
        .map(|i| unsafe { LATENCIES_TSC[i] } / tsc_per_us)
        .collect();
    samples_us.sort_unstable();

    let total_us: u64 = samples_us.iter().sum();
    let avg_us = total_us / ROUNDTRIPS as u64;
    let p50_us = samples_us[ROUNDTRIPS / 2];
    let p99_us = samples_us[(ROUNDTRIPS * 99) / 100];
    let worst_us = *samples_us.last().unwrap();
    let best_us = *samples_us.first().unwrap();

    kernel::serial_println!(
        "[ipc-rt-test] N={} best={}us p50={}us avg={}us p99={}us worst={}us total={}us",
        ROUNDTRIPS,
        best_us,
        p50_us,
        avg_us,
        p99_us,
        worst_us,
        total_us,
    );

    assert!(
        avg_us <= MAX_AVG_US,
        "avg roundtrip {avg_us}us exceeds budget {MAX_AVG_US}us",
    );
    assert!(
        p99_us <= MAX_P99_US,
        "p99 roundtrip {p99_us}us exceeds budget {MAX_P99_US}us",
    );
    assert!(
        worst_us <= MAX_WORST_US,
        "worst roundtrip {worst_us}us exceeds budget {MAX_WORST_US}us",
    );
}
