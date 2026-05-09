//! IDT handlers and the kernel ISR contract.
//!
//! Every `extern "x86-interrupt" fn …_handler` in this file — plus the
//! per-device handlers registered at runtime via
//! `register_device_irq` (e.g. `virtio_net_irq_handler`,
//! `virtio_blk_irq_handler`) — runs in interrupt context with
//! interrupts automatically disabled on the current CPU. They must obey
//! the following invariants:
//!
//! 1. **No allocation.** The global allocator may be held by the
//!    interrupted task.
//! 2. **No blocking.** No syscall dispatch, no IPC send / recv / reply,
//!    no `switch_context`, no userspace return. The handler runs to
//!    completion, acks the device, and returns.
//! 3. **No plain `spin::Mutex` acquisition on a lock that task-context
//!    callers hold with interrupts enabled.** A same-core ISR landing
//!    on such a lock spins forever (the interrupted holder cannot
//!    release while the ISR runs). A shared lock reachable from an
//!    ISR must be one of:
//!    - an `IrqSafeMutex` (canonical impl: `kernel/src/task/scheduler.rs`)
//!      that masks interrupts for the duration of its critical section,
//!      OR
//!    - a `spin::Mutex` whose every task-context acquisition runs
//!      inside `interrupts::without_interrupts(…)` (the
//!      `virtio_net::DRIVER`, `virtio_blk::DRIVER`, and
//!      `RAW_INPUT_ROUTER` patterns), OR
//!    - only accessed from ISR context (no task-context holder exists).
//! 4. **`scheduler::wake_task` is ISR-safe by design.** `SCHEDULER` is
//!    an `IrqSafeMutex<Scheduler>` and `enqueue_to_core` wraps its
//!    per-core `run_queue.lock()` in `without_interrupts`. Handlers
//!    may call it freely. Any new lock added along the wake callpath
//!    must extend the IRQ-safety audit.
//! 5. **EOI last.** Either `super::apic::lapic_eoi()` (APIC mode) or
//!    `PICS.lock().notify_end_of_interrupt(…)` (PIC mode). `PICS` is
//!    only acquired at boot (before interrupts are enabled) and from
//!    ISR context, so it is trivially ISR-safe.
//!
//! The 2026-04-21 post-mortem
//! (`docs/post-mortems/2026-04-21-scheduler-lock-isr-deadlock.md`)
//! formalised rule 3 after a pair of virtio IRQ handlers called
//! `wake_task` on top of a plain `spin::Mutex<Scheduler>` and
//! deterministically deadlocked same-core task-context holders. Rule
//! 3 is the rule that class of bug violated. Every handler below
//! relies on at least one of the three lock disciplines it enumerates.

use core::arch::global_asm;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use kernel_core::input::{ScancodeRouter, ScancodeSink};
use spin::{Lazy, Mutex};
use x86_64::VirtAddr;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};

use crate::panic_diag;
use crate::serial::_panic_print;

use super::gdt;
use super::preempt_trap_frame::{PreemptTrapFrameKernel, PreemptTrapFrameUser};

// ---------------------------------------------------------------------------
// APIC / PIC mode flag
// ---------------------------------------------------------------------------

/// When `true`, interrupt handlers send EOI to the Local APIC instead of the
/// legacy 8259 PIC. Set by `apic::init()` after the APIC subsystem is fully
/// programmed.
pub static USING_APIC: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Phase 57b D.3 — IRQ-return-to-ring-3 preempt_count assertion
// ---------------------------------------------------------------------------

/// Phase 57b D.3 — IRQ-return-to-ring-3 wrapper around
/// [`crate::task::scheduler::assert_preempt_count_zero_at_user_return`].
///
/// Called at the end of every `extern "x86-interrupt"` handler that may
/// have interrupted ring 3 — the body returns via `iretq` to user mode
/// when this branch is taken, and we want the same `preempt_count == 0`
/// invariant the syscall-return path enforces.
///
/// The assertion is gated on `stack_frame.code_segment.rpl() ==
/// PrivilegeLevel::Ring3` because under Phase 57d kernel-mode will hold
/// `preempt_count > 0` while inside spinlock-protected critical
/// sections — an IPI / IRQ that interrupted such a section would
/// (correctly) see a non-zero count and must not panic.  The "return to
/// ring 3" check distinguishes the two cases.
///
/// In Phase 57b nothing raises `preempt_count` yet (Tracks F and G are
/// future waves), so even unconditionally checking would pass — the
/// gate is in place from day one to keep the assertion future-correct
/// once F.1 wires `IrqSafeMutex::lock` into `preempt_disable`.
///
/// The check itself is a `debug_assert!` inside the helper; release
/// builds compile out the entire body via `cfg(debug_assertions)`.
#[inline]
fn assert_preempt_count_zero_on_return_to_user(stack_frame: &InterruptStackFrame) {
    // Phase 57e Bug #9 — clamp preempt_count to 0 at user-return in release
    // builds, panic on non-zero in debug builds.  Helper handles both modes;
    // gate ring-3 only.
    if stack_frame.code_segment.rpl() == x86_64::PrivilegeLevel::Ring3 {
        crate::task::scheduler::assert_preempt_count_zero_at_user_return();
    }
    // Phase 57d E.3: consume deferred reschedule at IRQ-return to user mode.
    #[cfg(feature = "preempt-voluntary")]
    if stack_frame.code_segment.rpl() == x86_64::PrivilegeLevel::Ring3
        && let Some(pc) = crate::smp::try_per_core()
        && pc
            .preempt_resched_pending
            .swap(false, core::sync::atomic::Ordering::AcqRel)
    {
        crate::task::signal_reschedule();
    }
}

// ---------------------------------------------------------------------------
// Two-phase fault kill path (T001)
// ---------------------------------------------------------------------------

/// PID of the process that triggered a ring-3 exception.
///
/// Written by the exception handler (in interrupt context, interrupts
/// disabled) and read by `fault_kill_trampoline` (in task context, outside
/// interrupt). Single-CPU: no concurrent writers.
static FAULT_KILL_PID: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// CoW fault resolution (P17-T031, T032, T033)
// ---------------------------------------------------------------------------

fn bump_current_addr_space_generation() {
    if let Some(addr_space) = crate::process::current_addr_space() {
        unsafe { addr_space.as_ref() }.bump_generation();
    }
}

/// Ring-0 trampoline that runs *outside* interrupt context.
///
/// The exception handler redirects IRET here so that locking and
/// context-switching (which are forbidden inside an ISR) can happen safely.
fn fault_kill_trampoline() -> ! {
    // Disable interrupts immediately — IRET restored user RFLAGS which may
    // have IF set, and we must not take interrupts before acquiring locks.
    x86_64::instructions::interrupts::disable();
    let pid = FAULT_KILL_PID.load(Ordering::Relaxed);
    log::warn!("[fault_kill] trampoline running for pid {}", pid);
    // Close all open FDs so pipe ref-counts reach 0 and EOF propagates.
    crate::process::close_all_fds_for(pid);
    // Deactivate this core's tracked AddressSpace *before* marking Zombie.
    // Once Zombie, another core can reap() and drop the last Arc, turning
    // our raw current_addrspace pointer into a dangling reference.
    if crate::smp::is_per_core_ready() {
        let pc = crate::smp::per_core();
        let old_as_ptr = pc.current_addrspace;
        if !old_as_ptr.is_null() {
            let core_id = pc.core_id;
            // SAFETY: Arc<AddressSpace> is still alive — process is not
            // yet Zombie so reap cannot have been called.
            unsafe { &*old_as_ptr }.deactivate_on_core(core_id);
            let pc_mut = pc as *const crate::smp::PerCoreData as *mut crate::smp::PerCoreData;
            unsafe { (*pc_mut).current_addrspace = core::ptr::null() };
        }
    }
    // Mark the process zombie with SIGSEGV exit code.
    {
        let mut table = crate::process::PROCESS_TABLE.lock();
        if let Some(proc) = table.find_mut(pid) {
            proc.state = crate::process::ProcessState::Zombie;
            proc.exit_code = Some(-11);
        }
    }
    // Deliver SIGCHLD to parent so waitpid unblocks.
    crate::process::send_sigchld_to_parent(pid);
    // Read the dying process's CR3 before we switch away from it.
    let cr3_phys = {
        let table = crate::process::PROCESS_TABLE.lock();
        table
            .find(pid)
            .and_then(|p| p.addr_space.as_ref().map(|a| a.pml4_phys()))
    };
    // Restore kernel page table before yielding — same reason as sys_exit.
    crate::mm::restore_kernel_cr3();
    // Free the process's user-space page table frames.
    if let Some(phys) = cr3_phys {
        crate::mm::free_process_page_table(phys.as_u64());
    }
    // Permanently remove the kernel task — the process is dead.
    crate::task::mark_current_dead();
}

/// Phase 57b post-review fix — invoke [`crate::smp::tlb::tlb_shootdown_range`]
/// safely from page-fault exception context.
///
/// CPU exception handlers (page fault, GP, etc.) run with `IF=0` set by
/// hardware on entry.  If two cores fault concurrently and both reach the
/// shootdown path, one takes `SHOOTDOWN_LOCK` and broadcasts an
/// `IPI_TLB_SHOOTDOWN` to the other, then spins waiting for the ack — but
/// the other core is contending for `SHOOTDOWN_LOCK` with `IF=0` and
/// cannot service the IPI.  Both cores deadlock.
///
/// `tlb_shootdown_range` itself uses the **preempt-only** discipline (no
/// IF masking), so it is safe to enable IF here for the duration of the
/// shootdown.  The page-fault handler has already finished its
/// synchronous page-table mutation under
/// [`crate::mm::AddressSpace::lock_page_tables`]; `IF=1` during the
/// shootdown does not race against that critical section because the
/// guard has been dropped before this helper runs.
///
/// On `iretq` the CPU pops the saved RFLAGS so the user's original IF
/// state is restored regardless of the IF bit at the moment of the
/// `iretq`.
fn tlb_shootdown_range_from_fault_context(
    addr_space: &crate::mm::AddressSpace,
    start: u64,
    end: u64,
) {
    let saved_if = x86_64::instructions::interrupts::are_enabled();
    if !saved_if {
        x86_64::instructions::interrupts::enable();
    }
    crate::smp::tlb::tlb_shootdown_range(addr_space, start, end);
    if !saved_if {
        x86_64::instructions::interrupts::disable();
    }
}

/// Resolve a copy-on-write page fault at `vaddr`.
///
/// Reads the current PTE, allocates a fresh frame, copies the page contents,
/// maps the new frame as writable, and decrements the old frame's refcount.
///
/// Returns `true` on success, `false` if the faulting mapping is no longer a
/// CoW page or if frame allocation fails (OOM).
pub fn resolve_cow_fault(vaddr: u64) -> bool {
    use x86_64::registers::control::Cr3;
    use x86_64::structures::paging::{PageTable, PageTableFlags};

    let phys_off = crate::mm::phys_offset();
    let phys_offset = VirtAddr::new(phys_off);
    let addr_space = crate::process::current_addr_space();
    let mut old_phys_to_free = None;
    {
        let _page_table_guard =
            addr_space.map(|addr_space| unsafe { addr_space.as_ref() }.lock_page_tables());

        let (cr3_frame, _) = Cr3::read();
        let pml4_phys = cr3_frame.start_address().as_u64();

        // Walk the page table to find the PTE for the faulting address.
        let p4_idx = ((vaddr >> 39) & 0x1FF) as usize;
        let p3_idx = ((vaddr >> 30) & 0x1FF) as usize;
        let p2_idx = ((vaddr >> 21) & 0x1FF) as usize;
        let p1_idx = ((vaddr >> 12) & 0x1FF) as usize;

        unsafe {
            let pml4: &PageTable = &*(phys_offset + pml4_phys).as_ptr::<PageTable>();
            let p4e = &pml4[p4_idx];
            if !p4e.flags().contains(PageTableFlags::PRESENT) {
                return false;
            }

            let pdpt: &PageTable = &*(phys_offset + p4e.addr().as_u64()).as_ptr::<PageTable>();
            let p3e = &pdpt[p3_idx];
            if !p3e.flags().contains(PageTableFlags::PRESENT) {
                return false;
            }

            let pd: &PageTable = &*(phys_offset + p3e.addr().as_u64()).as_ptr::<PageTable>();
            let p2e = &pd[p2_idx];
            if !p2e.flags().contains(PageTableFlags::PRESENT) {
                return false;
            }

            let pt: &mut PageTable =
                &mut *(phys_offset + p2e.addr().as_u64()).as_mut_ptr::<PageTable>();
            let pte = &mut pt[p1_idx];
            let pte_flags = pte.flags();
            if !pte_flags.contains(PageTableFlags::PRESENT)
                || !pte_flags.contains(PageTableFlags::BIT_9)
                || pte_flags.contains(PageTableFlags::WRITABLE)
            {
                return false;
            }

            let old_phys = pte.addr().as_u64();
            let old_refcount = crate::mm::frame_allocator::refcount_get(old_phys);

            if old_refcount <= 1 {
                // P17-T033: fast path — sole owner, just remap as writable
                // and clear the CoW marker bit.
                let flags = (pte.flags() | PageTableFlags::WRITABLE) & !PageTableFlags::BIT_9;
                pte.set_addr(pte.addr(), flags);
            } else {
                // Allocate a fresh frame. If out of memory, return false so the
                // page fault handler falls through to the kill path instead of
                // panicking the kernel (user-triggerable OOM must not be a DoS).
                let new_frame = match crate::mm::frame_allocator::allocate_frame() {
                    Some(f) => f,
                    None => return false,
                };
                let new_phys = new_frame.start_address().as_u64();

                let src = (phys_off + old_phys) as *const u8;
                let dst = (phys_off + new_phys) as *mut u8;
                core::ptr::copy_nonoverlapping(src, dst, 4096);

                // Map the new frame writable, clear the CoW marker.
                let flags = (pte.flags() | PageTableFlags::WRITABLE) & !PageTableFlags::BIT_9;
                pte.set_addr(new_frame.start_address(), flags);
                old_phys_to_free = Some(old_phys);
            }
        }
    }

    if crate::smp::is_per_core_ready()
        && let Some(addr_space) = addr_space
    {
        tlb_shootdown_range_from_fault_context(unsafe { addr_space.as_ref() }, vaddr, vaddr + 4096);
    } else {
        x86_64::instructions::tlb::flush(VirtAddr::new(vaddr));
    }
    if let Some(old_phys) = old_phys_to_free {
        crate::mm::frame_allocator::free_frame(old_phys);
    }
    bump_current_addr_space_generation();
    true
}

/// Check whether the PTE for `vaddr` has the guard-page marker bit (BIT_10) set.
fn has_guard_marker(vaddr: u64) -> bool {
    use x86_64::registers::control::Cr3;
    use x86_64::structures::paging::{PageTable, PageTableFlags};

    let phys_off = crate::mm::phys_offset();
    let phys_offset_va = VirtAddr::new(phys_off);

    let (cr3_frame, _) = Cr3::read();
    let pml4_phys = cr3_frame.start_address().as_u64();

    let p4_idx = ((vaddr >> 39) & 0x1FF) as usize;
    let p3_idx = ((vaddr >> 30) & 0x1FF) as usize;
    let p2_idx = ((vaddr >> 21) & 0x1FF) as usize;
    let p1_idx = ((vaddr >> 12) & 0x1FF) as usize;

    unsafe {
        let pml4: &PageTable = &*(phys_offset_va + pml4_phys).as_ptr::<PageTable>();
        if !pml4[p4_idx].flags().contains(PageTableFlags::PRESENT) {
            return false;
        }
        let pdpt: &PageTable =
            &*(phys_offset_va + pml4[p4_idx].addr().as_u64()).as_ptr::<PageTable>();
        if !pdpt[p3_idx].flags().contains(PageTableFlags::PRESENT) {
            return false;
        }
        let pd: &PageTable =
            &*(phys_offset_va + pdpt[p3_idx].addr().as_u64()).as_ptr::<PageTable>();
        if !pd[p2_idx].flags().contains(PageTableFlags::PRESENT) {
            return false;
        }
        let pt: &PageTable = &*(phys_offset_va + pd[p2_idx].addr().as_u64()).as_ptr::<PageTable>();
        pt[p1_idx].flags().contains(PageTableFlags::BIT_10)
    }
}

/// Phase 57e diag: walk the page tables for `vaddr` from BOTH the active CR3
/// PML4 and the kernel-original `KERNEL_PML4_PHYS`, printing each level.
///
/// Used from the page-fault handler to localise the slab UAF / spurious
/// PTE-clear residual to one of:
///   - Hyp #3 surviving: the active CR3's PML4[256] points at a different
///     PDPT than KERNEL_PML4 (i.e. kernel-half sharing was lost).
///   - Hyp #4: the walks converge but a leaf PTE / sub-table page is zero.
///   - Hyp #4b: the PT frame itself was overwritten (unmapped from the walk
///     by a parent-entry change rather than a leaf clear).
///
/// Output is best-effort and resilient to garbage page-table entries.
fn dump_pte_walk_diagnostics(vaddr: u64) {
    use x86_64::registers::control::Cr3;
    use x86_64::structures::paging::{PageTable, PageTableFlags};

    let phys_off = crate::mm::phys_offset();
    let phys_off_va = VirtAddr::new(phys_off);
    let p4 = ((vaddr >> 39) & 0x1FF) as usize;
    let p3 = ((vaddr >> 30) & 0x1FF) as usize;
    let p2 = ((vaddr >> 21) & 0x1FF) as usize;
    let p1 = ((vaddr >> 12) & 0x1FF) as usize;

    let (cr3_frame, _) = Cr3::read_raw();
    let active_pml4 = cr3_frame.start_address().as_u64();
    let kernel_pml4 = crate::mm::kernel_pml4_phys();

    _panic_print(format_args!(
        "[pf-diag] vaddr={:#018x} idx=[p4={} p3={} p2={} p1={}] active_cr3={:#x} kernel_pml4={:#x}\n",
        vaddr, p4, p3, p2, p1, active_pml4, kernel_pml4
    ));

    fn walk_one(label: &str, phys_off_va: VirtAddr, pml4_phys: u64, idx: [usize; 4]) {
        let [p4, p3, p2, p1] = idx;
        unsafe {
            let pml4: &PageTable = &*(phys_off_va + pml4_phys).as_ptr::<PageTable>();
            let e4 = &pml4[p4];
            _panic_print(format_args!(
                "[pf-diag] {}: PML4[{}] flags={:?} addr={:#x}\n",
                label,
                p4,
                e4.flags(),
                e4.addr().as_u64()
            ));
            if !e4.flags().contains(PageTableFlags::PRESENT) {
                return;
            }
            let pdpt: &PageTable = &*(phys_off_va + e4.addr().as_u64()).as_ptr::<PageTable>();
            let e3 = &pdpt[p3];
            _panic_print(format_args!(
                "[pf-diag] {}: PDPT[{}] flags={:?} addr={:#x}\n",
                label,
                p3,
                e3.flags(),
                e3.addr().as_u64()
            ));
            if !e3.flags().contains(PageTableFlags::PRESENT)
                || e3.flags().contains(PageTableFlags::HUGE_PAGE)
            {
                return;
            }
            let pd: &PageTable = &*(phys_off_va + e3.addr().as_u64()).as_ptr::<PageTable>();
            let e2 = &pd[p2];
            _panic_print(format_args!(
                "[pf-diag] {}: PD  [{}] flags={:?} addr={:#x}\n",
                label,
                p2,
                e2.flags(),
                e2.addr().as_u64()
            ));
            if !e2.flags().contains(PageTableFlags::PRESENT)
                || e2.flags().contains(PageTableFlags::HUGE_PAGE)
            {
                return;
            }
            let pt: &PageTable = &*(phys_off_va + e2.addr().as_u64()).as_ptr::<PageTable>();
            let e1 = &pt[p1];
            _panic_print(format_args!(
                "[pf-diag] {}: PT  [{}] flags={:?} addr={:#x}\n",
                label,
                p1,
                e1.flags(),
                e1.addr().as_u64()
            ));
        }
    }

    walk_one("active", phys_off_va, active_pml4, [p4, p3, p2, p1]);
    if active_pml4 != kernel_pml4 {
        walk_one("kernel", phys_off_va, kernel_pml4, [p4, p3, p2, p1]);
    } else {
        _panic_print(format_args!(
            "[pf-diag] kernel: (same as active, walk omitted)\n"
        ));
    }
}

/// Public entry point for kernel-context VMA demand paging.
///
/// Revalidates the current VMA metadata while holding the address-space
/// mutation lock so concurrent `munmap` / `mprotect` cannot publish stale
/// permissions across the lock boundary.
pub fn demand_map_vma_page_from_kernel(vaddr: u64, require_write: bool) -> bool {
    demand_map_vma_page(vaddr, require_write)
}

/// Demand-page a single 4 KiB user-accessible frame at the page containing
/// `vaddr`. Used for stack growth, VMA demand faults, and any other lazy
/// mapping.
///
/// `prot` uses POSIX constants: `PROT_READ=1`, `PROT_WRITE=2`, `PROT_EXEC=4`.
/// Pass `0x3` (`PROT_READ|PROT_WRITE`) for stack pages.
///
/// Called from the page fault ISR and from kernel-context demand faulting.
/// Returns `true` on success, `false` on OOM.
fn demand_map_user_page_locked(vaddr: u64, prot: u64) -> bool {
    use x86_64::structures::paging::PageTableFlags;
    use x86_64::structures::paging::Translate as _;

    const PROT_WRITE: u64 = 0x2;
    const PROT_EXEC: u64 = 0x4;

    let page_vaddr = VirtAddr::new(vaddr & !0xFFF);

    {
        let mapper = unsafe { crate::mm::paging::get_mapper() };
        if mapper.translate_addr(page_vaddr).is_some() {
            return true;
        }
    }

    // Zero-before-exposure (D.4): user-visible demand-paged frame.
    let frame = match crate::mm::frame_allocator::allocate_frame_zeroed() {
        Some(f) => f,
        None => return false,
    };

    // Build PTE flags from the POSIX prot bits.
    let mut data_flags = PageTableFlags::PRESENT | PageTableFlags::USER_ACCESSIBLE;
    if prot & PROT_WRITE != 0 {
        data_flags |= PageTableFlags::WRITABLE;
    }
    if prot & PROT_EXEC == 0 {
        data_flags |= PageTableFlags::NO_EXECUTE;
    }

    if unsafe { crate::mm::paging::map_current_user_page_locked(page_vaddr, frame, data_flags) }
        .is_err()
    {
        crate::mm::frame_allocator::free_frame(frame.start_address().as_u64());
        return false;
    }
    true
}

fn demand_map_user_page(vaddr: u64, prot: u64) -> bool {
    let addr_space = crate::process::current_addr_space();
    let page_base = vaddr & !0xFFF;
    let mapped = {
        let _page_table_guard =
            addr_space.map(|addr_space| unsafe { addr_space.as_ref() }.lock_page_tables());
        demand_map_user_page_locked(vaddr, prot)
    };
    if !mapped {
        return false;
    }
    if crate::smp::is_per_core_ready()
        && let Some(addr_space) = addr_space
    {
        tlb_shootdown_range_from_fault_context(
            unsafe { addr_space.as_ref() },
            page_base,
            page_base + 4096,
        );
    }
    bump_current_addr_space_generation();
    true
}

fn demand_map_vma_page(vaddr: u64, require_write: bool) -> bool {
    const PROT_READ: u64 = 0x1;
    const PROT_WRITE: u64 = 0x2;
    const PROT_EXEC: u64 = 0x4;

    let pid = crate::process::current_pid();
    if pid == 0 {
        return false;
    }

    let addr_space = crate::process::current_addr_space();
    let page_base = vaddr & !0xFFF;
    let mapped = {
        let _page_table_guard =
            addr_space.map(|addr_space| unsafe { addr_space.as_ref() }.lock_page_tables());

        let Some(prot) = crate::process::shared_vma_prot(pid, vaddr) else {
            return false;
        };

        let any_access = prot & (PROT_READ | PROT_WRITE | PROT_EXEC) != 0;
        let write_ok = !require_write || prot & PROT_WRITE != 0;
        if !any_access || !write_ok {
            return false;
        }

        demand_map_user_page_locked(vaddr, prot)
    };
    if !mapped {
        return false;
    }
    if crate::smp::is_per_core_ready()
        && let Some(addr_space) = addr_space
    {
        tlb_shootdown_range_from_fault_context(
            unsafe { addr_space.as_ref() },
            page_base,
            page_base + 4096,
        );
    }
    bump_current_addr_space_generation();
    true
}

// ---------------------------------------------------------------------------
// IDT
// ---------------------------------------------------------------------------

static IDT: Lazy<InterruptDescriptorTable> = Lazy::new(|| {
    let mut idt = InterruptDescriptorTable::new();

    // CPU exceptions
    idt.breakpoint.set_handler_fn(breakpoint_handler);
    idt.page_fault.set_handler_fn(page_fault_handler);
    idt.general_protection_fault
        .set_handler_fn(general_protection_fault_handler);
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault_handler)
            .set_stack_index(gdt::DOUBLE_FAULT_IST_INDEX);
    }

    // Hardware IRQs — timer and reschedule IPI use raw naked-asm entry stubs
    // (Phase 57d Track B) so they are installed via `set_handler_addr`.
    unsafe {
        idt[InterruptIndex::Timer as u8]
            .set_handler_addr(VirtAddr::new(timer_entry as *const () as u64));
    }
    idt[InterruptIndex::Keyboard as u8].set_handler_fn(keyboard_handler);
    // Vector 34 (`InterruptIndex::VirtioNet`) is reserved but no longer
    // installed — Phase 55 C.5 migrated virtio-net to the HAL IRQ contract
    // (allocated from the device-IRQ bank at `DEVICE_IRQ_VECTOR_BASE`).
    idt[InterruptIndex::Serial as u8].set_handler_fn(serial_handler);
    idt[InterruptIndex::Mouse as u8].set_handler_fn(mouse_handler);

    // APIC spurious interrupt vector — must NOT send EOI.
    idt[InterruptIndex::Spurious as u8].set_handler_fn(spurious_handler);

    // SMP IPI vectors (Phase 25).
    unsafe {
        idt[crate::smp::ipi::IPI_RESCHEDULE]
            .set_handler_addr(VirtAddr::new(reschedule_ipi_entry as *const () as u64));
    }
    idt[crate::smp::ipi::IPI_TLB_SHOOTDOWN].set_handler_fn(tlb_shootdown_ipi_handler);
    idt[crate::smp::ipi::IPI_CACHE_DRAIN].set_handler_fn(cache_drain_ipi_handler);

    // Phase 55 C.3: device MSI / MSI-X vector stubs.
    // Each stub dispatches through DEVICE_IRQ_TABLE; callers register
    // handlers at runtime via `register_device_irq`.
    let bank: &[(u8, extern "x86-interrupt" fn(InterruptStackFrame))] = &[
        (DEVICE_IRQ_VECTOR_BASE, device_irq_stub_0),
        (DEVICE_IRQ_VECTOR_BASE + 1, device_irq_stub_1),
        (DEVICE_IRQ_VECTOR_BASE + 2, device_irq_stub_2),
        (DEVICE_IRQ_VECTOR_BASE + 3, device_irq_stub_3),
        (DEVICE_IRQ_VECTOR_BASE + 4, device_irq_stub_4),
        (DEVICE_IRQ_VECTOR_BASE + 5, device_irq_stub_5),
        (DEVICE_IRQ_VECTOR_BASE + 6, device_irq_stub_6),
        (DEVICE_IRQ_VECTOR_BASE + 7, device_irq_stub_7),
        (DEVICE_IRQ_VECTOR_BASE + 8, device_irq_stub_8),
        (DEVICE_IRQ_VECTOR_BASE + 9, device_irq_stub_9),
        (DEVICE_IRQ_VECTOR_BASE + 10, device_irq_stub_10),
        (DEVICE_IRQ_VECTOR_BASE + 11, device_irq_stub_11),
        (DEVICE_IRQ_VECTOR_BASE + 12, device_irq_stub_12),
        (DEVICE_IRQ_VECTOR_BASE + 13, device_irq_stub_13),
        (DEVICE_IRQ_VECTOR_BASE + 14, device_irq_stub_14),
        (DEVICE_IRQ_VECTOR_BASE + 15, device_irq_stub_15),
    ];
    for (vec, stub) in bank {
        idt[*vec].set_handler_fn(*stub);
    }

    idt
});

/// Load the IDT.
pub fn init() {
    IDT.load();
}

// ---------------------------------------------------------------------------
// Phase 57d Track B — naked-asm preemption entry stubs
// ---------------------------------------------------------------------------
//
// `timer_entry` and `reschedule_ipi_entry` are ring-aware two-path stubs.
// They push all 15 GPRs BEFORE any Rust function prologue can clobber them,
// then call the appropriate Rust handler (`*_user` or `*_kernel`) with a
// pointer to the on-stack frame.
//
// GPR push order (both stubs, both paths): r15 first → rax last, so that
// rax ends up at the lowest address (gprs[0] in `PreemptTrapFrameUser/Kernel`).
//
// Kernel-path alignment: the kernel stack is not guaranteed 16-byte aligned
// at interrupt entry, so we save RSP in r12 (callee-saved), align with
// `and rsp, -16`, call, then restore from r12.  After the call, `pop r12`
// loads the interrupted task's r12 from the frame slot — the scratch value
// in the live register is overwritten by the pop, which is correct.

global_asm!(
    // -----------------------------------------------------------------------
    // Shared macros: GPR save / restore used by both stubs.
    // -----------------------------------------------------------------------
    ".macro save_gprs_all",
    "push r15",
    "push r14",
    "push r13",
    "push r12",
    "push r11",
    "push r10",
    "push r9",
    "push r8",
    "push rbp",
    "push rdi",
    "push rsi",
    "push rdx",
    "push rcx",
    "push rbx",
    "push rax",
    ".endm",
    "",
    ".macro restore_gprs_all",
    "pop rax",
    "pop rbx",
    "pop rcx",
    "pop rdx",
    "pop rsi",
    "pop rdi",
    "pop rbp",
    "pop r8",
    "pop r9",
    "pop r10",
    "pop r11",
    "pop r12",
    "pop r13",
    "pop r14",
    "pop r15",
    ".endm",
    "",
    // -----------------------------------------------------------------------
    // timer_entry
    // -----------------------------------------------------------------------
    ".global timer_entry",
    "timer_entry:",
    // CS is at [rsp+8] on both ring-0 (3-field frame: rip/cs/rflags) and
    // ring-3 (5-field frame: rip/cs/rflags/rsp/ss) IRQ entries.
    "test QWORD PTR [rsp+8], 3",
    "jnz .Ltimer_user",
    "",
    // --- Kernel path --------------------------------------------------------
    ".Ltimer_kernel:",
    "save_gprs_all",
    // After 15 pushes, rsp = &gprs[0].
    // interrupted RSP = rsp + 15*8 + 3*8 = rsp + 144.
    "lea rsi, [rsp + 144]", // arg2: captured_kernel_rsp
    "cld",
    "mov rdi, rsp", // arg1: &PreemptTrapFrameKernel
    "mov r12, rsp", // save pre-alignment rsp (r12 is callee-saved)
    "and rsp, -16", // align to 16 bytes for SysV call ABI
    "call timer_handler_kernel",
    "mov rsp, r12", // restore (pop r12 below loads original r12 from frame)
    "restore_gprs_all",
    "iretq",
    "",
    // --- User path ----------------------------------------------------------
    ".Ltimer_user:",
    "save_gprs_all",
    // After 15 GPR pushes + 5 CPU-pushed fields = 160 bytes.
    // If TSS.RSP0 is 16-aligned, 160 ≡ 0 (mod 16) → already aligned.
    "cld",
    "mov rdi, rsp", // arg1: &mut PreemptTrapFrameUser
    "call timer_handler_user",
    "restore_gprs_all",
    "iretq",
    "",
    // -----------------------------------------------------------------------
    // reschedule_ipi_entry
    // -----------------------------------------------------------------------
    ".global reschedule_ipi_entry",
    "reschedule_ipi_entry:",
    "test QWORD PTR [rsp+8], 3",
    "jnz .Lrescheduleipi_user",
    "",
    // --- Kernel path --------------------------------------------------------
    ".Lrescheduleipi_kernel:",
    "save_gprs_all",
    "lea rsi, [rsp + 144]",
    "cld",
    "mov rdi, rsp",
    "mov r12, rsp",
    "and rsp, -16",
    "call reschedule_ipi_handler_kernel",
    "mov rsp, r12",
    "restore_gprs_all",
    "iretq",
    "",
    // --- User path ----------------------------------------------------------
    ".Lrescheduleipi_user:",
    "save_gprs_all",
    "cld",
    "mov rdi, rsp",
    "call reschedule_ipi_handler_user",
    "restore_gprs_all",
    "iretq",
);

// ---------------------------------------------------------------------------
// Phase 57d C.2 — preempt_resume_to_user
// ---------------------------------------------------------------------------
//
// Restores the full user-mode register state saved by preempt_to_scheduler
// and returns to the interrupted user instruction via iretq.
//
// PreemptFrame offsets (kernel_core::preempt_frame::PreemptFrame):
//   rax=0   rbx=8   rcx=16  rdx=24  rsi=32  rdi=40  rbp=48
//   r8=56   r9=64   r10=72  r11=80  r12=88  r13=96  r14=104 r15=112
//   rip=120 cs=128  rflags=136 rsp=144 ss=152
//
// Calling convention: rdi = *const PreemptFrame (SysV AMD64 arg1).
// Called from the scheduler dispatch loop (D.3) with IRQs disabled.
// Never returns.

core::arch::global_asm!(
    ".global preempt_resume_to_user",
    "preempt_resume_to_user:",
    // Build the iretq frame on the current (scheduler) stack.
    // iretq pops: rip, cs, rflags, rsp, ss — push in reverse (ss first).
    "mov rax, [rdi + 152]", // ss
    "push rax",
    "mov rax, [rdi + 144]", // rsp (user-mode stack pointer)
    "push rax",
    "mov rax, [rdi + 136]", // rflags
    "push rax",
    "mov rax, [rdi + 128]", // cs
    "push rax",
    "mov rax, [rdi + 120]", // rip
    "push rax",
    // Restore GPRs — all except rax and rdi (rdi is still our frame pointer).
    "mov rbx, [rdi + 8]",
    "mov rcx, [rdi + 16]",
    "mov rdx, [rdi + 24]",
    "mov rsi, [rdi + 32]",
    "mov rbp, [rdi + 48]",
    "mov r8,  [rdi + 56]",
    "mov r9,  [rdi + 64]",
    "mov r10, [rdi + 72]",
    "mov r11, [rdi + 80]",
    "mov r12, [rdi + 88]",
    "mov r13, [rdi + 96]",
    "mov r14, [rdi + 104]",
    "mov r15, [rdi + 112]",
    // Restore rax, then rdi last (pointer becomes invalid after this).
    "mov rax, [rdi + 0]",
    "mov rdi, [rdi + 40]",
    "iretq",
    //
    // ---------------------------------------------------------------------------
    // Phase 57d D.3 (fix) — dispatch_preempted_and_resume
    //
    // Dispatches a preempted task by:
    //   1. Building a switch_context-compatible frame on the scheduler stack so
    //      that per_core_scheduler_rsp is updated BEFORE we iretq.
    //   2. Saving the new scheduler RSP to *per_sched_rsp_ptr (rdi).
    //   3. Jumping to preempt_resume_to_user to restore user GPRs and iretq.
    //
    // When the task is later preempted again (or cooperatively yields back),
    // preempt_to_scheduler calls switch_context(task_save, *per_sched_rsp_ptr).
    // switch_context loads our frame and `ret`s to .Ldispatch_preempted_resume,
    // which then `ret`s to the call site (the dispatch loop's epilogue).
    //
    // Stack layout built by this function (low address first, relative to the
    // saved_rsp value stored into *per_sched_rsp_ptr):
    //   [saved_rsp+0]:  RFLAGS (saved by pushf; IF=0 since caller had IRQs disabled)
    //   [saved_rsp+8]:  r15
    //   [saved_rsp+16]: r14
    //   [saved_rsp+24]: r13
    //   [saved_rsp+32]: r12
    //   [saved_rsp+40]: rbp
    //   [saved_rsp+48]: rbx
    //   [saved_rsp+56]: .Ldispatch_preempted_resume  ← switch_context ret target
    //   [saved_rsp+64]: call return addr              ← returned to by .Ldispatch_preempted_resume ret
    //
    // PRECONDITION: IRQs must be disabled on entry (pushf captures IF=0).
    // Args (SysV AMD64): rdi = per_sched_rsp_ptr, rsi = frame (*const PreemptFrame)
    // ---------------------------------------------------------------------------
    ".global dispatch_preempted_and_resume",
    "dispatch_preempted_and_resume:",
    // Push .Ldispatch_preempted_resume as the switch_context `ret` target.
    "lea rax, [rip + .Ldispatch_preempted_resume]",
    "push rax",
    // Push callee-saved registers (matching switch_context's push order).
    "push rbx",
    "push rbp",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    // pushf saves RFLAGS (including IF, which is 0 because IRQs are disabled).
    "pushf",
    "cli", // redundant but explicit: ensure IF=0 in the saved frame
    // Save the current RSP (= address of the RFLAGS word we just pushed).
    "mov [rdi], rsp",
    // Dispatch the preempted task. rsi already holds the frame pointer.
    "mov rdi, rsi",
    "jmp preempt_resume_to_user",
    // Landing label: switch_context restores callee-saves from our frame,
    // pops this label as the return address, and jumps here.
    // At this point RSP points at the `call dispatch_preempted_and_resume`
    // return address; the second `ret` returns to the dispatch loop.
    ".Ldispatch_preempted_resume:",
    "ret",
);

// ---------------------------------------------------------------------------
// Phase 57e Track C.1 / C.4 — preempt_resume_to_kernel + dispatch_preempted_and_resume_kernel
// ---------------------------------------------------------------------------
//
// Same-CPL iretq resume: the CPU pops only `rip / cs / rflags` (3 fields,
// 24 bytes) and does **not** swap rsp/ss.  Therefore RSP must already point
// at the interrupted task's kernel stack at the moment of iretq.  This
// routine's structure differs from `preempt_resume_to_user` in two ways:
//
//   1. We must `mov rsp, preempt_frame.rsp` BEFORE pushing the iretq frame
//      so the 3-field push lands at the correct location on the interrupted
//      kernel stack.
//   2. The iretq frame is 3 fields (rflags / cs / rip) instead of 5
//      (rflags / cs / rip / rsp / ss).
//
// Stack-safety note: the 24 bytes the iretq pushes occupy what was
// originally the CPU-pushed IRQ frame on the interrupted kernel stack
// (between `captured_kernel_rsp - 24` and `captured_kernel_rsp - 1`).
// That region was the same scratch area the CPU wrote when the IRQ fired,
// so no live kernel-stack data is overwritten.
//
// Calling convention: rdi = *const PreemptFrame.  Called only from
// `dispatch_preempted_and_resume_kernel` with IRQs disabled.  Never returns.
//
// Phase 57e deferral cleanup (2026-05-07): the kernel-mode preempt-resume
// asm stubs (`preempt_resume_to_kernel` and `dispatch_preempted_and_resume_kernel`)
// were removed.  They were only reachable from the now-removed
// `check_and_preempt_kernel` IRQ-side gate — see
// `docs/post-mortems/2026-05-07-57e-preempt-full-deferred.md`.

// Rust-side declarations for the asm entry symbols (used when installing
// into the IDT via `set_handler_addr`).
unsafe extern "C" {
    fn timer_entry();
    fn reschedule_ipi_entry();
    /// Phase 57d C.2 — resume a preempted user-mode task.
    ///
    /// Restores all 15 GPRs and the full iretq frame from `frame`, then
    /// executes `iretq` to return to the interrupted user-mode instruction.
    /// Never returns. Called only from dispatch_preempted_and_resume asm.
    #[cfg(feature = "preempt-voluntary")]
    #[allow(dead_code)]
    pub fn preempt_resume_to_user(frame: *const kernel_core::preempt_frame::PreemptFrame) -> !;
    /// Phase 57d D.3 (fix) — dispatch a preempted task after updating sched RSP.
    ///
    /// Builds a `switch_context`-compatible frame on the scheduler stack,
    /// saves the scheduler RSP, and jumps to `preempt_resume_to_user`.
    ///
    /// From Rust's perspective this function returns `()` — it returns after
    /// the task next switches back to the scheduler (via preemption or a
    /// cooperative yield). IRQs must be disabled on entry.
    #[cfg(feature = "preempt-voluntary")]
    pub(crate) fn dispatch_preempted_and_resume(
        per_sched_rsp_ptr: *mut u64,
        frame: *const kernel_core::preempt_frame::PreemptFrame,
    );
    // Phase 57e deferral cleanup: `preempt_resume_to_kernel` and
    // `dispatch_preempted_and_resume_kernel` removed (see global_asm
    // block above and the post-mortem).
}

// ---------------------------------------------------------------------------
// Exception handlers
// ---------------------------------------------------------------------------

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    // Use _panic_print to avoid deadlocking on the serial mutex if the exception
    // fires while normal code holds the lock.
    _panic_print(format_args!("[int] breakpoint: {:?}\n", stack_frame));
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

extern "x86-interrupt" fn page_fault_handler(
    mut stack_frame: InterruptStackFrame,
    err: PageFaultErrorCode,
) {
    let addr = x86_64::registers::control::Cr2::read();

    // Check if the fault came from ring 3 (user mode).
    if stack_frame.code_segment.rpl() == x86_64::PrivilegeLevel::Ring3 {
        // P17-T031: detect CoW faults — a write to a present, non-writable
        // page marked with BIT_9 (the CoW marker set by cow_clone_user_pages).
        let is_write = err.contains(PageFaultErrorCode::CAUSED_BY_WRITE);
        let is_present = err.contains(PageFaultErrorCode::PROTECTION_VIOLATION);
        if is_write
            && is_present
            && let Ok(fault_vaddr) = addr
        {
            let fault_addr_u64 = fault_vaddr.as_u64();
            // CoW fault — revalidate and resolve directly in the ISR. Safe
            // because the fault is from ring 3 (no kernel locks held), and
            // the CoW path serializes its page-table mutation under the
            // current address-space lock before issuing TLB shootdowns.
            if resolve_cow_fault(fault_addr_u64) {
                // DIAGNOSTIC: per-fault rusage accounting disabled to A/B-test
                // whether Phase 61 Track E.4's scheduler-lock attempts in IRQ
                // context are responsible for the vfs_server slow_req tail
                // latency observed on this branch. Re-enable to measure delta.
                // crate::task::scheduler::current_task_record_page_fault(false);
                assert_preempt_count_zero_on_return_to_user(&stack_frame);
                return;
            }
            // OOM or no-longer-CoW mapping — fall through to other handlers / kill.
        }

        // Demand-paging for the stack region: musl's __init_tls and malloc
        // write above ELF_STACK_TOP (Linux maps an 8 MB region so this is
        // always valid there). When the fault is a write to an unmapped page
        // within 8 MiB above ELF_STACK_TOP, allocate a fresh frame and map it.
        if is_write
            && !is_present
            && let Ok(fault_vaddr) = addr
        {
            let fault_addr_u64 = fault_vaddr.as_u64();
            let stack_top = crate::mm::elf::ELF_STACK_TOP;
            let stack_bottom = stack_top - crate::mm::elf::STACK_PAGES * 4096;
            // Allow demand-paging 8 MiB above ELF_STACK_TOP and down to guard page.
            const DEMAND_LIMIT: u64 = 8 * 1024 * 1024; // 8 MiB
            if fault_addr_u64 >= stack_bottom
                && fault_addr_u64 < stack_top + DEMAND_LIMIT
                && !has_guard_marker(fault_addr_u64)
                && demand_map_user_page(fault_addr_u64, 0x3)
            // PROT_READ|PROT_WRITE
            {
                assert_preempt_count_zero_on_return_to_user(&stack_frame);
                return;
            }
        }

        // Phase 36: VMA-based demand paging for mmap regions.
        // If the fault address is inside a valid VMA, allocate a frame on demand.
        if !is_present && let Ok(fault_vaddr) = addr {
            let fault_addr_u64 = fault_vaddr.as_u64();
            if demand_map_vma_page(fault_addr_u64, is_write) {
                assert_preempt_count_zero_on_return_to_user(&stack_frame);
                return;
            }
        }

        let pid = crate::process::current_pid();
        _panic_print(format_args!(
            "[int] userspace page fault: pid={} addr={:?} err={:?} rip={:#x} — process killed\n",
            pid,
            addr,
            err,
            stack_frame.instruction_pointer.as_u64()
        ));
        _panic_print(format_args!(
            "[int] RSP={:#x}\n",
            stack_frame.stack_pointer.as_u64()
        ));
        // Phase 57e diag: not-present userspace faults on what should be a
        // mapped code/data page are the same shape as the kernel slab UAF;
        // dump the PTE walk to localise the missing-PTE level.
        if !err.contains(PageFaultErrorCode::PROTECTION_VIOLATION)
            && let Ok(fault_va) = addr
        {
            dump_pte_walk_diagnostics(fault_va.as_u64());
        }
        if crate::smp::is_per_core_ready() {
            let task_idx = crate::smp::per_core()
                .current_task_idx
                .load(Ordering::Relaxed);
            if let Some(guard) = crate::task::try_lock_scheduler()
                && task_idx >= 0
                && let Some(task) = guard.get_task(task_idx as usize)
            {
                _panic_print(format_args!(
                    "[int] task[{}]: state={:?} saved_rsp=0x{:016x}\n",
                    task_idx, task.state, task.saved_rsp
                ));
            }
        }
        panic_diag::dump_crash_context();
        crate::trace::dump_trace_rings();
        // Store the PID for the trampoline. Safe: interrupts are disabled
        // during exception handling on a single CPU.
        FAULT_KILL_PID.store(pid, Ordering::Relaxed);
        // Redirect the interrupted context to fault_kill_trampoline, which
        // runs in ring 0 outside interrupt context where locking is safe.
        // SAFETY: we modify the interrupt return frame while interrupts are
        // disabled. The trampoline is a valid kernel function pointer.
        // We must also set RSP to the current kernel stack (not the user RSP
        // that was saved in the frame), otherwise IRET would pop the user RSP
        // and the trampoline would run with an unmapped stack → GPF.
        let kernel_rsp: u64;
        unsafe {
            core::arch::asm!("mov {}, rsp", out(reg) kernel_rsp);
        }
        unsafe {
            stack_frame.as_mut().update(|f| {
                f.instruction_pointer = VirtAddr::new(fault_kill_trampoline as *const () as u64);
                f.code_segment = gdt::kernel_code_selector();
                f.cpu_flags &= !x86_64::registers::rflags::RFlags::INTERRUPT_FLAG;
                f.stack_pointer = VirtAddr::new(kernel_rsp);
                f.stack_segment = gdt::kernel_data_selector();
            });
        }
        return;
    }

    // Ring-0 page fault: unrecoverable kernel bug.
    _panic_print(format_args!(
        "[int] kernel page fault: addr={:?} err={:?}\n{:?}\n",
        addr, err, stack_frame
    ));
    let (cr3_frame, _) = x86_64::registers::control::Cr3::read_raw();
    _panic_print(format_args!(
        "[int] KERNEL page fault — CR3=0x{:016x}\n",
        cr3_frame.start_address().as_u64()
    ));
    if let Ok(fault_va) = addr {
        dump_pte_walk_diagnostics(fault_va.as_u64());
    }
    // Phase 57e diag — the per-frame allocate/free trace ring (`mm::frame_trace`)
    // that originally dumped the active CR3 history here was retired alongside
    // the Phase 57e deferral cleanup (its sole purpose was diagnosing Bug #7,
    // closed in d8db950).  The static 512 KiB ring + per-allocation recording
    // also broke `kernel::mm::frame_allocator::tests::allocate_frame_hot_path_tolerates_reentrant_free`.
    let _ = cr3_frame;
    panic_diag::dump_crash_context();
    crate::trace::dump_trace_rings();
    crate::hlt_loop();
}

/// User-path replacement for `maybe_redirect_group_exit_trampoline`.
///
/// Operates directly on the on-stack [`PreemptTrapFrameUser`] so it can be
/// called from the naked-asm user-path handlers without requiring an
/// `InterruptStackFrame`.  The ring-3 check is omitted — the user-path
/// handler is only ever reached when `(cs & 3) == 3`.
fn maybe_redirect_group_exit_trampoline_user(frame: &mut PreemptTrapFrameUser) {
    if !crate::smp::is_per_core_ready() {
        return;
    }

    let task_idx = crate::smp::per_core()
        .current_task_idx
        .load(Ordering::Relaxed);
    let should_redirect = if let Some(guard) = crate::task::try_lock_scheduler() {
        task_idx >= 0
            && guard
                .get_task(task_idx as usize)
                .map(|task| task.group_exit_pending)
                .unwrap_or(false)
    } else {
        false
    };
    if !should_redirect {
        return;
    }

    let kernel_rsp: u64;
    unsafe {
        core::arch::asm!("mov {}, rsp", out(reg) kernel_rsp);
    }
    frame.rip = crate::arch::x86_64::syscall::forced_group_exit_trampoline as *const () as u64;
    frame.cs = u64::from(gdt::kernel_code_selector().0);
    frame.rflags &= !0x200u64; // clear INTERRUPT_FLAG (bit 9)
    frame.rsp = kernel_rsp;
    frame.ss = u64::from(gdt::kernel_data_selector().0);
}

extern "x86-interrupt" fn general_protection_fault_handler(
    mut stack_frame: InterruptStackFrame,
    _err: u64,
) {
    // Capture the user's callee-saved GPRs IMMEDIATELY, before any Rust
    // code can clobber them. With the `x86-interrupt` calling convention,
    // these still hold the user's values at the first instruction of the
    // handler body (caller-saved regs are spilled by the entry stub but
    // callee-saved are preserved across the Rust call boundary).
    // `panic_diag::capture_registers` runs deeper in the call chain, so
    // r12-r15 there reflect kernel state, not the user's view.
    let user_r12: u64;
    let user_r13: u64;
    let user_r14: u64;
    let user_r15: u64;
    let user_rbx: u64;
    let user_rbp: u64;
    unsafe {
        core::arch::asm!(
            "mov {0}, rbx",
            "mov {1}, rbp",
            "mov {2}, r12",
            "mov {3}, r13",
            "mov {4}, r14",
            "mov {5}, r15",
            out(reg) user_rbx,
            out(reg) user_rbp,
            out(reg) user_r12,
            out(reg) user_r13,
            out(reg) user_r14,
            out(reg) user_r15,
            options(nostack, preserves_flags),
        );
    }
    // Check if the fault came from ring 3.
    if stack_frame.code_segment.rpl() == x86_64::PrivilegeLevel::Ring3 {
        let pid = crate::process::current_pid();
        _panic_print(format_args!(
            "[int] userspace GPF: pid={} — process killed\n{:?}\n",
            pid, stack_frame
        ));
        _panic_print(format_args!(
            "[int] user GPRs at fault: rbx={:#018x} rbp={:#018x} r12={:#018x} r13={:#018x} r14={:#018x} r15={:#018x}\n",
            user_rbx, user_rbp, user_r12, user_r13, user_r14, user_r15
        ));
        if crate::smp::is_per_core_ready() {
            let task_idx = crate::smp::per_core()
                .current_task_idx
                .load(Ordering::Relaxed);
            if let Some(guard) = crate::task::try_lock_scheduler()
                && task_idx >= 0
                && let Some(task) = guard.get_task(task_idx as usize)
            {
                _panic_print(format_args!(
                    "[int] pid={} task[{}]: state={:?}\n",
                    pid, task_idx, task.state
                ));
            }
        }
        let selector_idx = _err >> 3;
        let table = (_err >> 1) & 3;
        let external = _err & 1;
        _panic_print(format_args!(
            "[int] GPF error_code={:#x} (selector_idx={}, table={}, external={})\n",
            _err, selector_idx, table, external
        ));
        panic_diag::dump_crash_context();
        crate::trace::dump_trace_rings();
        // Store the PID and redirect to the kill trampoline (same pattern as
        // page_fault_handler — no blocking allowed inside an ISR).
        FAULT_KILL_PID.store(pid, Ordering::Relaxed);
        // SAFETY: same as page_fault_handler above.
        let kernel_rsp: u64;
        unsafe {
            core::arch::asm!("mov {}, rsp", out(reg) kernel_rsp);
        }
        unsafe {
            stack_frame.as_mut().update(|f| {
                f.instruction_pointer = VirtAddr::new(fault_kill_trampoline as *const () as u64);
                f.code_segment = gdt::kernel_code_selector();
                f.cpu_flags &= !x86_64::registers::rflags::RFlags::INTERRUPT_FLAG;
                f.stack_pointer = VirtAddr::new(kernel_rsp);
                f.stack_segment = gdt::kernel_data_selector();
            });
        }
        return;
    }
    _panic_print(format_args!("[int] GPF: {:?}\n", stack_frame));
    let selector_idx = _err >> 3;
    let table = (_err >> 1) & 3;
    let external = _err & 1;
    _panic_print(format_args!(
        "[int] GPF error_code={:#x} (selector_idx={}, table={}, external={})\n",
        _err, selector_idx, table, external
    ));
    panic_diag::dump_crash_context();
    crate::trace::dump_trace_rings();
    crate::hlt_loop();
}

extern "x86-interrupt" fn double_fault_handler(stack_frame: InterruptStackFrame, _err: u64) -> ! {
    _panic_print(format_args!("[int] DOUBLE FAULT: {:?}\n", stack_frame));
    _panic_print(format_args!(
        "[int] IST RSP={:#x}\n",
        stack_frame.stack_pointer.as_u64()
    ));
    panic_diag::dump_crash_context();
    crate::trace::dump_trace_rings();
    crate::hlt_loop();
}

// ---------------------------------------------------------------------------
// Hardware IRQ vector offsets
// ---------------------------------------------------------------------------

/// IRQ vectors remapped to start above the CPU exception range.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum InterruptIndex {
    Timer = 32,
    Keyboard = 33,
    /// Reserved. Was used for virtio-net pre-Phase-55; virtio-net now
    /// allocates from the device-IRQ bank at `DEVICE_IRQ_VECTOR_BASE` via
    /// the HAL. Kept in the enum so the vector number isn't silently
    /// repurposed before we decide what (if anything) to put here.
    #[allow(dead_code)]
    VirtioNet = 34,
    Serial = 36,
    /// Phase 56 Track B.2 — PS/2 AUX (mouse) IRQ12. With the standard PIC
    /// remap (master=32, slave=40), IRQ12 → vector 44.
    Mouse = 44,
    Spurious = 0xFF,
}

// ---------------------------------------------------------------------------
// PIC
// ---------------------------------------------------------------------------

static PICS: Mutex<pic8259::ChainedPics> = Mutex::new(unsafe { pic8259::ChainedPics::new(32, 40) });

/// Initialize and unmask the 8259 PIC.
///
/// # Safety
///
/// Must be called after the IDT is loaded and before interrupts are enabled.
/// Calling it out of order can cause IRQs to fire without a registered handler,
/// resulting in a triple fault.
pub unsafe fn init_pics() {
    // Phase 57b G.8 — `PICS` is classified `explicit-preempt-and-cli` per
    // Track A.1 audit (`kernel/src/arch/x86_64/interrupts.rs:756`). The ISR
    // EOI callsites (vectors 32, 33, 36, 44) already run with IF=0 and do
    // not touch the per-task preempt counter, so they need no migration.
    // The lone task-context callsite is this `init_pics` body, which runs
    // before interrupts are enabled but must still pair `preempt_disable` /
    // `preempt_enable` to keep F.1 preempt-discipline balanced. The
    // `without_interrupts` wrap is a no-op for IF (already off) but stays
    // for shape consistency with G.8.b / G.5.c.
    //
    // `preempt_disable` is lock-free (Phase 57b D.2), so calling it before
    // `without_interrupts` cannot recurse.
    crate::task::scheduler::preempt_disable();
    x86_64::instructions::interrupts::without_interrupts(|| unsafe {
        let mut pics = PICS.lock();
        pics.initialize();
        // Mask every IRQ line except: IRQ0 (timer), IRQ1 (keyboard),
        // IRQ2 (cascade — required to receive any slave IRQ), and IRQ12
        // (PS/2 AUX / mouse, slave bit 4).
        //
        // A set bit disables the line. Any unmasked line without an IDT
        // handler would vector into an uninitialized entry and cause a
        // triple fault.
        //
        // master: bits 3–7 masked (0b1111_1000) — IRQ0/1/2 unmasked.
        // slave:  bits 0–3 + 5–7 masked (0b1110_1111) — IRQ12 unmasked.
        pics.write_masks(0b1111_1000, 0b1110_1111);
    });
    crate::task::scheduler::preempt_enable();
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

static TICK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Return the current timer tick count (monotonically increasing).
pub fn tick_count() -> u64 {
    TICK_COUNT.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Phase 57d Track G — voluntary preemption IRQ-return helpers
// ---------------------------------------------------------------------------

/// Source of a voluntary-preemption event, for tracing.
#[cfg(feature = "preempt-voluntary")]
#[derive(Debug, Clone, Copy)]
pub enum PreemptTrigger {
    Timer,
    RescheduleIpi,
    #[allow(dead_code)]
    PreemptEnableZeroCrossing,
}

/// Emit a preemption trace entry to the sched-trace ring.
///
/// Only compiled under `cfg(feature = "sched-trace")`. Zero overhead in
/// the default build.
///
/// Phase 57e Track F.3: encodes `kernel_mode` into bit 7 of the trigger
/// discriminant (low 7 bits = `PreemptTrigger`, high bit = `kernel_mode`).
/// The existing `SchedTrace` schema (`pid` / `old_state` / `new_state`) is
/// preserved unchanged; downstream tooling that already reads `new_state`
/// continues to see the trigger discriminant after masking with `0x7F`.
#[cfg(feature = "sched-trace")]
fn emit_preempt_trace(rip: u64, trigger: PreemptTrigger, kernel_mode: bool) {
    let new_state = (trigger as u8) | if kernel_mode { 0x80 } else { 0 };
    // Repurpose sched_trace::record: pid=preempted_rip (truncated to u32),
    // old_state=255 (preempt sentinel), new_state=encoded trigger+kernel_mode.
    crate::task::sched_trace::record(rip as u32, 255, new_state);
}

/// Check the four voluntary-preemption conditions and, if met, preempt the
/// interrupted user-mode task.
///
/// Called from `timer_handler_user` and `reschedule_ipi_handler_user` after
/// the IRQ's tick/EOI work is complete.
///
/// # Conditions checked (all must be true to preempt)
/// 1. `from_user` — implicit: callers are in the `_user` handler path.
/// 2. `preempt_count == 0` — no preempt-disable lock held by the task.
/// 3. `reschedule || preempt_resched_pending` — scheduler flagged a switch.
///
/// The function also guards against group-exit redirects: if
/// `maybe_redirect_group_exit_trampoline_user` rewrites `frame.cs` to the
/// kernel code selector before this call, the frame is skipped (preemption
/// would corrupt the iretq return because `preempt_resume_to_user` always
/// builds a 5-slot ring-3 frame).
///
/// # Safety
/// Must be called with IRQs disabled (guaranteed by IRQ handler context).
/// `frame` must point to a valid on-stack `PreemptTrapFrameUser`.
#[cfg(feature = "preempt-voluntary")]
unsafe fn check_and_preempt_user(frame: &mut PreemptTrapFrameUser, trigger: PreemptTrigger) {
    // Guard: maybe_redirect_group_exit_trampoline_user (called before this
    // function) can rewrite frame.cs to the kernel code selector (ring 0)
    // when group_exit_pending is set. preempt_resume_to_user always builds
    // a 5-slot user-return iretq frame; resuming a ring-0 CS frame via iretq
    // causes the CPU to pop only 3 slots instead of 5 (same-privilege return),
    // leaving RSP pointing at frame data and corrupting the return. Skip
    // preemption whenever the frame is no longer ring 3.
    if frame.cs & 3 != 3 {
        return;
    }
    let pc = crate::task::scheduler::peek_preempt_count_irq();
    if pc != 0 {
        return; // task holds a preempt-disable lock — do not preempt
    }
    let Some(core) = crate::smp::try_per_core() else {
        return;
    };
    let reschedule = core
        .reschedule
        .swap(false, core::sync::atomic::Ordering::AcqRel);
    let pending = core
        .preempt_resched_pending
        .swap(false, core::sync::atomic::Ordering::AcqRel);
    if !reschedule && !pending {
        return; // no rescheduling requested
    }
    // All conditions met — capture and preempt.
    #[cfg(feature = "sched-trace")]
    unsafe {
        emit_preempt_trace(frame.rip, trigger, false);
    }
    #[cfg(not(feature = "sched-trace"))]
    let _ = trigger;
    crate::task::scheduler::preempt_to_scheduler(frame);
}

// Phase 57e deferral cleanup (2026-05-07): `check_and_preempt_kernel`
// removed.  It was the kernel-mode counterpart of
// `check_and_preempt_user` and was called from `timer_handler_kernel`
// and `reschedule_ipi_handler_kernel` under preempt-full.  Both call
// sites are now early-return for the kernel-mode path; see the
// post-mortem at `docs/post-mortems/2026-05-07-57e-preempt-full-deferred.md`.

/// Phase 57d Track B — timer handler, user (ring 3) path.
///
/// Called from the `timer_entry` naked stub when the interrupted context was
/// ring 3.  `frame` points directly at the on-stack [`PreemptTrapFrameUser`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn timer_handler_user(frame: &mut PreemptTrapFrameUser) {
    if !USING_APIC.load(Ordering::Relaxed) || crate::smp::is_bsp() {
        TICK_COUNT.fetch_add(1, Ordering::Relaxed);
        crate::time::on_timer_tick_isr();
        // Phase 57d follow-up: poll the i8042 from the timer ISR as a
        // backstop against the held-key cursor freeze. Diagnostic
        // counters showed mouse bytes never reach the controller's
        // output buffer during sustained keyboard autorepeat — IRQ12
        // is being suppressed (we suspect by QEMU's PS/2 arbitration
        // when the CPU is actively servicing IRQ1). The timer ISR
        // runs at the configured tick rate (default 100 Hz) and is
        // not on the IRQ1/IRQ12 priority class, so it can drain the
        // i8042 even while kbd autorepeat is hot. The drain helper
        // bails immediately when the output buffer is empty, so the
        // common case (no input) costs one port read per tick.
        ps2_drain_all_bytes();
    }
    // DIAGNOSTIC: per-tick CS sampling (Phase 61 Track E.2) disabled to
    // A/B-test whether the per-tick scheduler-lock attempts on every core
    // are responsible for the vfs_server slow_req tail latency observed
    // on this branch. Re-enable to measure delta.
    // let period_ms: u64 = if crate::smp::is_bsp() { 1 } else { 10 };
    // crate::task::scheduler::tick_account_current_task(true, period_ms);
    crate::task::signal_reschedule();
    maybe_redirect_group_exit_trampoline_user(frame);
    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    } else {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Timer as u8);
        }
    }
    // Phase 57e Bug #9 — clamp preempt_count to 0 in release builds; panic on
    // non-zero in debug builds.  Helper handles the mode split.
    crate::task::scheduler::assert_preempt_count_zero_at_user_return();
    #[cfg(feature = "preempt-voluntary")]
    unsafe {
        check_and_preempt_user(frame, PreemptTrigger::Timer);
    }
}

/// Phase 57d Track B — timer handler, kernel path.
///
/// Called from the `timer_entry` naked stub when the interrupted context was
/// ring 0.  `captured_kernel_rsp` is the RSP value the interrupted kernel
/// code had immediately before the interrupt fired.
///
/// Under `preempt-voluntary` only, this handler runs the tick / EOI /
/// reschedule-flag work and returns — kernel-mode preemption is structurally
/// absent (57d behaviour preserved).
///
/// Under `preempt-full` (Phase 57e Track F.1), the handler additionally
/// performs the same preempt check as the user-path handler and tail-calls
/// `preempt_to_scheduler_kernel` when all conditions are met.  The group-exit
/// redirect is intentionally **not** applied on this path — see
/// `check_and_preempt_kernel` for the asymmetry rationale.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn timer_handler_kernel(
    frame: &mut PreemptTrapFrameKernel,
    captured_kernel_rsp: u64,
) {
    if !USING_APIC.load(Ordering::Relaxed) || crate::smp::is_bsp() {
        TICK_COUNT.fetch_add(1, Ordering::Relaxed);
        crate::time::on_timer_tick_isr();
        // See the matching note in `timer_handler_user`.
        ps2_drain_all_bytes();
    }
    // DIAGNOSTIC: see matching note in `timer_handler_user` above.
    // let period_ms: u64 = if crate::smp::is_bsp() { 1 } else { 10 };
    // crate::task::scheduler::tick_account_current_task(false, period_ms);
    crate::task::signal_reschedule();
    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    } else {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Timer as u8);
        }
    }
    // Phase 57e Bug #12 part 7 — drop timer-driven kernel-mode preemption.
    //
    // The unconditional `signal_reschedule()` above sets `reschedule = true`
    // on every 1 ms timer tick.  Under voluntary mode (no
    // `check_and_preempt_kernel` exists), the flag sits until the running
    // task transitions to user mode, where `check_and_preempt_user` consumes
    // it — kernel-mode tasks run uninterrupted, the lag-free pattern the
    // user confirmed on real hardware.
    //
    // Under preempt-full's prior shape, `check_and_preempt_kernel` consumed
    // the flag on the same tick, preempting every kernel-mode task on
    // every 1 ms boundary — even when there was no actual wake event,
    // because the flag had been set by THIS tick's `signal_reschedule`.
    // The input pipeline's typically-microsecond syscalls were getting
    // unnecessary mid-syscall context switches at every quantum, surfacing
    // as user-visible input lag (and getting strictly worse with the 4 ms
    // quantum experiment in 17099f6, reverted in 5ff8a35, because that
    // delayed the WAKEE rather than the WAKER).
    //
    // Cross-core wake delivery is preserved by `reschedule_ipi_handler_kernel`
    // (Phase 57e Track F.1), which keeps `check_and_preempt_kernel` for the
    // RescheduleIpi trigger — an IPI fires *only* when there is an actual
    // cross-core wake (`enqueue_to_core` sends `IPI_RESCHEDULE` only when
    // the target differs from the caller's core).  Same-core wakes wait
    // for the running task's natural yield (matches voluntary's behaviour).
    //
    // Trade-off: a kernel-mode task that hogs the CPU without yielding is
    // not bounded by the timer.  Same as voluntary.  All current m3OS
    // kernel-mode work yields cooperatively (IPC blocks, deadline-based
    // sleeps, syscall returns); hog-bounding can be added back as a longer
    // quantum (e.g. 100 ms) if a future workload introduces a hog.
    let _ = frame;
    let _ = captured_kernel_rsp;
}

// ---------------------------------------------------------------------------
// Keyboard scancode ring buffer
// ---------------------------------------------------------------------------
//
// There are TWO separate ring buffers:
//
//   SCANCODE_BUF  — normal TTY / kbd_server path; consumed via
//                   `read_scancode()`.  Populated when no raw-input
//                   framebuffer client is active.
//
//   RAW_SCANCODE_BUF — game input path; consumed via `read_raw_scancode()`
//                   (sys_read_scancode syscall 0x1007).  Only populated when
//                   the framebuffer owner explicitly keeps raw input enabled.
//
// Routing is exclusive: each scancode goes to exactly one buffer based on the
// raw-input policy. This prevents stale scancodes from accumulating in
// SCANCODE_BUF during gameplay and replaying when the game exits, while still
// allowing the display_server compositor to own pixels without stealing input
// from kbd_server / stdin_feeder.

const SCANCODE_BUF_SIZE: usize = 256;
// Bitmask wraparound requires a power-of-two buffer size.
const _: () = assert!(
    SCANCODE_BUF_SIZE.is_power_of_two(),
    "SCANCODE_BUF_SIZE must be a power of two for bitmask wraparound"
);

// TTY path buffer
static mut SCANCODE_BUF: [u8; SCANCODE_BUF_SIZE] = [0u8; SCANCODE_BUF_SIZE];
static SCANCODE_BUF_HEAD: AtomicUsize = AtomicUsize::new(0);
static SCANCODE_BUF_TAIL: AtomicUsize = AtomicUsize::new(0);

// Raw / game-input path buffer
static mut RAW_SCANCODE_BUF: [u8; SCANCODE_BUF_SIZE] = [0u8; SCANCODE_BUF_SIZE];
static RAW_SCANCODE_BUF_HEAD: AtomicUsize = AtomicUsize::new(0);
static RAW_SCANCODE_BUF_TAIL: AtomicUsize = AtomicUsize::new(0);
static RAW_INPUT_ROUTER: Mutex<ScancodeRouter> = Mutex::new(ScancodeRouter::new());

/// Pop one scancode from the **TTY** ring buffer, or `None` if it is empty.
#[allow(dead_code)]
pub fn read_scancode() -> Option<u8> {
    let head = SCANCODE_BUF_HEAD.load(Ordering::Acquire);
    let tail = SCANCODE_BUF_TAIL.load(Ordering::Acquire);
    if head == tail {
        return None;
    }
    // Safety: single consumer; head is only advanced here and never overtakes tail.
    let byte = unsafe { SCANCODE_BUF[head] };
    SCANCODE_BUF_HEAD.store((head + 1) & (SCANCODE_BUF_SIZE - 1), Ordering::Release);
    Some(byte)
}

/// Phase 57b G.7 — Task-context `RAW_INPUT_ROUTER` acquisition helper.
///
/// `RAW_INPUT_ROUTER` is an IRQ-shared `spin::Mutex`: `keyboard_handler`
/// (the ISR) acquires it to route each drained byte, so converting to
/// `IrqSafeMutex` would not work with the ISR's existing pattern (the ISR
/// runs with IF=0 and never raises the per-task preempt counter).  Instead,
/// every task-context reader runs with explicit
/// `preempt_disable` + `interrupts::without_interrupts` + `preempt_enable`
/// boilerplate.  This helper centralises that pattern so the two readers
/// (`read_raw_scancode` and `reset_raw_input_state`) cannot drift.
///
/// IF must be masked on the current CPU while the lock is held, otherwise a
/// same-core keyboard IRQ landing on this path while the lock is held would
/// deadlock the ISR (same bug class as the 2026-04-21 `SCHEDULER.lock`
/// post-mortem; see `docs/post-mortems/2026-04-21-scheduler-lock-isr-deadlock.md`).
///
/// Lock-ordering: `preempt_disable` is lock-free (Phase 57b D.2), so calling
/// it before `without_interrupts` cannot recurse.
fn with_raw_input_router<R>(f: impl FnOnce(&mut ScancodeRouter) -> R) -> R {
    crate::task::scheduler::preempt_disable();
    let result = x86_64::instructions::interrupts::without_interrupts(|| {
        let mut router = RAW_INPUT_ROUTER.lock();
        f(&mut router)
    });
    crate::task::scheduler::preempt_enable();
    result
}

/// Pop one scancode from the **raw / game-input** ring buffer, or `None`.
///
/// `RAW_INPUT_ROUTER` is also held by `keyboard_handler` in ISR context,
/// so task-context acquisition must run with interrupts masked on the
/// current CPU — otherwise a same-core keyboard IRQ landing here while a
/// task holds the lock deadlocks the ISR (same bug class as the 2026-04-21
/// `SCHEDULER.lock` post-mortem). See
/// `docs/post-mortems/2026-04-21-scheduler-lock-isr-deadlock.md`.
///
/// Phase 57b G.7 — `with_raw_input_router` wraps the
/// `preempt_disable` + `without_interrupts` boilerplate.
pub fn read_raw_scancode() -> Option<u8> {
    with_raw_input_router(|_router| {
        let head = RAW_SCANCODE_BUF_HEAD.load(Ordering::Acquire);
        let tail = RAW_SCANCODE_BUF_TAIL.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let byte = unsafe { RAW_SCANCODE_BUF[head] };
        RAW_SCANCODE_BUF_HEAD.store((head + 1) & (SCANCODE_BUF_SIZE - 1), Ordering::Release);
        Some(byte)
    })
}

/// Reset the raw/game-input router state and drain its ring buffer.
///
/// See [`read_raw_scancode`] for the ISR-safety rationale around
/// `RAW_INPUT_ROUTER`.
pub fn reset_raw_input_state() {
    with_raw_input_router(|router| {
        router.reset();
        RAW_SCANCODE_BUF_HEAD.store(0, Ordering::Release);
        RAW_SCANCODE_BUF_TAIL.store(0, Ordering::Release);
    });
}

/// Phase 57d follow-up — inject break codes for commonly-held keys into
/// the TTY scancode buffer so `kbd_server`'s modifier tracker and
/// repeat scheduler clear any state stuck across a fullscreen-takeover
/// session.
///
/// Background: while a takeover program (e.g. doom) owns the FB,
/// scancodes route to the RAW buffer and `kbd_server` sees nothing —
/// any release scancodes for keys held at yield-time go to the
/// takeover program. When ownership returns to `display_server`,
/// `kbd_server` resumes reading from TTY but its tracker still
/// believes those keys are pressed; the repeat scheduler then emits
/// repeat events forever (e.g. stuck Enter after exiting doom).
///
/// Called from `sys_fb_reacquire` after a successful FB reclaim so the
/// injection happens once per takeover session and does not run at
/// initial display_server boot (where there's no prior input state to
/// clear, and where pushing synthetic scancodes before kbd_server is
/// even up could only do harm).
///
/// `SCANCODE_BUF` is normally written by the keyboard ISR (single
/// producer); calling `push_to_buf` from task context introduces a
/// second producer, so we wrap with `without_interrupts` on the
/// current CPU to serialise against IRQ1 here. Other CPUs do not
/// drive the keyboard ISR (IRQ1 is steered to BSP by the APIC init),
/// so masking IF on the calling CPU is sufficient.
///
/// The break codes are make-code | 0x80 for: Enter (0x9C), LShift
/// (0xAA), RShift (0xB6), LCtrl (0x9D), LAlt (0xB8), Space (0xB9).
/// Any of these were possibly held at yield-time and won't be
/// cleared by the next user keystroke (a fresh down doesn't cancel a
/// prior down in the scheduler — it just refreshes it).
pub fn inject_release_all_held_modifiers() {
    // Order is intentional: Enter first because it's the most common
    // stuck case (user hits Enter to launch the takeover program).
    const BREAK_CODES: &[u8] = &[
        0x9C, // Enter
        0xAA, // LShift
        0xB6, // RShift
        0x9D, // LCtrl
        0xB8, // LAlt
        0xB9, // Space
    ];
    x86_64::instructions::interrupts::without_interrupts(|| {
        for &sc in BREAK_CODES {
            unsafe {
                push_to_buf(
                    (&raw mut SCANCODE_BUF).cast::<u8>(),
                    &SCANCODE_BUF_HEAD,
                    &SCANCODE_BUF_TAIL,
                    sc,
                );
            }
        }
    });
    // Wake kbd_server so it drains the synthetic releases promptly.
    crate::ipc::notification::signal_irq(1);
}

#[inline(always)]
unsafe fn push_to_buf(buf: *mut u8, head: &AtomicUsize, tail: &AtomicUsize, byte: u8) {
    let t = tail.load(Ordering::Relaxed);
    let next = (t + 1) & (SCANCODE_BUF_SIZE - 1);
    if next != head.load(Ordering::Acquire) {
        // Safety: caller guarantees `buf` points to a [u8; SCANCODE_BUF_SIZE]
        // and that this is the sole writer (single-producer ISR context).
        unsafe { buf.add(t).write(byte) };
        tail.store(next, Ordering::Release);
    }
    // else: buffer full — silently drop (prefer losing a typematic repeat
    // over blocking an interrupt handler).
}

/// Shared i8042 drain shape used by both [`keyboard_handler`] and
/// [`mouse_handler`].
///
/// ## Why both ISRs drain both byte types
///
/// The previous design had each ISR bail on the wrong byte type (kbd
/// ISR bailed on AUX bytes, mouse ISR bailed on non-AUX bytes), with
/// the assumption that the corresponding IRQ would fire and pick up
/// the stranded byte. That assumption breaks under sustained
/// keyboard activity: when a key is held and PS/2 hardware autorepeat
/// fires kbd scancodes at ~30 Hz, mouse motion bytes queue behind
/// kbd bytes in the i8042 internal FIFOs. The mouse ISR fires once,
/// reads status, sees `AUX=0` (a kbd byte at the head), bails. The
/// kbd ISR drains the kbd byte but does NOT re-trigger IRQ12 — the
/// LAPIC IRR bit for vector 44 was already acknowledged by the
/// previous mouse ISR dispatch, and the i8042 only re-asserts on the
/// next *new* AUX byte arriving from the device. The pending mouse
/// byte sits in the FIFO until either fresh mouse motion fires
/// another IRQ12 or some other side effect drains it. Visible
/// symptom: cursor freezes during a held key, jumps to current
/// position when the key releases.
///
/// Linux's i8042 driver solves this with a single combined ISR that
/// drains ALL pending bytes regardless of which IRQ fired, dispatching
/// each byte by its `STATUS_AUX_OUTPUT` bit. We adopt the same shape:
/// each ISR (still installed on its own vector) calls this helper, so
/// either IRQ will drain whatever is queued.
fn ps2_drain_all_bytes() {
    use x86_64::instructions::port::Port;

    const STATUS_OUTPUT_FULL: u8 = 1 << 0;
    const STATUS_AUX_OUTPUT: u8 = 1 << 5;
    /// Bound the per-ISR iteration count. Sized for: 16 PS/2 mouse
    /// packets (3 bytes each) + a worst-case 16-byte kbd burst, with
    /// headroom. The 8042 hardware FIFO is small (typically 16 bytes
    /// per device); this cap is comfortably above that.
    const MAX_DRAIN: usize = 64;

    let mut data_port: Port<u8> = Port::new(0x60);
    let mut status_port: Port<u8> = Port::new(0x64);

    // Fast path: peek the status port without locking. The timer ISR
    // calls this 100×/s as a backstop against IRQ12 suppression; in
    // the common case there's nothing to drain and we shouldn't pay
    // for a `RAW_INPUT_ROUTER` lock acquisition. The kbd/mouse ISRs
    // already check status indirectly by being invoked, but the
    // peek is cheap so they share the gate.
    let initial_status = unsafe { status_port.read() };
    if initial_status & STATUS_OUTPUT_FULL == 0 {
        return;
    }

    let mut got_tty_byte = false;
    let mut produced_mouse_packet = false;
    let mut raw_input_router = RAW_INPUT_ROUTER.lock();

    for _ in 0..MAX_DRAIN {
        let status = unsafe { status_port.read() };
        if status & STATUS_OUTPUT_FULL == 0 {
            break; // output buffer empty — nothing left to read
        }
        let byte = unsafe { data_port.read() };
        if status & STATUS_AUX_OUTPUT != 0 {
            // Mouse byte. `feed_byte_isr` locks `MOUSE_DECODER`
            // separately from `RAW_INPUT_ROUTER`; the lock order is
            // (RAW_INPUT_ROUTER, MOUSE_DECODER) and matches between
            // both ISRs, so no deadlock potential.
            if super::ps2::feed_byte_isr(byte) {
                produced_mouse_packet = true;
            }
            continue;
        }
        // Keyboard byte. Route through `ScancodeRouter` so multi-byte
        // prefixes (`0xE0`, `0xE1`) stay latched to the sink that
        // received their first byte even if an ownership handoff
        // happens mid-sequence.
        match raw_input_router.route_byte(byte, crate::fb::raw_input_active()) {
            ScancodeSink::Raw => unsafe {
                push_to_buf(
                    (&raw mut RAW_SCANCODE_BUF).cast::<u8>(),
                    &RAW_SCANCODE_BUF_HEAD,
                    &RAW_SCANCODE_BUF_TAIL,
                    byte,
                );
            },
            ScancodeSink::Tty => {
                unsafe {
                    push_to_buf(
                        (&raw mut SCANCODE_BUF).cast::<u8>(),
                        &SCANCODE_BUF_HEAD,
                        &SCANCODE_BUF_TAIL,
                        byte,
                    );
                }
                got_tty_byte = true;
            }
        }
    }

    // Signal once per ISR after the whole drain so neither server
    // gets a wake-up storm. Releasing the router lock before
    // signalling keeps the wait-queue path off the lock-held window.
    drop(raw_input_router);
    if got_tty_byte {
        crate::ipc::notification::signal_irq(1);
    }
    if produced_mouse_packet {
        crate::ipc::notification::signal_irq(12);
    }
}

extern "x86-interrupt" fn keyboard_handler(stack_frame: InterruptStackFrame) {
    super::ps2::IRQ1_ENTRIES.fetch_add(1, Ordering::Relaxed);
    ps2_drain_all_bytes();

    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    } else {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Keyboard as u8);
        }
    }
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

// ---------------------------------------------------------------------------
// PS/2 AUX (mouse) IRQ handler — Phase 56 Track B.2
// ---------------------------------------------------------------------------

/// IRQ12 handler. Drains pending bytes from the 8042 data port (0x60),
/// feeding each byte to `kernel-core`'s pure-logic `Ps2MouseDecoder`. When a
/// complete packet is assembled it is pushed onto the lock-free
/// `MOUSE_PACKET_RING`; userspace reads via the `sys_read_mouse_packet`
/// (0x1015) syscall.
///
/// The IRQ12 line is shared with the slave PIC; we therefore only consume
/// bytes whose status byte indicates the AUX port owns them. The 8042
/// reports this via the AUX-OUTPUT bit (status bit 5).
extern "x86-interrupt" fn mouse_handler(stack_frame: InterruptStackFrame) {
    super::ps2::IRQ12_ENTRIES.fetch_add(1, Ordering::Relaxed);
    // Both kbd and mouse bytes drain through the same helper — see the
    // doc comment on `ps2_drain_all_bytes` for why each ISR drains
    // both byte types.
    ps2_drain_all_bytes();

    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    } else {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Mouse as u8);
        }
    }
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

// ---------------------------------------------------------------------------
// APIC spurious interrupt handler
// ---------------------------------------------------------------------------

extern "x86-interrupt" fn spurious_handler(stack_frame: InterruptStackFrame) {
    // Spurious interrupt (vector 0xFF) — no EOI must be sent.
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

// ---------------------------------------------------------------------------
// SMP IPI handlers (Phase 25)
// ---------------------------------------------------------------------------

/// Phase 57d Track B — reschedule IPI handler, user (ring 3) path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reschedule_ipi_handler_user(frame: &mut PreemptTrapFrameUser) {
    if let Some(pc) = crate::smp::try_per_core() {
        let n = pc
            .ipi_recv_log_budget
            .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            log::debug!("[ipi] reschedule received core={}", pc.core_id);
        }
    }
    crate::task::signal_reschedule();
    maybe_redirect_group_exit_trampoline_user(frame);
    super::apic::lapic_eoi();
    // Phase 57e Bug #9 — clamp preempt_count to 0 in release builds; panic on
    // non-zero in debug builds.  Helper handles the mode split.
    crate::task::scheduler::assert_preempt_count_zero_at_user_return();
    #[cfg(feature = "preempt-voluntary")]
    unsafe {
        check_and_preempt_user(frame, PreemptTrigger::RescheduleIpi);
    }
}

/// Phase 57d Track B — reschedule IPI handler, kernel path.
///
/// Phase 57e deferral cleanup (2026-05-07): the `cfg(preempt-full)` branch
/// that called `check_and_preempt_kernel` was removed.  An IPI delivered
/// to a kernel-mode core now sets `reschedule = true` (via
/// `signal_reschedule()`) and returns; the flag is consumed at the next
/// user-mode return boundary (`check_and_preempt_user`) or at the
/// running task's next cooperative yield/block point.  This matches
/// Phase 57d voluntary's behaviour and is what the 2026-05-07 post-mortem
/// concludes is the right model for m3OS's microkernel architecture.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn reschedule_ipi_handler_kernel(
    frame: &mut PreemptTrapFrameKernel,
    captured_kernel_rsp: u64,
) {
    if let Some(pc) = crate::smp::try_per_core() {
        let n = pc
            .ipi_recv_log_budget
            .fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
        if n > 0 {
            log::debug!("[ipi] reschedule received core={}", pc.core_id);
        }
    }
    crate::task::signal_reschedule();
    super::apic::lapic_eoi();
    let _ = frame;
    let _ = captured_kernel_rsp;
}

/// TLB shootdown IPI handler (vector 0xFD).
///
/// Invalidates a specific page on this core's TLB. The target address and
/// synchronization are managed by the TLB shootdown request in `smp::tlb`.
extern "x86-interrupt" fn tlb_shootdown_ipi_handler(stack_frame: InterruptStackFrame) {
    crate::smp::tlb::handle_tlb_shootdown_ipi();
    super::apic::lapic_eoi();
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

/// Allocator-local cache drain IPI handler (vector 0xFC).
///
/// Flushes this core's per-CPU page cache when a page-cache drain round is
/// active and also services slab-local reclaim handshakes when requested. The
/// handler always runs on the owning core, so mutating CPU-local cache state is
/// safe.
extern "x86-interrupt" fn cache_drain_ipi_handler(stack_frame: InterruptStackFrame) {
    crate::mm::frame_allocator::handle_cache_drain_ipi();
    super::apic::lapic_eoi();
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

// ---------------------------------------------------------------------------
// Serial (COM1) IRQ handler — vector 36
// ---------------------------------------------------------------------------

extern "x86-interrupt" fn serial_handler(stack_frame: InterruptStackFrame) {
    crate::serial::handle_serial_irq();

    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    } else {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Serial as u8);
        }
    }
    assert_preempt_count_zero_on_return_to_user(&stack_frame);
}

// ---------------------------------------------------------------------------
// Phase 55 C.3 — device IRQ contract
// ---------------------------------------------------------------------------
//
// Drivers register MSI / MSI-X / legacy-INTx handlers via
// [`crate::pci::register_device_irq`]. Each registered vector walks through a
// pre-declared stub (see `device_irq_stub_N` below) which dispatches to the
// installed handler and sends EOI. Handlers run in ISR context and must obey
// the ISR contract: no allocation, no blocking, no IPC. The expected body is
// "read/ack a device register, signal a wait queue via `wake_task`, return."
//
// We reserve a bank of 16 consecutive IDT vectors starting at
// [`DEVICE_IRQ_VECTOR_BASE`]. That is enough for the virtio + NVMe + e1000
// targets this phase adds. If a driver asks for more vectors than are
// available, registration returns `None` and the driver is expected to fall
// back to legacy INTx routing or fail init.

/// Base IDT vector for device MSI / MSI-X handlers.
///
/// Must match the `MSI_VECTOR_BASE` used by the kernel-side MSI pool so the
/// allocated vector numbers land on installed IDT stubs. The `+ 0x10` gap
/// above the existing 0x60 baseline leaves room for the PIC/IPI block.
pub const DEVICE_IRQ_VECTOR_BASE: u8 = 0x60;

/// Number of device IRQ slots covered by the stub bank.
pub const DEVICE_IRQ_VECTOR_COUNT: u8 = 16;

/// Entry in the device IRQ dispatch table.
pub struct DeviceIrqEntry {
    /// Driver-supplied handler. Runs in ISR context.
    pub handler: fn(),
    /// IRQ kind — legacy INTx handlers gate on ISR status; MSI/MSI-X skip it.
    pub kind: DeviceIrqKind,
}

/// What kind of interrupt this is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceIrqKind {
    /// Legacy INTx (level-triggered, potentially shared). Handler contract:
    /// check the device's ISR status register and return early if this
    /// interrupt is not for you.
    LegacyIntx,
    /// MSI or MSI-X (vector-specific, not shared).
    Msi,
}

/// Installed handlers, keyed by vector offset from [`DEVICE_IRQ_VECTOR_BASE`].
///
/// Written rarely (device init), read on every matching IRQ. Guarded by a
/// spin mutex; the ISR path uses a copy snapshot outside the lock.
static DEVICE_IRQ_TABLE: Mutex<[Option<DeviceIrqEntry>; DEVICE_IRQ_VECTOR_COUNT as usize]> =
    Mutex::new([const { None }; DEVICE_IRQ_VECTOR_COUNT as usize]);

/// Install `entry` at `vector`. Returns `Err` if the vector is outside the
/// device-IRQ bank or already occupied.
///
/// The critical section runs with interrupts disabled so an MSI/MSI-X vector
/// firing on this CPU cannot re-enter `dispatch_device_irq` and deadlock on
/// `DEVICE_IRQ_TABLE`.
pub fn register_device_irq(vector: u8, entry: DeviceIrqEntry) -> Result<(), &'static str> {
    if !(DEVICE_IRQ_VECTOR_BASE..DEVICE_IRQ_VECTOR_BASE + DEVICE_IRQ_VECTOR_COUNT).contains(&vector)
    {
        return Err("vector out of device IRQ range");
    }
    let idx = (vector - DEVICE_IRQ_VECTOR_BASE) as usize;
    // Phase 57b G.8 — `DEVICE_IRQ_TABLE` is `explicit-preempt-and-cli` per
    // Track A.1 audit (IRQ-shared via `dispatch_device_irq`, called from
    // every `device_irq_stub_*` ISR). Pair `preempt_disable` /
    // `preempt_enable` with the existing `without_interrupts` wrap so the
    // F.1 preempt-discipline stays balanced. `preempt_disable` is
    // lock-free (Phase 57b D.2), so calling it before
    // `without_interrupts` cannot recurse.
    crate::task::scheduler::preempt_disable();
    let result = x86_64::instructions::interrupts::without_interrupts(|| {
        let mut tbl = DEVICE_IRQ_TABLE.lock();
        if tbl[idx].is_some() {
            return Err("device IRQ vector already registered");
        }
        tbl[idx] = Some(entry);
        Ok(())
    });
    crate::task::scheduler::preempt_enable();
    result
}

/// Remove the handler installed at `vector`. Silently ignores missing entries.
///
/// The critical section runs with interrupts disabled for the same reason
/// as `register_device_irq` — the dispatch path locks the same table from
/// ISR context.
#[allow(dead_code)]
pub fn unregister_device_irq(vector: u8) {
    if !(DEVICE_IRQ_VECTOR_BASE..DEVICE_IRQ_VECTOR_BASE + DEVICE_IRQ_VECTOR_COUNT).contains(&vector)
    {
        return;
    }
    let idx = (vector - DEVICE_IRQ_VECTOR_BASE) as usize;
    // Phase 57b G.8 — pair `preempt_disable` / `preempt_enable` around the
    // existing `without_interrupts` wrap so the F.1 preempt-discipline
    // stays balanced. See `register_device_irq` for the lock-classification
    // rationale.
    crate::task::scheduler::preempt_disable();
    x86_64::instructions::interrupts::without_interrupts(|| {
        let mut tbl = DEVICE_IRQ_TABLE.lock();
        tbl[idx] = None;
    });
    crate::task::scheduler::preempt_enable();
}

/// Dispatch a device IRQ to its registered handler. Runs in ISR context.
///
/// Snapshots the handler pointer under the lock, then releases the lock
/// before invoking so the handler itself can (for example) call
/// `register_device_irq` for a sibling queue without reentering.
///
/// Phase 57b D.3: also runs the user-mode-return preempt_count assertion
/// when `iretq` will return to ring 3 (i.e. the IRQ interrupted user
/// mode).  All 16 device-IRQ stubs share this dispatch path, so a single
/// gate here covers every device IRQ — DRY-clean per the Engineering
/// Practice Gates.
#[inline(always)]
fn dispatch_device_irq(vector: u8, stack_frame: &InterruptStackFrame) {
    let idx = (vector - DEVICE_IRQ_VECTOR_BASE) as usize;
    let snapshot: Option<(fn(), DeviceIrqKind)> = {
        let tbl = DEVICE_IRQ_TABLE.lock();
        tbl[idx].as_ref().map(|e| (e.handler, e.kind))
    };
    if let Some((h, _kind)) = snapshot {
        h();
    }
    // Always EOI, even if no handler — spurious interrupts must not stall the
    // APIC.
    if USING_APIC.load(Ordering::Relaxed) {
        super::apic::lapic_eoi();
    }
    assert_preempt_count_zero_on_return_to_user(stack_frame);
}

/// Test-only entry point into the device-IRQ dispatcher.
///
/// The Phase 55b Track B.4 `#[test_case]` harness needs to drive the exact
/// ISR shim the hardware will invoke without programming an MSI capability
/// (which is impossible from the `test_main` runner's PID). This re-exports
/// [`dispatch_device_irq`] under a test-only name so the unit test can
/// deliver a synthetic IRQ and observe the same `notification::signal_irq_bit`
/// side effect. The function is `#[cfg(test)]`-gated so it does not ship in
/// release builds.
///
/// Synthesises a fresh `InterruptStackFrame` on the stack so the
/// dispatch helper has a real reference to forward.  Tests run in ring 0
/// (the kernel test harness boots before any userspace task), so the
/// frame's CS naturally reflects ring 0 and the user-mode-return
/// assertion is a no-op for the test path.
#[cfg(test)]
pub fn dispatch_device_irq_for_test(vector: u8) {
    // Build a synthetic `InterruptStackFrame` whose CS encodes ring 0 —
    // the user-mode-return assertion gate will skip it.  We never
    // `iretq` through this frame; it exists only to satisfy
    // `dispatch_device_irq`'s signature.
    let frame = InterruptStackFrame::new(
        VirtAddr::new(0),
        gdt::kernel_code_selector(),
        x86_64::registers::rflags::RFlags::empty(),
        VirtAddr::new(0),
        gdt::kernel_data_selector(),
    );
    dispatch_device_irq(vector, &frame);
}

// Stubs — one per vector slot. The IDT requires a real
// `extern "x86-interrupt"` function at each vector; we cannot generate them
// at runtime. Each stub thunks to `dispatch_device_irq` with a compile-time
// vector number.
extern "x86-interrupt" fn device_irq_stub_0(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_1(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 1, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_2(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 2, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_3(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 3, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_4(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 4, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_5(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 5, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_6(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 6, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_7(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 7, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_8(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 8, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_9(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 9, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_10(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 10, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_11(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 11, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_12(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 12, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_13(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 13, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_14(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 14, &stack_frame);
}
extern "x86-interrupt" fn device_irq_stub_15(stack_frame: InterruptStackFrame) {
    dispatch_device_irq(DEVICE_IRQ_VECTOR_BASE + 15, &stack_frame);
}
