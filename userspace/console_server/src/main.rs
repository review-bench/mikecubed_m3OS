//! Userspace console service for m3OS (Phase 52).
//!
//! Maps the framebuffer into its address space, creates an IPC endpoint,
//! registers as the "console" service, and serves CONSOLE_WRITE requests
//! by rendering text (with ANSI escape sequence support) onto the
//! framebuffer.  Also echoes output to serial via stdout.
#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

extern crate alloc;

use core::alloc::Layout;
use kernel_core::fb::{AnsiParser, ConsoleCmd, SgrParams};
use syscall_lib::STDOUT_FILENO;
use syscall_lib::heap::BrkAllocator;

#[global_allocator]
static ALLOCATOR: BrkAllocator = BrkAllocator::new();

#[alloc_error_handler]
fn alloc_error(_layout: Layout) -> ! {
    syscall_lib::write_str(STDOUT_FILENO, "console_server: alloc error\n");
    syscall_lib::exit(99)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

syscall_lib::entry_point!(program_main);

fn program_main(_args: &[&str]) -> i32 {
    syscall_lib::write_str(STDOUT_FILENO, "console_server: starting\n");

    // Phase 52 transitional: skip framebuffer_mmap because it calls
    // try_yield_console which suppresses ALL kernel framebuffer text output
    // (login prompt, shell, everything).  The kernel's fb console still
    // handles sys_write rendering.  The userspace console_server runs as an
    // IPC-only service: CONSOLE_WRITE payloads are echoed to stdout which
    // the kernel renders to both serial and framebuffer.
    //
    // Full framebuffer takeover will be enabled in a later phase when
    // sys_write(STDOUT) is routed through IPC to this service.

    // 1. Create an IPC endpoint.
    let ep_handle = syscall_lib::create_endpoint();
    if ep_handle == u64::MAX {
        syscall_lib::write_str(STDOUT_FILENO, "console_server: create_endpoint failed\n");
        return 1;
    }
    let ep_handle = ep_handle as u32;

    // 2. Register as "console" service (takes over from kernel entry).
    let ret = syscall_lib::ipc_register_service(ep_handle, "console");
    if ret == u64::MAX {
        syscall_lib::write_str(STDOUT_FILENO, "console_server: register_service failed\n");
        return 1;
    }

    syscall_lib::write_str(STDOUT_FILENO, "console_server: registered as 'console'\n");
    syscall_lib::write_str(STDOUT_FILENO, "console_server: ready\n");

    // 3. Enter the IPC server loop (no framebuffer renderer — stdout only).
    server_loop_stdout(ep_handle);
}

// ---------------------------------------------------------------------------
// IPC protocol constants
// ---------------------------------------------------------------------------

/// CONSOLE_WRITE message label.
const CONSOLE_WRITE: u64 = 0;

/// Maximum bytes per CONSOLE_WRITE request.
const MAX_CONSOLE_WRITE_LEN: usize = 4096;

/// Reply cap handle — the kernel inserts the one-shot reply cap at handle 1
/// after each successful `ipc_recv`.
const REPLY_CAP_HANDLE: u32 = 1;

// ---------------------------------------------------------------------------
// Server loop
// ---------------------------------------------------------------------------

/// Transitional server loop: receives CONSOLE_WRITE IPC and echoes the
/// payload to stdout.  The kernel renders stdout to both serial and
/// framebuffer, so text appears in both places without needing a direct
/// framebuffer mapping.
fn server_loop_stdout(ep_handle: u32) -> ! {
    let mut msg = syscall_lib::IpcMessage::new(0);
    let mut buf = [0u8; MAX_CONSOLE_WRITE_LEN];

    // First receive — blocks until a client sends a message.
    syscall_lib::ipc_recv_msg(ep_handle, &mut msg, &mut buf);

    loop {
        let reply_label = match msg.label {
            CONSOLE_WRITE => {
                let len = (msg.data[1] as usize).min(MAX_CONSOLE_WRITE_LEN);
                if len > 0 {
                    if let Ok(text) = core::str::from_utf8(&buf[..len]) {
                        syscall_lib::write_str(STDOUT_FILENO, text);
                    }
                }
                0
            }
            _ => u64::MAX,
        };

        // Reply and wait for the next message.
        msg = syscall_lib::IpcMessage::new(0);
        syscall_lib::ipc_reply_recv_msg(
            REPLY_CAP_HANDLE,
            reply_label,
            ep_handle,
            &mut msg,
            &mut buf,
        );
    }
}

#[allow(dead_code)]
fn server_loop(renderer: &mut FbRenderer, ep_handle: u32) -> ! {
    let mut msg = syscall_lib::IpcMessage::new(0);
    let mut buf = [0u8; MAX_CONSOLE_WRITE_LEN];

    syscall_lib::ipc_recv_msg(ep_handle, &mut msg, &mut buf);

    loop {
        let reply_label = match msg.label {
            CONSOLE_WRITE => {
                let len = (msg.data[1] as usize).min(MAX_CONSOLE_WRITE_LEN);
                handle_console_write(renderer, &buf[..len])
            }
            _ => u64::MAX,
        };

        msg = syscall_lib::IpcMessage::new(0);
        syscall_lib::ipc_reply_recv_msg(
            REPLY_CAP_HANDLE,
            reply_label,
            ep_handle,
            &mut msg,
            &mut buf,
        );
    }
}

/// Handle a CONSOLE_WRITE message by rendering the payload to the
/// framebuffer and echoing it to serial via stdout.
fn handle_console_write(renderer: &mut FbRenderer, data: &[u8]) -> u64 {
    if data.is_empty() {
        return u64::MAX;
    }
    // Validate UTF-8.  Non-UTF-8 payloads are rejected.
    if let Ok(text) = core::str::from_utf8(data) {
        renderer.write_str(text);
        syscall_lib::write_str(STDOUT_FILENO, text);
        0
    } else {
        u64::MAX
    }
}

// ---------------------------------------------------------------------------
// Framebuffer info
// ---------------------------------------------------------------------------

/// Packed framebuffer info struct matching the kernel's FbInfo layout.
#[repr(C)]
struct FbInfo {
    width: u32,
    height: u32,
    stride: u32,
    bpp: u32,
    pixel_format: u32,
}

fn get_fb_info() -> Option<FbInfo> {
    let mut buf = [0u8; 20];
    let ret = syscall_lib::framebuffer_info(&mut buf);
    if ret != 0 {
        return None;
    }
    // SAFETY: buf is 20 bytes, matching the FbInfo struct layout.
    let info = unsafe { core::ptr::read_unaligned(buf.as_ptr() as *const FbInfo) };
    Some(info)
}

/// Check if a syscall return value is an error (negative errno encoded as u64).
fn is_error(val: u64) -> bool {
    val > u64::MAX - 4096
}

// ---------------------------------------------------------------------------
// Framebuffer renderer
// ---------------------------------------------------------------------------

/// 8x16 VGA font constants.
const CHAR_W: usize = 8;
const CHAR_H: usize = 16;
const FONT_FIRST: u8 = 0x20;
const FONT_LAST: u8 = 0x7E;
#[allow(dead_code)]
const FONT_GLYPHS: usize = (FONT_LAST - FONT_FIRST + 1) as usize; // 95

/// Placeholder glyph for characters outside the printable range.
const PLACEHOLDER: [u8; CHAR_H] = [0xFF; CHAR_H];

/// IBM VGA 8x16 font data for ASCII 0x20-0x7E.
#[rustfmt::skip]
static FONT: [[u8; CHAR_H]; 95] = [
    // 0x20 ' '
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x21 '!'
    [0x00,0x00,0x18,0x3C,0x3C,0x3C,0x18,0x18,0x18,0x00,0x18,0x18,0x00,0x00,0x00,0x00],
    // 0x22 '"'
    [0x00,0x00,0x66,0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x23 '#'
    [0x00,0x00,0x36,0x36,0x7F,0x36,0x36,0x36,0x7F,0x36,0x36,0x36,0x00,0x00,0x00,0x00],
    // 0x24 '$'
    [0x00,0x00,0x0C,0x0C,0x3E,0x63,0x61,0x60,0x3E,0x03,0x43,0x63,0x3E,0x0C,0x0C,0x00],
    // 0x25 '%'
    [0x00,0x00,0x00,0x00,0x00,0x61,0x63,0x06,0x0C,0x18,0x33,0x63,0x00,0x00,0x00,0x00],
    // 0x26 '&'
    [0x00,0x00,0x1C,0x36,0x36,0x1C,0x3B,0x6E,0x66,0x66,0x66,0x3B,0x00,0x00,0x00,0x00],
    // 0x27 '\''
    [0x00,0x00,0x0C,0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x28 '('
    [0x00,0x00,0x06,0x0C,0x18,0x18,0x18,0x18,0x18,0x18,0x0C,0x06,0x00,0x00,0x00,0x00],
    // 0x29 ')'
    [0x00,0x00,0x30,0x18,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x18,0x30,0x00,0x00,0x00,0x00],
    // 0x2A '*'
    [0x00,0x00,0x00,0x00,0x00,0x36,0x1C,0x7F,0x1C,0x36,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x2B '+'
    [0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x2C ','
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x18,0x30,0x00,0x00,0x00],
    // 0x2D '-'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x2E '.'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00],
    // 0x2F '/'
    [0x00,0x00,0x00,0x00,0x03,0x06,0x06,0x0C,0x0C,0x18,0x30,0x30,0x00,0x00,0x00,0x00],
    // 0x30 '0'
    [0x00,0x00,0x3E,0x63,0x63,0x63,0x6B,0x6B,0x63,0x63,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x31 '1'
    [0x00,0x00,0x0C,0x1C,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3F,0x00,0x00,0x00,0x00],
    // 0x32 '2'
    [0x00,0x00,0x3E,0x63,0x03,0x06,0x0C,0x18,0x30,0x61,0x63,0x7F,0x00,0x00,0x00,0x00],
    // 0x33 '3'
    [0x00,0x00,0x3E,0x63,0x03,0x03,0x1E,0x03,0x03,0x03,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x34 '4'
    [0x00,0x00,0x06,0x0E,0x1E,0x36,0x66,0x66,0x7F,0x06,0x06,0x0F,0x00,0x00,0x00,0x00],
    // 0x35 '5'
    [0x00,0x00,0x7F,0x60,0x60,0x60,0x7E,0x03,0x03,0x03,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x36 '6'
    [0x00,0x00,0x1C,0x30,0x60,0x60,0x7E,0x63,0x63,0x63,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x37 '7'
    [0x00,0x00,0x7F,0x63,0x03,0x06,0x06,0x0C,0x0C,0x18,0x18,0x18,0x00,0x00,0x00,0x00],
    // 0x38 '8'
    [0x00,0x00,0x3E,0x63,0x63,0x63,0x3E,0x63,0x63,0x63,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x39 '9'
    [0x00,0x00,0x3E,0x63,0x63,0x63,0x63,0x3F,0x03,0x03,0x06,0x3C,0x00,0x00,0x00,0x00],
    // 0x3A ':'
    [0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x00,0x00],
    // 0x3B ';'
    [0x00,0x00,0x00,0x00,0x18,0x18,0x00,0x00,0x00,0x18,0x18,0x30,0x00,0x00,0x00,0x00],
    // 0x3C '<'
    [0x00,0x00,0x00,0x06,0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x06,0x00,0x00,0x00,0x00],
    // 0x3D '='
    [0x00,0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x7E,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x3E '>'
    [0x00,0x00,0x00,0x60,0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x60,0x00,0x00,0x00,0x00],
    // 0x3F '?'
    [0x00,0x00,0x3E,0x63,0x63,0x06,0x0C,0x0C,0x0C,0x00,0x0C,0x0C,0x00,0x00,0x00,0x00],
    // 0x40 '@'
    [0x00,0x00,0x3E,0x63,0x63,0x6F,0x6B,0x6B,0x6F,0x60,0x60,0x3E,0x00,0x00,0x00,0x00],
    // 0x41 'A'
    [0x00,0x00,0x08,0x1C,0x36,0x63,0x63,0x7F,0x63,0x63,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x42 'B'
    [0x00,0x00,0x7E,0x33,0x33,0x33,0x3E,0x33,0x33,0x33,0x33,0x7E,0x00,0x00,0x00,0x00],
    // 0x43 'C'
    [0x00,0x00,0x1E,0x33,0x61,0x60,0x60,0x60,0x60,0x61,0x33,0x1E,0x00,0x00,0x00,0x00],
    // 0x44 'D'
    [0x00,0x00,0x7C,0x36,0x33,0x33,0x33,0x33,0x33,0x33,0x36,0x7C,0x00,0x00,0x00,0x00],
    // 0x45 'E'
    [0x00,0x00,0x7F,0x33,0x31,0x34,0x3C,0x34,0x30,0x31,0x33,0x7F,0x00,0x00,0x00,0x00],
    // 0x46 'F'
    [0x00,0x00,0x7F,0x33,0x31,0x34,0x3C,0x34,0x30,0x30,0x30,0x78,0x00,0x00,0x00,0x00],
    // 0x47 'G'
    [0x00,0x00,0x1E,0x33,0x61,0x60,0x60,0x6F,0x63,0x63,0x37,0x1D,0x00,0x00,0x00,0x00],
    // 0x48 'H'
    [0x00,0x00,0x63,0x63,0x63,0x63,0x7F,0x63,0x63,0x63,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x49 'I'
    [0x00,0x00,0x3C,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00],
    // 0x4A 'J'
    [0x00,0x00,0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x6C,0x6C,0x38,0x00,0x00,0x00,0x00],
    // 0x4B 'K'
    [0x00,0x00,0x67,0x33,0x36,0x36,0x3C,0x36,0x36,0x33,0x33,0x67,0x00,0x00,0x00,0x00],
    // 0x4C 'L'
    [0x00,0x00,0x78,0x30,0x30,0x30,0x30,0x30,0x30,0x31,0x33,0x7F,0x00,0x00,0x00,0x00],
    // 0x4D 'M'
    [0x00,0x00,0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x63,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x4E 'N'
    [0x00,0x00,0x63,0x73,0x7B,0x7F,0x6F,0x67,0x63,0x63,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x4F 'O'
    [0x00,0x00,0x1C,0x36,0x63,0x63,0x63,0x63,0x63,0x63,0x36,0x1C,0x00,0x00,0x00,0x00],
    // 0x50 'P'
    [0x00,0x00,0x7E,0x33,0x33,0x33,0x3E,0x30,0x30,0x30,0x30,0x78,0x00,0x00,0x00,0x00],
    // 0x51 'Q'
    [0x00,0x00,0x1C,0x36,0x63,0x63,0x63,0x63,0x6F,0x6B,0x36,0x1D,0x00,0x00,0x00,0x00],
    // 0x52 'R'
    [0x00,0x00,0x7E,0x33,0x33,0x33,0x3E,0x36,0x33,0x33,0x33,0x73,0x00,0x00,0x00,0x00],
    // 0x53 'S'
    [0x00,0x00,0x1E,0x33,0x33,0x30,0x1C,0x06,0x03,0x33,0x33,0x1E,0x00,0x00,0x00,0x00],
    // 0x54 'T'
    [0x00,0x00,0xFF,0xDB,0x99,0x18,0x18,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00],
    // 0x55 'U'
    [0x00,0x00,0x63,0x63,0x63,0x63,0x63,0x63,0x63,0x63,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x56 'V'
    [0x00,0x00,0x63,0x63,0x63,0x63,0x63,0x63,0x63,0x36,0x1C,0x08,0x00,0x00,0x00,0x00],
    // 0x57 'W'
    [0x00,0x00,0x63,0x63,0x63,0x63,0x6B,0x6B,0x7F,0x77,0x63,0x41,0x00,0x00,0x00,0x00],
    // 0x58 'X'
    [0x00,0x00,0x63,0x63,0x36,0x36,0x1C,0x1C,0x36,0x36,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x59 'Y'
    [0x00,0x00,0xC3,0xC3,0xC3,0x66,0x3C,0x18,0x18,0x18,0x18,0x3C,0x00,0x00,0x00,0x00],
    // 0x5A 'Z'
    [0x00,0x00,0x7F,0x63,0x43,0x06,0x0C,0x18,0x30,0x61,0x63,0x7F,0x00,0x00,0x00,0x00],
    // 0x5B '['
    [0x00,0x00,0x3C,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x30,0x3C,0x00,0x00,0x00,0x00],
    // 0x5C '\'
    [0x00,0x00,0x00,0x00,0x60,0x30,0x30,0x18,0x0C,0x0C,0x06,0x00,0x00,0x00,0x00,0x00],
    // 0x5D ']'
    [0x00,0x00,0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00,0x00,0x00,0x00],
    // 0x5E '^'
    [0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x5F '_'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00,0x00,0x00],
    // 0x60 '`'
    [0x00,0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    // 0x61 'a'
    [0x00,0x00,0x00,0x00,0x00,0x3E,0x03,0x03,0x3F,0x63,0x63,0x3F,0x00,0x00,0x00,0x00],
    // 0x62 'b'
    [0x00,0x00,0x70,0x30,0x30,0x3E,0x33,0x33,0x33,0x33,0x33,0x6E,0x00,0x00,0x00,0x00],
    // 0x63 'c'
    [0x00,0x00,0x00,0x00,0x00,0x1E,0x33,0x60,0x60,0x60,0x33,0x1E,0x00,0x00,0x00,0x00],
    // 0x64 'd'
    [0x00,0x00,0x0E,0x06,0x06,0x1E,0x36,0x66,0x66,0x66,0x66,0x3B,0x00,0x00,0x00,0x00],
    // 0x65 'e'
    [0x00,0x00,0x00,0x00,0x00,0x3E,0x63,0x63,0x7F,0x60,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x66 'f'
    [0x00,0x00,0x1C,0x36,0x30,0x30,0x7C,0x30,0x30,0x30,0x30,0x78,0x00,0x00,0x00,0x00],
    // 0x67 'g'
    [0x00,0x00,0x00,0x00,0x00,0x3B,0x66,0x66,0x66,0x66,0x3E,0x06,0x66,0x3C,0x00,0x00],
    // 0x68 'h'
    [0x00,0x00,0x70,0x30,0x30,0x36,0x3B,0x33,0x33,0x33,0x33,0x73,0x00,0x00,0x00,0x00],
    // 0x69 'i'
    [0x00,0x00,0x0C,0x0C,0x00,0x1C,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00,0x00,0x00,0x00],
    // 0x6A 'j'
    [0x00,0x00,0x06,0x06,0x00,0x0E,0x06,0x06,0x06,0x06,0x06,0x66,0x66,0x3C,0x00,0x00],
    // 0x6B 'k'
    [0x00,0x00,0x70,0x30,0x30,0x33,0x36,0x3C,0x38,0x3C,0x36,0x73,0x00,0x00,0x00,0x00],
    // 0x6C 'l'
    [0x00,0x00,0x1C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00,0x00,0x00,0x00],
    // 0x6D 'm'
    [0x00,0x00,0x00,0x00,0x00,0x6B,0x7F,0x7F,0x7F,0x6B,0x63,0x63,0x00,0x00,0x00,0x00],
    // 0x6E 'n'
    [0x00,0x00,0x00,0x00,0x00,0x6E,0x33,0x33,0x33,0x33,0x33,0x33,0x00,0x00,0x00,0x00],
    // 0x6F 'o'
    [0x00,0x00,0x00,0x00,0x00,0x1E,0x33,0x33,0x33,0x33,0x33,0x1E,0x00,0x00,0x00,0x00],
    // 0x70 'p'
    [0x00,0x00,0x00,0x00,0x00,0x6E,0x33,0x33,0x33,0x33,0x3E,0x30,0x30,0x78,0x00,0x00],
    // 0x71 'q'
    [0x00,0x00,0x00,0x00,0x00,0x3B,0x66,0x66,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00,0x00],
    // 0x72 'r'
    [0x00,0x00,0x00,0x00,0x00,0x6E,0x3B,0x33,0x30,0x30,0x30,0x78,0x00,0x00,0x00,0x00],
    // 0x73 's'
    [0x00,0x00,0x00,0x00,0x00,0x3E,0x63,0x60,0x3E,0x03,0x63,0x3E,0x00,0x00,0x00,0x00],
    // 0x74 't'
    [0x00,0x00,0x08,0x18,0x18,0x7E,0x18,0x18,0x18,0x18,0x1A,0x0C,0x00,0x00,0x00,0x00],
    // 0x75 'u'
    [0x00,0x00,0x00,0x00,0x00,0x63,0x63,0x63,0x63,0x63,0x67,0x3B,0x00,0x00,0x00,0x00],
    // 0x76 'v'
    [0x00,0x00,0x00,0x00,0x00,0x63,0x63,0x63,0x63,0x36,0x1C,0x08,0x00,0x00,0x00,0x00],
    // 0x77 'w'
    [0x00,0x00,0x00,0x00,0x00,0x63,0x63,0x6B,0x6B,0x7F,0x77,0x63,0x00,0x00,0x00,0x00],
    // 0x78 'x'
    [0x00,0x00,0x00,0x00,0x00,0x63,0x36,0x1C,0x1C,0x1C,0x36,0x63,0x00,0x00,0x00,0x00],
    // 0x79 'y'
    [0x00,0x00,0x00,0x00,0x00,0x63,0x63,0x63,0x63,0x63,0x3F,0x03,0x06,0x3C,0x00,0x00],
    // 0x7A 'z'
    [0x00,0x00,0x00,0x00,0x00,0x7F,0x33,0x06,0x0C,0x18,0x31,0x7F,0x00,0x00,0x00,0x00],
    // 0x7B '{'
    [0x00,0x00,0x0E,0x18,0x18,0x18,0x70,0x18,0x18,0x18,0x18,0x0E,0x00,0x00,0x00,0x00],
    // 0x7C '|'
    [0x00,0x00,0x18,0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x18,0x18,0x00,0x00,0x00,0x00],
    // 0x7D '}'
    [0x00,0x00,0x70,0x18,0x18,0x18,0x0E,0x18,0x18,0x18,0x18,0x70,0x00,0x00,0x00,0x00],
    // 0x7E '~'
    [0x00,0x00,0x3B,0x6E,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
];

/// Pixel colour (r, g, b).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Colour {
    r: u8,
    g: u8,
    b: u8,
}

const FG: Colour = Colour {
    r: 0xFF,
    g: 0xFF,
    b: 0xFF,
};
const BG: Colour = Colour {
    r: 0x00,
    g: 0x00,
    b: 0x00,
};

/// Standard VGA color palette (SGR 30-37 / 40-47).
const VGA_COLORS: [Colour; 8] = [
    Colour {
        r: 0x00,
        g: 0x00,
        b: 0x00,
    }, // Black
    Colour {
        r: 0xAA,
        g: 0x00,
        b: 0x00,
    }, // Red
    Colour {
        r: 0x00,
        g: 0xAA,
        b: 0x00,
    }, // Green
    Colour {
        r: 0xAA,
        g: 0x55,
        b: 0x00,
    }, // Yellow/Brown
    Colour {
        r: 0x00,
        g: 0x00,
        b: 0xAA,
    }, // Blue
    Colour {
        r: 0xAA,
        g: 0x00,
        b: 0xAA,
    }, // Magenta
    Colour {
        r: 0x00,
        g: 0xAA,
        b: 0xAA,
    }, // Cyan
    Colour {
        r: 0xAA,
        g: 0xAA,
        b: 0xAA,
    }, // White (light gray)
];

/// Bright VGA color palette (SGR 90-97 / 100-107).
const VGA_BRIGHT_COLORS: [Colour; 8] = [
    Colour {
        r: 0x55,
        g: 0x55,
        b: 0x55,
    }, // Bright Black
    Colour {
        r: 0xFF,
        g: 0x55,
        b: 0x55,
    }, // Bright Red
    Colour {
        r: 0x55,
        g: 0xFF,
        b: 0x55,
    }, // Bright Green
    Colour {
        r: 0xFF,
        g: 0xFF,
        b: 0x55,
    }, // Bright Yellow
    Colour {
        r: 0x55,
        g: 0x55,
        b: 0xFF,
    }, // Bright Blue
    Colour {
        r: 0xFF,
        g: 0x55,
        b: 0xFF,
    }, // Bright Magenta
    Colour {
        r: 0x55,
        g: 0xFF,
        b: 0xFF,
    }, // Bright Cyan
    Colour {
        r: 0xFF,
        g: 0xFF,
        b: 0xFF,
    }, // Bright White
];

/// Pixel format: 0 = RGB, 1 = BGR (matches kernel FbInfo encoding).
const PIXEL_FORMAT_RGB: u32 = 0;
const PIXEL_FORMAT_BGR: u32 = 1;

/// Userspace framebuffer renderer with ANSI escape sequence support.
struct FbRenderer {
    buf: *mut u8,
    byte_len: usize,
    width: usize,
    height: usize,
    stride: usize,
    bytes_per_pixel: usize,
    pixel_format: u32,
    cursor_col: usize,
    cursor_row: usize,
    parser: AnsiParser,
    fg_color: Colour,
    bg_color: Colour,
    cursor_visible: bool,
    cursor_rendered: bool,
}

// SAFETY: FbRenderer holds a raw pointer to the framebuffer mapping in our
// address space. It is only used from a single-threaded server loop.
unsafe impl Send for FbRenderer {}

impl FbRenderer {
    fn new(buf: *mut u8, info: &FbInfo) -> Self {
        let bytes_per_pixel = info.bpp as usize;
        let byte_len = info.stride as usize * bytes_per_pixel * info.height as usize;
        FbRenderer {
            buf,
            byte_len,
            width: info.width as usize,
            height: info.height as usize,
            stride: info.stride as usize,
            bytes_per_pixel,
            pixel_format: info.pixel_format,
            cursor_col: 0,
            cursor_row: 0,
            parser: AnsiParser::new(),
            fg_color: FG,
            bg_color: BG,
            cursor_visible: true,
            cursor_rendered: false,
        }
    }

    fn cols(&self) -> usize {
        self.width / CHAR_W
    }

    fn rows(&self) -> usize {
        self.height / CHAR_H
    }

    /// Write a single pixel at (px, py).
    fn write_pixel(&mut self, px: usize, py: usize, colour: Colour) {
        let offset = py * self.stride * self.bytes_per_pixel + px * self.bytes_per_pixel;
        if offset + self.bytes_per_pixel > self.byte_len {
            return;
        }
        let pixel =
            unsafe { core::slice::from_raw_parts_mut(self.buf.add(offset), self.bytes_per_pixel) };
        if self.bytes_per_pixel < 3 {
            return;
        }
        match self.pixel_format {
            PIXEL_FORMAT_RGB => {
                pixel[0] = colour.r;
                pixel[1] = colour.g;
                pixel[2] = colour.b;
            }
            PIXEL_FORMAT_BGR => {
                pixel[0] = colour.b;
                pixel[1] = colour.g;
                pixel[2] = colour.r;
            }
            _ => {
                pixel[0] = colour.r;
                pixel[1] = colour.g;
                pixel[2] = colour.b;
            }
        }
    }

    /// Render a single character cell at grid position (col, row).
    ///
    /// Phase 69b Track B.1 — codepoint payload widened to `u32` to
    /// match the parser's [`ConsoleCmd::PutChar(u32)`].
    fn render_char_at(&mut self, col: usize, row: usize, codepoint: u32) {
        let px_x = col * CHAR_W;
        let px_y = row * CHAR_H;

        let glyph: &[u8; CHAR_H] = {
            if codepoint >= FONT_FIRST as u32 && codepoint <= FONT_LAST as u32 {
                &FONT[(codepoint - FONT_FIRST as u32) as usize]
            } else {
                &PLACEHOLDER
            }
        };

        let fg = self.fg_color;
        let bg = self.bg_color;
        for (gy, &row_bits) in glyph.iter().enumerate() {
            for gx in 0..CHAR_W {
                let set = (row_bits >> (7 - gx)) & 1 != 0;
                let colour = if set { fg } else { bg };
                self.write_pixel(px_x + gx, px_y + gy, colour);
            }
        }
    }

    /// XOR-invert the cell at the cursor position.
    fn xor_cursor_cell(&mut self) {
        let cols = self.cols();
        let rows = self.rows();
        if cols == 0 || rows == 0 {
            return;
        }
        let col = self.cursor_col.min(cols - 1);
        let row = self.cursor_row.min(rows - 1);
        let px_x = col * CHAR_W;
        let px_y = row * CHAR_H;

        for gy in 0..CHAR_H {
            for gx in 0..CHAR_W {
                let x = px_x + gx;
                let y = px_y + gy;
                let offset = (y * self.stride + x) * self.bytes_per_pixel;
                if offset + self.bytes_per_pixel > self.byte_len {
                    continue;
                }
                let pixel = unsafe {
                    core::slice::from_raw_parts_mut(self.buf.add(offset), self.bytes_per_pixel)
                };
                for byte in pixel.iter_mut() {
                    *byte ^= 0xFF;
                }
            }
        }
    }

    fn show_cursor(&mut self) {
        if self.cursor_visible && !self.cursor_rendered {
            self.xor_cursor_cell();
            self.cursor_rendered = true;
        }
    }

    fn hide_cursor(&mut self) {
        if self.cursor_rendered {
            self.xor_cursor_cell();
            self.cursor_rendered = false;
        }
    }

    /// Scroll the framebuffer up by one character row.
    fn scroll_up(&mut self) {
        let row_bytes = self.stride * self.bytes_per_pixel * CHAR_H;
        let total = self.stride * self.bytes_per_pixel * self.height;
        if row_bytes == 0 || total == 0 || row_bytes >= total {
            self.clear_region(0, 0, self.cols(), self.rows());
            return;
        }
        unsafe {
            core::ptr::copy(self.buf.add(row_bytes), self.buf, total - row_bytes);
        }
        let rows = self.rows();
        if rows > 0 {
            self.clear_region(0, rows - 1, self.cols(), rows);
        }
    }

    /// Render one visible character, advancing the cursor with line wrapping.
    ///
    /// Phase 69b Track B.1 — codepoint payload widened to `u32`.
    fn put_visible_char(&mut self, codepoint: u32) {
        let rows = self.rows();
        let cols = self.cols();
        if rows == 0 || cols == 0 {
            return;
        }

        if self.cursor_col >= cols {
            self.cursor_col = 0;
            self.cursor_row += 1;
            if self.cursor_row >= rows {
                self.scroll_up();
                self.cursor_row = rows - 1;
            }
        }
        self.render_char_at(self.cursor_col, self.cursor_row, codepoint);
        self.cursor_col += 1;
    }

    /// Clear a rectangular region of character cells with the background color.
    fn clear_region(&mut self, col_start: usize, row_start: usize, col_end: usize, row_end: usize) {
        let cols = self.cols();
        let rows = self.rows();
        let bg = self.bg_color;
        for row in row_start..core::cmp::min(row_end, rows) {
            for col in col_start..core::cmp::min(col_end, cols) {
                let px_x = col * CHAR_W;
                let px_y = row * CHAR_H;
                for gy in 0..CHAR_H {
                    for gx in 0..CHAR_W {
                        self.write_pixel(px_x + gx, px_y + gy, bg);
                    }
                }
            }
        }
    }

    /// Execute a parsed console command.
    fn execute_cmd(&mut self, cmd: ConsoleCmd) {
        let rows = self.rows();
        let cols = self.cols();
        if rows == 0 || cols == 0 {
            return;
        }

        match cmd {
            ConsoleCmd::PutChar(c) => self.put_visible_char(c),
            ConsoleCmd::CarriageReturn => {
                self.cursor_col = 0;
            }
            ConsoleCmd::Newline => {
                self.cursor_col = 0;
                self.cursor_row += 1;
                if self.cursor_row >= rows {
                    self.scroll_up();
                    self.cursor_row = rows - 1;
                }
            }
            ConsoleCmd::Backspace => {
                if self.cursor_col > 0 {
                    self.cursor_col -= 1;
                } else if self.cursor_row > 0 {
                    self.cursor_row -= 1;
                    self.cursor_col = cols - 1;
                } else {
                    return;
                }
                self.render_char_at(self.cursor_col, self.cursor_row, b' ' as u32);
            }
            ConsoleCmd::Tab => {
                let next_tab = (self.cursor_col + 8) & !7;
                if next_tab >= cols {
                    self.cursor_col = 0;
                    self.cursor_row += 1;
                    if self.cursor_row >= rows {
                        self.scroll_up();
                        self.cursor_row = rows - 1;
                    }
                } else {
                    self.cursor_col = next_tab;
                }
            }
            ConsoleCmd::CursorUp(n) => {
                self.cursor_row = self.cursor_row.saturating_sub(n as usize);
            }
            ConsoleCmd::CursorDown(n) => {
                self.cursor_row = core::cmp::min(self.cursor_row + n as usize, rows - 1);
            }
            ConsoleCmd::CursorForward(n) => {
                self.cursor_col = core::cmp::min(self.cursor_col + n as usize, cols - 1);
            }
            ConsoleCmd::CursorBack(n) => {
                self.cursor_col = self.cursor_col.saturating_sub(n as usize);
            }
            ConsoleCmd::CursorHorizontalAbsolute(n) => {
                let col = (n as usize).saturating_sub(1);
                self.cursor_col = core::cmp::min(col, cols - 1);
            }
            ConsoleCmd::CursorPosition(row, col) => {
                let r = (row as usize).saturating_sub(1);
                let c = (col as usize).saturating_sub(1);
                self.cursor_row = core::cmp::min(r, rows - 1);
                self.cursor_col = core::cmp::min(c, cols - 1);
            }
            ConsoleCmd::EraseLine(mode) => match mode {
                0 => {
                    self.clear_region(self.cursor_col, self.cursor_row, cols, self.cursor_row + 1);
                }
                1 => {
                    self.clear_region(0, self.cursor_row, self.cursor_col + 1, self.cursor_row + 1);
                }
                2 => {
                    self.clear_region(0, self.cursor_row, cols, self.cursor_row + 1);
                }
                _ => {}
            },
            ConsoleCmd::EraseDisplay(mode) => match mode {
                0 => {
                    self.clear_region(self.cursor_col, self.cursor_row, cols, self.cursor_row + 1);
                    if self.cursor_row + 1 < rows {
                        self.clear_region(0, self.cursor_row + 1, cols, rows);
                    }
                }
                1 => {
                    if self.cursor_row > 0 {
                        self.clear_region(0, 0, cols, self.cursor_row);
                    }
                    self.clear_region(0, self.cursor_row, self.cursor_col + 1, self.cursor_row + 1);
                }
                2 => {
                    self.clear_region(0, 0, cols, rows);
                }
                _ => {}
            },
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                // Phase 69 Track B — only DECTCEM (?25) is meaningful for
                // the text-mode FB console; alt-screen, mouse, and
                // bracketed-paste codes are owned by `term`. We iterate
                // each parsed code so multi-param sequences like
                // `\E[?25;1006h` still toggle the cursor even when the
                // terminfo entry batched it with a `term`-only code.
                for &code in &codes[..count.min(codes.len())] {
                    if code == 25 {
                        if set {
                            self.cursor_visible = true;
                            self.show_cursor();
                        } else {
                            self.hide_cursor();
                            self.cursor_visible = false;
                        }
                    }
                    // Any other code is owned by userspace `term`;
                    // dropping it silently here keeps legacy text-mode
                    // sessions undisturbed.
                }
            }
            // Phase 69 Track F — DECSCUSR cursor shape. The text-mode
            // FB console only paints a single-cell block today, so we
            // accept and ignore the shape selection.
            ConsoleCmd::CursorShape { .. } => {}
            ConsoleCmd::Sgr(sgr) => {
                self.apply_sgr(&sgr);
            }
            // Phase 69d follow-up — the console_server is a single
            // text-mode surface with no scroll-region or insert/delete
            // semantics.  These variants exist for the userspace `term`
            // emulator's incremental-repaint path; we accept the wire
            // bytes (so the parser doesn't fall into garbage state) and
            // drop them as no-ops, except for VPA + ECH which have
            // cheap cell-grid analogs.
            ConsoleCmd::SetScrollRegion { .. } => {}
            ConsoleCmd::VerticalPositionAbsolute(n) => {
                let rows = self.rows();
                let r = (n as usize).saturating_sub(1);
                self.cursor_row = core::cmp::min(r, rows.saturating_sub(1));
            }
            ConsoleCmd::InsertLines(_) => {}
            ConsoleCmd::DeleteLines(_) => {}
            ConsoleCmd::InsertChars(_) => {}
            ConsoleCmd::DeleteChars(_) => {}
            ConsoleCmd::EraseChars(n) => {
                let cols = self.cols();
                let n = n.max(1) as usize;
                let end_col = core::cmp::min(self.cursor_col + n, cols);
                self.clear_region(
                    self.cursor_col,
                    self.cursor_row,
                    end_col,
                    self.cursor_row + 1,
                );
            }
            ConsoleCmd::ScrollUp(_) => {}
            ConsoleCmd::ScrollDown(_) => {}
            ConsoleCmd::Nop => {}
        }
    }

    /// Apply SGR (Select Graphic Rendition) parameters.
    fn apply_sgr(&mut self, sgr: &SgrParams) {
        for i in 0..sgr.count {
            match sgr.params[i] {
                0 => {
                    self.fg_color = FG;
                    self.bg_color = BG;
                }
                1 => {
                    // Bold/bright
                    if let Some(idx) = VGA_COLORS.iter().position(|&col| col == self.fg_color) {
                        self.fg_color = VGA_BRIGHT_COLORS[idx];
                    }
                }
                n @ 30..=37 => {
                    self.fg_color = VGA_COLORS[(n - 30) as usize];
                }
                39 => {
                    self.fg_color = FG;
                }
                n @ 40..=47 => {
                    self.bg_color = VGA_COLORS[(n - 40) as usize];
                }
                49 => {
                    self.bg_color = BG;
                }
                n @ 90..=97 => {
                    self.fg_color = VGA_BRIGHT_COLORS[(n - 90) as usize];
                }
                n @ 100..=107 => {
                    self.bg_color = VGA_BRIGHT_COLORS[(n - 100) as usize];
                }
                _ => {}
            }
        }
    }

    /// Write all characters in `s`, processing through the ANSI parser.
    pub fn write_str(&mut self, s: &str) {
        self.hide_cursor();
        for c in s.chars() {
            let cmd = self.parser.process_char(c);
            self.execute_cmd(cmd);
        }
        self.show_cursor();
    }
}

// ---------------------------------------------------------------------------
// Panic handler
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    syscall_lib::write_str(STDOUT_FILENO, "console_server: PANIC\n");
    syscall_lib::exit(101)
}
