//! Framebuffer text console with ANSI escape sequence support.
//!
//! Provides a fixed-font (8×16 VGA) text renderer on top of the UEFI linear
//! framebuffer.  Supports VT100/ANSI escape sequences (cursor movement, erase,
//! SGR color) needed for Ion's `liner` library to redraw prompts in-place.
//!
//! Public API
//! ----------
//! * `init(fb)`   — call once from `kernel_main` before `task::run()`.
//! * `write_str(s)` — thread-safe via an internal `spin::Mutex`.

#![allow(dead_code)]

use bootloader_api::info::{FrameBuffer, FrameBufferInfo, PixelFormat};
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use kernel_core::fb::{AnsiParser, ConsoleCmd, SgrParams};

use crate::task::scheduler::IrqSafeMutex;

// ---------------------------------------------------------------------------
// 8×16 VGA font — IBM Code Page 437, glyphs 0x20–0x7E (95 characters).
//
// Each entry is 16 bytes: one byte per row, bit 7 = leftmost pixel.
// Source: public-domain VGA BIOS font data (IBM VGA 8×16).
// ---------------------------------------------------------------------------

/// Number of font glyphs (ASCII 0x20 ' ' through 0x7E '~').
const FONT_FIRST: u8 = 0x20;
const FONT_LAST: u8 = 0x7E;
const FONT_GLYPHS: usize = (FONT_LAST - FONT_FIRST + 1) as usize; // 95
const CHAR_W: usize = 8;
const CHAR_H: usize = 16;

/// Placeholder glyph used for characters outside the printable range.
const PLACEHOLDER: [u8; CHAR_H] = [0xFF; CHAR_H];

/// IBM VGA 8×16 font data for ASCII 0x20–0x7E.
/// Each glyph is exactly 16 bytes.  Bit 7 of each byte is the leftmost pixel.
#[rustfmt::skip]
static FONT: [[u8; CHAR_H]; FONT_GLYPHS] = [
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

// ---------------------------------------------------------------------------
// FbConsole — internal state stored under the global Mutex
// ---------------------------------------------------------------------------

/// Pixel colour representation (r, g, b).
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
}; // white
const BG: Colour = Colour {
    r: 0x00,
    g: 0x00,
    b: 0x00,
}; // black

/// Standard VGA color palette (SGR 30–37 / 40–47).
const VGA_COLORS: [Colour; 8] = [
    Colour {
        r: 0x00,
        g: 0x00,
        b: 0x00,
    }, // 0: Black
    Colour {
        r: 0xAA,
        g: 0x00,
        b: 0x00,
    }, // 1: Red
    Colour {
        r: 0x00,
        g: 0xAA,
        b: 0x00,
    }, // 2: Green
    Colour {
        r: 0xAA,
        g: 0x55,
        b: 0x00,
    }, // 3: Yellow/Brown
    Colour {
        r: 0x00,
        g: 0x00,
        b: 0xAA,
    }, // 4: Blue
    Colour {
        r: 0xAA,
        g: 0x00,
        b: 0xAA,
    }, // 5: Magenta
    Colour {
        r: 0x00,
        g: 0xAA,
        b: 0xAA,
    }, // 6: Cyan
    Colour {
        r: 0xAA,
        g: 0xAA,
        b: 0xAA,
    }, // 7: White (light gray)
];

/// Bright VGA color palette (SGR 90–97 / 100–107).
const VGA_BRIGHT_COLORS: [Colour; 8] = [
    Colour {
        r: 0x55,
        g: 0x55,
        b: 0x55,
    }, // 0: Bright Black (dark gray)
    Colour {
        r: 0xFF,
        g: 0x55,
        b: 0x55,
    }, // 1: Bright Red
    Colour {
        r: 0x55,
        g: 0xFF,
        b: 0x55,
    }, // 2: Bright Green
    Colour {
        r: 0xFF,
        g: 0xFF,
        b: 0x55,
    }, // 3: Bright Yellow
    Colour {
        r: 0x55,
        g: 0x55,
        b: 0xFF,
    }, // 4: Bright Blue
    Colour {
        r: 0xFF,
        g: 0x55,
        b: 0xFF,
    }, // 5: Bright Magenta
    Colour {
        r: 0x55,
        g: 0xFF,
        b: 0xFF,
    }, // 6: Bright Cyan
    Colour {
        r: 0xFF,
        g: 0xFF,
        b: 0xFF,
    }, // 7: Bright White
];

/// Internal framebuffer console state.
struct FbConsole {
    // SAFETY: This raw pointer is derived from `&'static mut FrameBuffer`
    // (obtained from BootInfo before any tasks start).  The pointer is valid
    // for the lifetime of the kernel and is only ever accessed while the
    // enclosing `spin::Mutex` is held, so there are no concurrent aliased
    // writes.
    buf: *mut u8,
    byte_len: usize,
    width: usize,
    height: usize,
    /// Pixels between the start of one row and the start of the next.
    stride: usize,
    bytes_per_pixel: usize,
    pixel_format: PixelFormat,
    cursor_col: usize,
    cursor_row: usize,
    /// ANSI escape sequence parser state machine.
    parser: AnsiParser,
    /// Current foreground color for text rendering.
    fg_color: Colour,
    /// Current background color for text rendering.
    bg_color: Colour,
    /// Whether the cursor is visible (DECTCEM).
    cursor_visible: bool,
    /// Whether the cursor is currently rendered on screen (XOR'd).
    cursor_rendered: bool,
}

// SAFETY: FbConsole is only accessed under a spin::Mutex; the raw pointer is
// derived from a &'static mut framebuffer and is never aliased outside the lock.
unsafe impl Send for FbConsole {}

impl FbConsole {
    fn new(buf: *mut u8, info: FrameBufferInfo) -> Self {
        FbConsole {
            buf,
            byte_len: info.byte_len,
            width: info.width,
            height: info.height,
            stride: info.stride,
            bytes_per_pixel: info.bytes_per_pixel,
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

    /// Number of text columns that fit on screen.
    fn cols(&self) -> usize {
        self.width / CHAR_W
    }

    /// Number of text rows that fit on screen.
    fn rows(&self) -> usize {
        self.height / CHAR_H
    }

    /// Write a single pixel at `(px, py)` with the given colour.
    ///
    /// # Safety
    /// Caller must ensure `px < width` and `py < height`.
    fn write_pixel(&mut self, px: usize, py: usize, colour: Colour) {
        let offset = py * self.stride * self.bytes_per_pixel + px * self.bytes_per_pixel;
        if offset + self.bytes_per_pixel > self.byte_len {
            return;
        }
        // SAFETY: offset is within [0, byte_len) as checked above; buf is the
        // static framebuffer pointer held under the mutex.
        let pixel =
            unsafe { core::slice::from_raw_parts_mut(self.buf.add(offset), self.bytes_per_pixel) };

        match self.pixel_format {
            PixelFormat::Rgb if self.bytes_per_pixel >= 3 => {
                pixel[0] = colour.r;
                pixel[1] = colour.g;
                pixel[2] = colour.b;
                // bytes_per_pixel may be 4; 4th byte left as-is (padding).
            }
            PixelFormat::Rgb => {
                // bytes_per_pixel < 3 — nothing we can do safely.
            }
            PixelFormat::Bgr if self.bytes_per_pixel >= 3 => {
                pixel[0] = colour.b;
                pixel[1] = colour.g;
                pixel[2] = colour.r;
            }
            PixelFormat::Bgr => {
                // bytes_per_pixel < 3 — nothing we can do safely.
            }
            PixelFormat::U8 => {
                // Greyscale: use luminance approximation.
                let luma = ((colour.r as u16 * 77 + colour.g as u16 * 150 + colour.b as u16 * 29)
                    >> 8) as u8;
                pixel[0] = luma;
            }
            // Best-effort for unknown pixel formats: write RGB bytes when
            // there are at least 3 bytes per pixel.
            PixelFormat::Unknown { .. } if self.bytes_per_pixel >= 3 => {
                pixel[0] = colour.r;
                pixel[1] = colour.g;
                pixel[2] = colour.b;
            }
            PixelFormat::Unknown { .. } => {
                // bytes_per_pixel < 3 — nothing we can do safely.
            }
            // Non-exhaustive enum — silently ignore future variants.
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    /// Render a single character cell at grid position `(col, row)`.
    ///
    /// Phase 69b Track B.1 — accepts a `u32` codepoint to match the
    /// widened [`ConsoleCmd::PutChar`] payload. The kernel framebuffer
    /// console still only ships the 7-bit ASCII bitmap font; anything
    /// outside `FONT_FIRST..=FONT_LAST` falls through to `PLACEHOLDER`.
    /// The richer Latin-1 / box-drawing tables live in the userspace
    /// `term` renderer (kernel-core::session::font), so applications
    /// running under `term` see the full Phase 69b coverage while the
    /// kernel console keeps its minimal footprint.
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

    /// XOR-invert the cell at the cursor position (unconditional toggle).
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

    /// Ensure the cursor is visually rendered on screen (if visible).
    fn show_cursor(&mut self) {
        if self.cursor_visible && !self.cursor_rendered {
            self.xor_cursor_cell();
            self.cursor_rendered = true;
        }
    }

    /// Ensure the cursor is visually erased from screen (if rendered).
    fn hide_cursor(&mut self) {
        if self.cursor_rendered {
            self.xor_cursor_cell();
            self.cursor_rendered = false;
        }
    }

    /// Scroll the framebuffer up by one character row, clearing the last row.
    fn scroll_up(&mut self) {
        let row_bytes = self.stride * self.bytes_per_pixel * CHAR_H;
        let total = self.stride * self.bytes_per_pixel * self.height;
        if row_bytes == 0 || total == 0 || row_bytes >= total {
            // No full text row fits in the framebuffer. Clear what we have.
            self.clear_region(0, 0, self.cols(), self.rows());
            return;
        }
        // SAFETY: `self.buf` points to the framebuffer with `total` bytes. We
        // intentionally copy the overlapping range `[buf + row_bytes, buf + total)`
        // down to `[buf, buf + total - row_bytes)` to scroll the contents up.
        // `core::ptr::copy` is used because it provides memmove semantics for
        // overlapping source and destination regions.
        unsafe {
            // Shift buffer up by one text row.
            core::ptr::copy(self.buf.add(row_bytes), self.buf, total - row_bytes);
        }
        // Clear the last row using the current background color.
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
                let n = n as usize;
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            ConsoleCmd::CursorDown(n) => {
                let n = n as usize;
                self.cursor_row = core::cmp::min(self.cursor_row + n, rows - 1);
            }
            ConsoleCmd::CursorForward(n) => {
                let n = n as usize;
                self.cursor_col = core::cmp::min(self.cursor_col + n, cols - 1);
            }
            ConsoleCmd::CursorBack(n) => {
                let n = n as usize;
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            ConsoleCmd::CursorHorizontalAbsolute(n) => {
                // n is 1-based.
                let col = (n as usize).saturating_sub(1);
                self.cursor_col = core::cmp::min(col, cols - 1);
            }
            ConsoleCmd::CursorPosition(row, col) => {
                // Both 1-based.
                let r = (row as usize).saturating_sub(1);
                let c = (col as usize).saturating_sub(1);
                self.cursor_row = core::cmp::min(r, rows - 1);
                self.cursor_col = core::cmp::min(c, cols - 1);
            }
            ConsoleCmd::EraseLine(mode) => {
                match mode {
                    0 => {
                        // Erase from cursor to end of line.
                        self.clear_region(
                            self.cursor_col,
                            self.cursor_row,
                            cols,
                            self.cursor_row + 1,
                        );
                    }
                    1 => {
                        // Erase from start of line to cursor.
                        self.clear_region(
                            0,
                            self.cursor_row,
                            self.cursor_col + 1,
                            self.cursor_row + 1,
                        );
                    }
                    2 => {
                        // Erase entire line.
                        self.clear_region(0, self.cursor_row, cols, self.cursor_row + 1);
                    }
                    _ => {}
                }
            }
            ConsoleCmd::EraseDisplay(mode) => {
                match mode {
                    0 => {
                        // Erase from cursor to end of screen.
                        self.clear_region(
                            self.cursor_col,
                            self.cursor_row,
                            cols,
                            self.cursor_row + 1,
                        );
                        if self.cursor_row + 1 < rows {
                            self.clear_region(0, self.cursor_row + 1, cols, rows);
                        }
                    }
                    1 => {
                        // Erase from start of screen to cursor.
                        if self.cursor_row > 0 {
                            self.clear_region(0, 0, cols, self.cursor_row);
                        }
                        self.clear_region(
                            0,
                            self.cursor_row,
                            self.cursor_col + 1,
                            self.cursor_row + 1,
                        );
                    }
                    2 => {
                        // Erase entire screen (don't move cursor).
                        self.clear_region(0, 0, cols, rows);
                    }
                    _ => {}
                }
            }
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                // Phase 69 Track B — the kernel FB console is a single
                // text-mode surface; only DECTCEM (?25) is meaningful
                // here. We iterate each parsed code so multi-param
                // sequences like `\E[?25;1006h` still toggle the
                // cursor when terminfo batches `?25` with a
                // `term`-only code. Other codes (alt-screen, mouse,
                // bracketed paste) are owned by userspace `term` and
                // drop silently to keep the kernel console undisturbed.
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
                }
            }
            // Phase 69 Track F — DECSCUSR cursor shape is purely
            // cosmetic for the kernel FB console; accept and ignore.
            ConsoleCmd::CursorShape { .. } => {}
            ConsoleCmd::Sgr(sgr) => {
                self.apply_sgr(&sgr);
            }
            // The kernel framebuffer console is a single text-mode
            // surface with no scroll-region, scrollback, or per-row
            // insert/delete semantics.  These variants exist for the
            // userspace `term` emulator's incremental-repaint path
            // (Phase 69d follow-up — less and other TUI apps emit
            // these per the m3os-term terminfo).  Dropping them as
            // no-ops here keeps the kernel console undisturbed while
            // letting the variants land in the wire protocol.
            ConsoleCmd::SetScrollRegion { .. } => {}
            ConsoleCmd::VerticalPositionAbsolute(n) => {
                let row = (n as usize).saturating_sub(1);
                self.cursor_row = core::cmp::min(row, rows.saturating_sub(1));
            }
            ConsoleCmd::InsertLines(_) => {}
            ConsoleCmd::DeleteLines(_) => {}
            ConsoleCmd::InsertChars(_) => {}
            ConsoleCmd::DeleteChars(_) => {}
            ConsoleCmd::EraseChars(n) => {
                // The cell-grid analog of ECH: blank `n` cells starting
                // at the cursor without moving it.  Cheap to implement
                // even on the kernel console because we already have
                // clear_region.
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
                    // Reset all attributes.
                    self.fg_color = FG;
                    self.bg_color = BG;
                }
                1 => {
                    // Bold/bright — map standard foreground colors to their
                    // bright variants. This is idempotent: if the current
                    // foreground is already a bright color (or a non-palette
                    // color), we leave it unchanged.
                    if let Some(idx) = VGA_COLORS.iter().position(|&col| col == self.fg_color) {
                        self.fg_color = VGA_BRIGHT_COLORS[idx];
                    }
                }
                // Standard foreground colors 30–37.
                n @ 30..=37 => {
                    self.fg_color = VGA_COLORS[(n - 30) as usize];
                }
                39 => {
                    // Default foreground.
                    self.fg_color = FG;
                }
                // Standard background colors 40–47.
                n @ 40..=47 => {
                    self.bg_color = VGA_COLORS[(n - 40) as usize];
                }
                49 => {
                    // Default background.
                    self.bg_color = BG;
                }
                // Bright foreground colors 90–97.
                n @ 90..=97 => {
                    self.fg_color = VGA_BRIGHT_COLORS[(n - 90) as usize];
                }
                // Bright background colors 100–107.
                n @ 100..=107 => {
                    self.bg_color = VGA_BRIGHT_COLORS[(n - 100) as usize];
                }
                _ => {} // Unknown SGR parameter — ignore.
            }
        }
    }

    /// Write all characters in `s`, processing through the ANSI parser.
    fn write_str(&mut self, s: &str) {
        // Erase cursor before modifying the framebuffer.
        self.hide_cursor();
        for c in s.chars() {
            let cmd = self.parser.process_char(c);
            self.execute_cmd(cmd);
        }
        // Redraw cursor at (possibly new) position.
        self.show_cursor();
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/// Phase 57b G.7 — IrqSafeMutex inherits Track F.1's preempt-discipline.
/// CONSOLE is only acquired from task / panic context (kernel logs,
/// framebuffer init, panic handler).  No ISR reaches it.
static CONSOLE: IrqSafeMutex<Option<FbConsole>> = IrqSafeMutex::new(None);

/// When `true`, framebuffer text output is suppressed (a graphical process owns
/// the framebuffer directly).  Serial output is unaffected.
static CONSOLE_YIELDED: AtomicBool = AtomicBool::new(false);

/// PID of the process that currently owns the raw framebuffer (0 = none).
static FB_OWNER_PID: AtomicU32 = AtomicU32::new(0);
const FB_OWNER_TRANSITIONING: u32 = u32::MAX;

/// Whether the current framebuffer owner should receive raw/game scancodes.
///
/// Direct framebuffer clients such as DOOM need exclusive raw keyboard input,
/// but the ring-3 compositor owns the framebuffer while still expecting input
/// to flow through `kbd_server` / `stdin_feeder`. Keep that policy explicit so
/// pixel ownership does not implicitly steal the normal TTY input route.
static FB_RAW_INPUT_ENABLED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Initialise the framebuffer text console.
///
/// Called once from `kernel_main` before `task::run()`.  Extracts the raw
/// pointer and layout from `fb`, stores them in the global [`CONSOLE`], and
/// clears the screen.
///
/// # Safety
/// `fb` must be `&'static mut` (derived from `boot_info.framebuffer`).  The
/// caller must not access `boot_info.framebuffer` again after this call.
///
/// Returns `true` if the framebuffer was large enough to enable the text console.
pub fn init(fb: &'static mut FrameBuffer) -> bool {
    let info: FrameBufferInfo = fb.info();
    // Extract the raw mutable pointer from the &'static mut FrameBuffer.
    // SAFETY: fb is &'static mut so the pointer is valid for the kernel
    // lifetime.  We store it inside a Mutex<Option<FbConsole>> so it is
    // only ever accessed with the lock held, preventing aliased writes.
    let buf_ptr: *mut u8 = fb.buffer_mut().as_mut_ptr();
    // SAFETY: buf_ptr and info are both derived from &'static mut FrameBuffer.
    unsafe { init_from_parts(buf_ptr, info) }
}

/// Initialise the framebuffer text console from pre-extracted raw components.
///
/// This variant exists to work around the borrow-checker constraint imposed by
/// `mm::init` consuming `&'static mut BootInfo`: callers can extract the raw
/// pointer and [`FrameBufferInfo`] before calling `mm::init`, then call this
/// function afterwards without any live borrow on `BootInfo`.
///
/// # Safety
/// * `buf_ptr` must be a valid, non-null, writable pointer to a framebuffer
///   buffer of at least `info.byte_len` bytes.
/// * The buffer must remain valid for the lifetime of the kernel (i.e. it must
///   be the `'static` UEFI framebuffer mapping set up by the bootloader).
/// * No other code may write to the framebuffer memory without holding the
///   internal [`CONSOLE`] mutex.
///
/// Returns `true` if the framebuffer was large enough to enable the text console.
pub unsafe fn init_from_parts(buf_ptr: *mut u8, info: FrameBufferInfo) -> bool {
    let total_bytes = match info
        .stride
        .checked_mul(info.height)
        .and_then(|pixels| pixels.checked_mul(info.bytes_per_pixel))
    {
        Some(total) if total <= info.byte_len => total,
        _ => {
            *CONSOLE.lock() = None;
            return false;
        }
    };

    if info.width < CHAR_W
        || info.height < CHAR_H
        || info.byte_len == 0
        || info.bytes_per_pixel == 0
        || matches!(info.pixel_format, PixelFormat::Rgb | PixelFormat::Bgr)
            && info.bytes_per_pixel < 3
    {
        *CONSOLE.lock() = None;
        return false;
    }

    let mut console = FbConsole::new(buf_ptr, info);

    // Clear the mapped framebuffer region before handing over to the cursor
    // logic. `total_bytes` was checked for overflow and bounded by `byte_len`.
    unsafe {
        core::ptr::write_bytes(buf_ptr, 0x00, total_bytes);
    }

    console.cursor_col = 0;
    console.cursor_row = 0;

    *CONSOLE.lock() = Some(console);
    true
}

/// Return the framebuffer console text dimensions (rows, cols), or None
/// if no framebuffer console is active.
pub fn console_text_size() -> Option<(u16, u16)> {
    let guard = CONSOLE.lock();
    guard.as_ref().map(|c| (c.rows() as u16, c.cols() as u16))
}

/// Write a string to the framebuffer console at the current cursor position.
///
/// Handles `'\n'` (newline) and `'\x08'` (backspace).  Characters outside
/// ASCII 0x20–0x7E are rendered as a filled-block placeholder glyph.
/// Thread-safe via the internal [`CONSOLE`] mutex.
///
/// Does nothing if [`init`] has not been called yet.
pub fn write_str(s: &str) {
    // Fast path: skip taking the lock if the console is clearly yielded.
    if CONSOLE_YIELDED.load(Ordering::Acquire) {
        return;
    }
    let mut guard = CONSOLE.lock();
    // Re-check under the lock: serializes with try_yield_console's
    // lock-guarded flag transition so a yield cannot slip between our
    // pre-check above and the framebuffer write below.
    if CONSOLE_YIELDED.load(Ordering::Acquire) {
        return;
    }
    if let Some(ref mut console) = *guard {
        console.write_str(s);
    }
}

// ---------------------------------------------------------------------------
// Phase 47: framebuffer info helpers and console yield/restore
// ---------------------------------------------------------------------------

/// Returns `(width, height, stride, bytes_per_pixel, pixel_format)` or `None`
/// if the framebuffer console has not been initialised.
pub fn framebuffer_raw_info() -> Option<(usize, usize, usize, usize, PixelFormat)> {
    let guard = CONSOLE.lock();
    guard.as_ref().map(|c| {
        (
            c.width,
            c.height,
            c.stride,
            c.bytes_per_pixel,
            c.pixel_format,
        )
    })
}

/// Returns `(buf_virt_addr, byte_len)` of the raw framebuffer, or `None`.
pub fn framebuffer_buf_addr() -> Option<(u64, usize)> {
    let guard = CONSOLE.lock();
    guard.as_ref().map(|c| (c.buf as u64, c.byte_len))
}

/// Suppresses all framebuffer console output.
///
/// For internal use only.  External callers that need to claim the
/// framebuffer should use [`try_yield_console`] which performs an
/// atomic compare-and-swap ownership check.
fn yield_console(owner_pid: u32, raw_input_enabled: bool) {
    FB_OWNER_PID.store(owner_pid, Ordering::Release);
    FB_RAW_INPUT_ENABLED.store(raw_input_enabled, Ordering::Release);
    crate::arch::x86_64::interrupts::reset_raw_input_state();
    let _guard = CONSOLE.lock();
    CONSOLE_YIELDED.store(true, Ordering::Release);
}

/// Atomically claim the framebuffer for `owner_pid` using compare-and-swap,
/// then suppress console text output.
///
/// Returns `true` if the framebuffer was unowned (CAS succeeded) or is already
/// owned by `owner_pid` (re-entrant call).  Returns `false` if another process
/// owns it — caller should return `EBUSY` without doing any mapping.
pub fn try_yield_console(owner_pid: u32, raw_input_enabled: bool) -> bool {
    match FB_OWNER_PID.compare_exchange(0, owner_pid, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            FB_RAW_INPUT_ENABLED.store(raw_input_enabled, Ordering::Release);
            crate::arch::x86_64::interrupts::reset_raw_input_state();
            // Phase 57d follow-up — when transitioning to a graphical-server
            // FB owner (raw_input_enabled = false: display_server, not a
            // takeover program), inject break codes for commonly-held keys
            // so kbd_server's tracker / repeat scheduler clear any keys
            // that were held when the previous FB owner took over.
            //
            // This fires at every FB ownership transition into a
            // graphical-server — both the initial display_server boot
            // claim AND post-takeover reclaim. The boot fire is benign
            // (no keys held, no prior tracker state) but empirically
            // affects subsequent takeover behavior, so we keep it. The
            // injection is wrapped in `without_interrupts` inside
            // `inject_release_all_held_modifiers` to serialise the
            // task-context push against the keyboard ISR.
            if !raw_input_enabled {
                crate::arch::x86_64::interrupts::inject_release_all_held_modifiers();
            }
            // Hold the console lock while flipping the flag so write_str
            // cannot slip through between the CAS and the flag transition.
            let _guard = CONSOLE.lock();
            CONSOLE_YIELDED.store(true, Ordering::Release);
            true
        }
        Err(current) if current == owner_pid => {
            FB_RAW_INPUT_ENABLED.store(raw_input_enabled, Ordering::Release);
            // Re-entrant: we already own the FB.  The original claim may still
            // be waiting to acquire CONSOLE before setting CONSOLE_YIELDED=true,
            // so ensure the flag is set before we return to avoid a window where
            // write_str observes CONSOLE_YIELDED=false while we're the owner.
            if !CONSOLE_YIELDED.load(Ordering::Acquire) {
                let _guard = CONSOLE.lock();
                CONSOLE_YIELDED.store(true, Ordering::Release);
            }
            true
        }
        Err(_) => false, // owned by another process
    }
}

/// Release a previously claimed console ownership without a full restore.
///
/// Used to roll back `try_yield_console` when the framebuffer mapping fails
/// after a successful CAS.  Does not clear the screen.
pub fn release_console_claim(owner_pid: u32) {
    if FB_OWNER_PID
        .compare_exchange(
            owner_pid,
            FB_OWNER_TRANSITIONING,
            Ordering::AcqRel,
            Ordering::Relaxed,
        )
        .is_ok()
    {
        crate::arch::x86_64::interrupts::reset_raw_input_state();
        FB_RAW_INPUT_ENABLED.store(false, Ordering::Release);
        CONSOLE_YIELDED.store(false, Ordering::Release);
        FB_OWNER_PID.store(0, Ordering::Release);
    }
}

/// Restores framebuffer console output after a graphical process exits.
///
/// Clears the framebuffer and resets the owner PID.  The console lock is held
/// throughout the clear so no writer can sneak in between the flag flip and the
/// clear and have its output immediately wiped.
///
/// `FB_OWNER_PID` is cleared **last** — after `CONSOLE_YIELDED` is already
/// false and the lock is dropped — so `try_yield_console` cannot win the CAS
/// while the restore is still in progress (the CAS requires owner == 0).
pub fn restore_console() {
    // Hold the lock while clearing so no concurrent writer observes
    // CONSOLE_YIELDED = false and starts writing before the clear completes.
    {
        let mut guard = CONSOLE.lock();
        if let Some(ref mut console) = *guard {
            let rows = console.rows();
            let cols = console.cols();
            if rows > 0 && cols > 0 {
                console.clear_region(0, 0, cols, rows);
            }
        }
        // Flip the flag only after the clear is done and while we still hold
        // the lock, so the first write after restore sees a freshly-cleared
        // screen.
        CONSOLE_YIELDED.store(false, Ordering::Release);
    } // ← lock drops here

    crate::arch::x86_64::interrupts::reset_raw_input_state();
    FB_RAW_INPUT_ENABLED.store(false, Ordering::Release);

    // Clear ownership last: a concurrent try_yield_console sees owner != 0
    // until this point, preventing it from racing mid-restore and then having
    // CONSOLE_YIELDED overwritten back to false by this function.
    FB_OWNER_PID.store(0, Ordering::Release);
}

/// Returns the PID of the process currently owning the raw framebuffer
/// (0 = no owner).
pub fn fb_owner_pid() -> u32 {
    FB_OWNER_PID.load(Ordering::Acquire)
}

/// True when IRQ1 should route scancodes to the raw/game-input buffer.
pub fn raw_input_active() -> bool {
    fb_owner_pid() != 0 && FB_RAW_INPUT_ENABLED.load(Ordering::Acquire)
}
