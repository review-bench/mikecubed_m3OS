//! Phase 57 Track G.4 — screen state machine + ANSI-parser consumer.
//!
//! `Screen` owns the cell buffer (fixed `cols * rows` cells), the
//! cursor, the active colours, the ANSI parser ([`kernel_core::fb::AnsiParser`]
//! reused per the G.4 acceptance "Reuse Phase 22b's parser"), and a
//! ring of evicted lines capped at [`SCROLLBACK_LINES`].
//!
//! Each call to [`Screen::feed`] passes one byte through the parser,
//! interprets the resulting [`ConsoleCmd`], updates the cell buffer,
//! and pushes one or more typed [`RenderCommand`] values into the
//! caller-supplied output `Vec`.  No allocation per character — only
//! scrollback growth allocates, and only when a row evicts.
//!
//! BEL (0x07) is intercepted *before* the parser so it does not become
//! a `PutChar('\x07')`; it maps directly to [`RenderCommand::Bell`].
//! All other control bytes (CR, LF, BS, TAB) flow through the parser
//! and are handled via the parser's [`ConsoleCmd`] vocabulary.

use alloc::vec::Vec;

use kernel_core::fb::{AnsiParser, ConsoleCmd, SgrOp};
use kernel_core::session::width_of;
use kernel_core::utf8::{DecoderOutput, Utf8Decoder};

use crate::{DEFAULT_COLS, DEFAULT_ROWS, SCROLLBACK_LINES};

/// Phase 69b Track B.2 — Unicode replacement character emitted by
/// [`Screen::feed`] whenever the UTF-8 decoder reports `Invalid`.
/// Matches the W3C / WHATWG replacement-character contract.
pub const REPLACEMENT_CHARACTER: u32 = 0xFFFD;

/// Errors observable on the screen public surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScreenError {
    /// Render command requested an `(row, col)` outside the cell grid.
    OutOfBounds,
}

/// Default foreground colour (white, BGRA8888 packed).
pub const DEFAULT_FG: u32 = 0xFFFF_FFFF;

/// Default background colour (black, BGRA8888 packed).
pub const DEFAULT_BG: u32 = 0x0000_0000;

/// 8-entry SGR colour palette (BGRA8888).  Index `n` corresponds to
/// SGR 30+`n` (foreground) and 40+`n` (background).
const SGR_PALETTE: [u32; 8] = [
    0x0000_0000, // 0 black
    0xFFAA_0000, // 1 red
    0xFF00_AA00, // 2 green
    0xFFAA_5500, // 3 yellow / brown
    0xFF00_00AA, // 4 blue
    0xFFAA_00AA, // 5 magenta
    0xFF00_AAAA, // 6 cyan
    0xFFAA_AAAA, // 7 light grey
];

/// Phase 69 Track C — 16-entry palette covering bright variants for
/// indices 8..=15 of the xterm 256 palette. Mirrors xterm's default
/// palette so 256-color themes render with the expected hues.
const SGR_BRIGHT_PALETTE: [u32; 8] = [
    0xFF55_5555, // 8  bright black / dark grey
    0xFFFF_5555, // 9  bright red
    0xFF55_FF55, // 10 bright green
    0xFFFF_FF55, // 11 bright yellow
    0xFF55_55FF, // 12 bright blue
    0xFFFF_55FF, // 13 bright magenta
    0xFF55_FFFF, // 14 bright cyan
    0xFFFF_FFFF, // 15 white
];

/// Phase 69 Track C — full xterm 256-color palette in BGRA8888.
///
/// Layout follows the canonical xterm scheme:
/// - 0..=7    : 8 standard ANSI colors (matches [`SGR_PALETTE`]).
/// - 8..=15   : 8 bright ANSI colors (matches [`SGR_BRIGHT_PALETTE`]).
/// - 16..=231 : 6×6×6 RGB cube. Index `16 + 36r + 6g + b` maps each
///              component to one of the six steps `{0, 95, 135, 175,
///              215, 255}`.
/// - 232..=255: 24-step greyscale ramp from `(8,8,8)` to `(238,238,238)`
///              in steps of 10.
pub const XTERM_256_PALETTE: [u32; 256] = build_xterm_256_palette();

const fn build_xterm_256_palette() -> [u32; 256] {
    let mut p: [u32; 256] = [0; 256];
    // 0..=7 standard ANSI
    let std: [u32; 8] = SGR_PALETTE;
    let mut i: usize = 0;
    while i < 8 {
        p[i] = std[i];
        i += 1;
    }
    // 8..=15 bright ANSI
    let bright: [u32; 8] = SGR_BRIGHT_PALETTE;
    let mut j: usize = 0;
    while j < 8 {
        p[8 + j] = bright[j];
        j += 1;
    }
    // 16..=231 — 6×6×6 cube.
    let steps: [u32; 6] = [0, 95, 135, 175, 215, 255];
    let mut r: usize = 0;
    while r < 6 {
        let mut g: usize = 0;
        while g < 6 {
            let mut b: usize = 0;
            while b < 6 {
                let red = steps[r];
                let grn = steps[g];
                let blu = steps[b];
                let idx = 16 + 36 * r + 6 * g + b;
                p[idx] = 0xFF00_0000 | (red << 16) | (grn << 8) | blu;
                b += 1;
            }
            g += 1;
        }
        r += 1;
    }
    // 232..=255 — 24-step greyscale.
    let mut k: usize = 0;
    while k < 24 {
        let v: u32 = (8 + 10 * k as u32) & 0xff;
        p[232 + k] = 0xFF00_0000 | (v << 16) | (v << 8) | v;
        k += 1;
    }
    p
}

/// Phase 69 Track C — resolve any `Color` to a BGRA8888 32-bit pixel.
/// Centralised so [`Screen::apply_sgr`] and any later consumer can
/// share one mapping with no per-pixel allocation.
pub fn color_to_bgra(color: Color) -> u32 {
    match color {
        Color::Default { foreground } => {
            if foreground {
                DEFAULT_FG
            } else {
                DEFAULT_BG
            }
        }
        Color::Standard(i) => SGR_PALETTE[(i & 7) as usize],
        Color::Bright(i) => SGR_BRIGHT_PALETTE[(i & 7) as usize],
        Color::Indexed(i) => XTERM_256_PALETTE[i as usize],
        Color::Rgb(r, g, b) => 0xFF00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32),
    }
}

/// Phase 69 Track C — typed colour value the screen state machine
/// resolves to a packed pixel via [`color_to_bgra`]. Carries enough
/// information to compress SGR state into a single typed slot without
/// the renderer having to re-walk parameter lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    /// `SGR 39` / `SGR 49` — restore the documented default.
    Default { foreground: bool },
    /// 8-color ANSI (palette index 0..=7).
    Standard(u8),
    /// Bright 8-color ANSI (palette index 8..=15).
    Bright(u8),
    /// 256-color indexed palette entry (0..=255).
    Indexed(u8),
    /// 24-bit RGB truecolor.
    Rgb(u8, u8, u8),
}

/// Output of the screen state machine.  Each command is a single typed
/// hint to the renderer; the renderer batches commands per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderCommand {
    /// Write `codepoint` at `(row, col)` with the given colours.
    PutGlyph {
        row: u16,
        col: u16,
        codepoint: u32,
        fg: u32,
        bg: u32,
    },
    /// Scroll the visible region by `amount` rows (positive = down).
    Scroll { amount: i16 },
    /// Update the active foreground/background colour.
    SetColor { fg: u32, bg: u32 },
    /// BEL (0x07): the renderer rings the audio bell.
    Bell,
    /// Move the cursor to `(row, col)`, both 0-based.
    MoveCursor { row: u16, col: u16 },
    /// Clear the entire screen to the active background colour.
    Clear,
    /// Phase 69 Track E — out-of-band mouse-mode notification. The
    /// renderer ignores this; the binary's main loop intercepts it
    /// and forwards to the [`mouse::MouseReporter`].
    SetMouseMode { code: u16, set: bool },
}

/// One cell in the screen buffer.  `codepoint` is the glyph; `fg`/`bg`
/// are the BGRA8888 packed colours at the time the cell was written.
///
/// Phase 69b Track F — wide-cell accounting. A double-width codepoint
/// (per the EAW table in `kernel-core::session::width_of`) occupies
/// two adjacent cells: the leading cell carries the codepoint, and the
/// trailing cell carries `codepoint = 0`, `wide_continuation = true`.
/// The renderer skips wide-continuation cells when painting so the
/// leading glyph's pixels are not stomped by a half-width repaint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub codepoint: u32,
    pub fg: u32,
    pub bg: u32,
    /// Phase 69b Track F — true when this cell is the trailing
    /// half of a wide (double-width) glyph; the leading cell at
    /// `(row, col - 1)` carries the codepoint.
    pub wide_continuation: bool,
}

impl Cell {
    /// Empty cell painted with the active colours.
    const fn blank(fg: u32, bg: u32) -> Self {
        Self {
            codepoint: 0x20,
            fg,
            bg,
            wide_continuation: false,
        }
    }
}

/// Phase 69 Track B — which of the dual cell grids is currently active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenSelect {
    /// Default cell grid (`buf`); the one that absorbs shell scrollback.
    Primary,
    /// Alternate cell grid (`alt_buf`); selected by `?1049h` or `?47h`
    /// and used by full-screen applications.
    Alt,
}

/// Phase 69 Track F — visual cursor shape established via DECSCUSR.
/// Defaults to [`CursorShape::BlinkingBlock`] to match xterm's
/// default. Each value is paired with one of the seven DEC codes
/// `0..=6`; `0` and `1` both map to `BlinkingBlock` since DEC uses
/// `0` to mean "default", and the xterm default is a blinking block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorShape {
    /// DECSCUSR 0 (default) / 1.
    BlinkingBlock,
    /// DECSCUSR 2.
    SteadyBlock,
    /// DECSCUSR 3.
    BlinkingUnderline,
    /// DECSCUSR 4.
    SteadyUnderline,
    /// DECSCUSR 5.
    BlinkingBar,
    /// DECSCUSR 6.
    SteadyBar,
}

impl CursorShape {
    /// Map a DEC numeric code to the matching shape. Codes `> 6` are
    /// filtered upstream in the parser, so we never observe them here.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::SteadyBlock,
            3 => Self::BlinkingUnderline,
            4 => Self::SteadyUnderline,
            5 => Self::BlinkingBar,
            6 => Self::SteadyBar,
            _ => Self::BlinkingBlock,
        }
    }

    /// Returns `true` when the shape's blink rate is part of its
    /// definition (so the renderer must paint with the visibility
    /// toggle).
    pub fn is_blinking(self) -> bool {
        matches!(
            self,
            Self::BlinkingBlock | Self::BlinkingUnderline | Self::BlinkingBar
        )
    }
}

/// Phase 69 Track B — saved cursor state captured on alt-screen entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct SavedCursor {
    row: u16,
    col: u16,
    fg: u32,
    bg: u32,
}

/// Cell-buffer + cursor + ANSI-parser-driven screen state machine.
///
/// The buffer is fixed-size (`cols * rows`) and pre-allocated; no
/// allocation per character.  Scrollback is a ring of evicted rows
/// capped at [`SCROLLBACK_LINES`].
pub struct Screen {
    cols: u16,
    rows: u16,
    /// Primary `(row * cols + col)`-indexed cell buffer.
    buf: Vec<Cell>,
    /// Phase 69 Track B — alternate `(row * cols + col)`-indexed cell
    /// buffer used while `ScreenSelect::Alt` is active.
    alt_buf: Vec<Cell>,
    /// Phase 69 Track B — which grid is currently active.
    active: ScreenSelect,
    /// Phase 69 Track B — cursor + colours saved on alt-screen entry
    /// so [`Screen::switch_to_primary`] can restore them.
    saved_cursor: SavedCursor,
    /// Evicted rows, oldest first, capped at [`SCROLLBACK_LINES`].
    scrollback: Vec<Vec<Cell>>,
    /// Cursor row, 0-based.
    cursor_row: u16,
    /// Cursor col, 0-based.  May equal `cols` to mean "past the right
    /// edge"; the next printable byte triggers a wrap.
    cursor_col: u16,
    /// Active foreground colour.
    fg: u32,
    /// Active background colour.
    bg: u32,
    /// Phase 69 Track F — current cursor shape (DECSCUSR).
    cursor_shape: CursorShape,
    /// Phase 69 Track G — `?2004` bracketed-paste enabled bit.
    bracketed_paste_enabled: bool,
    /// Phase 69d-FU — DECSTBM scroll region top (0-based, inclusive).
    /// Defaults to `0`.  Less and other TUI apps narrow this with `csr`
    /// so subsequent newlines / SU / SD / IL / DL operate on a
    /// sub-rectangle of the screen.
    scroll_top: u16,
    /// Phase 69d-FU — DECSTBM scroll region bottom (0-based, inclusive).
    /// Defaults to `rows - 1`.
    scroll_bottom: u16,
    /// ANSI parser state.
    parser: AnsiParser,
    /// Phase 69b Track B.2 — UTF-8 byte-stream decoder. Bytes are
    /// pushed through this decoder before reaching the ANSI parser;
    /// `Pending` short-circuits the feed (waiting for more bytes),
    /// `Codepoint(c)` is routed through the parser, and `Invalid` is
    /// remapped to U+FFFD per the W3C replacement-character rule.
    decoder: Utf8Decoder,
}

impl Screen {
    /// Create a new screen with the documented default geometry
    /// ([`DEFAULT_COLS`] × [`DEFAULT_ROWS`]).
    pub fn new() -> Self {
        Self::with_geometry(DEFAULT_COLS, DEFAULT_ROWS)
    }

    /// Create a screen with the supplied geometry. Used by tests; the
    /// production binary always calls [`Screen::new`].
    pub fn with_geometry(cols: u16, rows: u16) -> Self {
        let total = cols as usize * rows as usize;
        let buf = alloc::vec![Cell::blank(DEFAULT_FG, DEFAULT_BG); total];
        let alt_buf = alloc::vec![Cell::blank(DEFAULT_FG, DEFAULT_BG); total];
        Self {
            cols,
            rows,
            buf,
            alt_buf,
            active: ScreenSelect::Primary,
            saved_cursor: SavedCursor::default(),
            scrollback: Vec::new(),
            cursor_row: 0,
            cursor_col: 0,
            fg: DEFAULT_FG,
            bg: DEFAULT_BG,
            cursor_shape: CursorShape::BlinkingBlock,
            bracketed_paste_enabled: false,
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            parser: AnsiParser::new(),
            decoder: Utf8Decoder::new(),
        }
    }

    /// Phase 69 Track B — which cell grid is active.
    pub fn active(&self) -> ScreenSelect {
        self.active
    }

    /// Phase 69 Track F — current cursor shape.
    pub fn cursor_shape(&self) -> CursorShape {
        self.cursor_shape
    }

    /// Phase 69 Track G — `?2004` bracketed-paste mode bit.
    pub fn bracketed_paste_enabled(&self) -> bool {
        self.bracketed_paste_enabled
    }

    fn active_buf(&self) -> &[Cell] {
        match self.active {
            ScreenSelect::Primary => &self.buf,
            ScreenSelect::Alt => &self.alt_buf,
        }
    }

    fn active_buf_mut(&mut self) -> &mut [Cell] {
        match self.active {
            ScreenSelect::Primary => &mut self.buf,
            ScreenSelect::Alt => &mut self.alt_buf,
        }
    }

    /// Phase 69 Track B — enter the alternate screen. Saves the
    /// primary cursor + colours into [`SavedCursor`], clears the
    /// alternate grid, and resets the cursor to the top-left.
    /// A second call while already on the alternate screen is a no-op.
    pub fn switch_to_alt(&mut self, out: &mut Vec<RenderCommand>) {
        if matches!(self.active, ScreenSelect::Alt) {
            return;
        }
        self.saved_cursor = SavedCursor {
            row: self.cursor_row,
            col: self.cursor_col,
            fg: self.fg,
            bg: self.bg,
        };
        self.active = ScreenSelect::Alt;
        for cell in self.alt_buf.iter_mut() {
            *cell = Cell::blank(self.fg, self.bg);
        }
        self.cursor_row = 0;
        self.cursor_col = 0;
        out.push(RenderCommand::Clear);
        out.push(RenderCommand::MoveCursor {
            row: self.cursor_row,
            col: self.cursor_col,
        });
    }

    /// Phase 69 Track B — leave the alternate screen, restoring the
    /// primary cursor + colours saved by [`Screen::switch_to_alt`].
    /// A call while already on the primary is a no-op.
    pub fn switch_to_primary(&mut self, out: &mut Vec<RenderCommand>) {
        if matches!(self.active, ScreenSelect::Primary) {
            return;
        }
        self.active = ScreenSelect::Primary;
        self.fg = self.saved_cursor.fg;
        self.bg = self.saved_cursor.bg;
        self.cursor_row = self.saved_cursor.row.min(self.rows.saturating_sub(1));
        self.cursor_col = self.saved_cursor.col.min(self.cols);
        out.push(RenderCommand::SetColor {
            fg: self.fg,
            bg: self.bg,
        });
        out.push(RenderCommand::Clear);
        // Re-emit every primary cell so the renderer repaints what was
        // hidden behind the alternate screen.
        let cols = self.cols;
        let rows = self.rows;
        for row in 0..rows {
            for col in 0..cols {
                let idx = row as usize * cols as usize + col as usize;
                let cell = self.buf[idx];
                out.push(RenderCommand::PutGlyph {
                    row,
                    col,
                    codepoint: cell.codepoint,
                    fg: cell.fg,
                    bg: cell.bg,
                });
            }
        }
        out.push(RenderCommand::MoveCursor {
            row: self.cursor_row,
            col: self.cursor_col,
        });
    }

    /// Phase 69 Track D — react to a SIGWINCH-style resize. Reallocates
    /// both grids, clamps the cursor to the new bounds, blanks any new
    /// cells, and pushes a [`RenderCommand::Clear`] hint so callers
    /// repaint the full surface.
    pub fn resize(&mut self, cols: u16, rows: u16, out: &mut Vec<RenderCommand>) {
        if cols == 0 || rows == 0 {
            return;
        }
        let total = cols as usize * rows as usize;
        let mut new_buf = alloc::vec![Cell::blank(self.fg, self.bg); total];
        let mut new_alt = alloc::vec![Cell::blank(self.fg, self.bg); total];
        let copy_cols = cols.min(self.cols) as usize;
        let copy_rows = rows.min(self.rows) as usize;
        for r in 0..copy_rows {
            for c in 0..copy_cols {
                let src = r * self.cols as usize + c;
                let dst = r * cols as usize + c;
                new_buf[dst] = self.buf[src];
                new_alt[dst] = self.alt_buf[src];
            }
        }
        self.buf = new_buf;
        self.alt_buf = new_alt;
        self.cols = cols;
        self.rows = rows;
        if self.cursor_row >= rows {
            self.cursor_row = rows.saturating_sub(1);
        }
        if self.cursor_col > cols {
            self.cursor_col = cols;
        }
        // Phase 69d-FU — reclamp the scroll region.  Any saved bounds
        // outside the new geometry collapse to the full screen so a
        // subsequent line_feed cannot reference a row we no longer have.
        self.scroll_top = self.scroll_top.min(rows.saturating_sub(1));
        self.scroll_bottom = self.scroll_bottom.min(rows.saturating_sub(1));
        if self.scroll_top >= self.scroll_bottom {
            self.scroll_top = 0;
            self.scroll_bottom = rows.saturating_sub(1);
        }
        out.push(RenderCommand::Clear);
        let cols_now = self.cols;
        let rows_now = self.rows;
        for row in 0..rows_now {
            for col in 0..cols_now {
                let idx = row as usize * cols_now as usize + col as usize;
                let cell = self.active_buf()[idx];
                out.push(RenderCommand::PutGlyph {
                    row,
                    col,
                    codepoint: cell.codepoint,
                    fg: cell.fg,
                    bg: cell.bg,
                });
            }
        }
        out.push(RenderCommand::MoveCursor {
            row: self.cursor_row,
            col: self.cursor_col,
        });
    }

    /// Cell-grid columns.
    pub fn cols(&self) -> u16 {
        self.cols
    }

    /// Cell-grid rows.
    pub fn rows(&self) -> u16 {
        self.rows
    }

    /// Cursor position (row, col), both 0-based.
    pub fn cursor(&self) -> (u16, u16) {
        (self.cursor_row, self.cursor_col)
    }

    /// Active colours, `(fg, bg)`.
    pub fn colors(&self) -> (u32, u32) {
        (self.fg, self.bg)
    }

    /// Number of evicted lines currently in the scrollback ring.
    pub fn scrollback_len(&self) -> usize {
        self.scrollback.len()
    }

    /// Read one cell from the currently active grid. Returns
    /// [`ScreenError::OutOfBounds`] when `(row, col)` is outside the
    /// grid.
    pub fn cell(&self, row: u16, col: u16) -> Result<Cell, ScreenError> {
        if row >= self.rows || col >= self.cols {
            return Err(ScreenError::OutOfBounds);
        }
        Ok(self.active_buf()[row as usize * self.cols as usize + col as usize])
    }

    /// Phase 69 Track B — read one cell from the primary grid
    /// regardless of which grid is currently active. Used by tests
    /// that verify `?1049l` restores scroll history.
    pub fn cell_primary(&self, row: u16, col: u16) -> Result<Cell, ScreenError> {
        if row >= self.rows || col >= self.cols {
            return Err(ScreenError::OutOfBounds);
        }
        Ok(self.buf[row as usize * self.cols as usize + col as usize])
    }

    /// Feed one byte through the UTF-8 decoder, then the ANSI parser,
    /// and update the screen state. Push the typed render commands
    /// produced into `out`. The function allocates only when scrollback
    /// grows, never per character.
    ///
    /// Phase 69b Track B.2 — bytes are first routed through the
    /// [`Utf8Decoder`]. `Pending` short-circuits the feed (waiting for
    /// the next continuation byte). `Codepoint(c)` is processed by the
    /// parser as if it were a single character — pure-ASCII codepoints
    /// (which is what all escape-sequence bytes are) decode in one
    /// step, so Phase 22b sequence handling is unaffected. `Invalid`
    /// is replaced with U+FFFD per the W3C replacement-character rule
    /// and forwarded to [`Screen::put_char`] directly (escape-sequence
    /// state is preserved across malformed payload bytes — a malformed
    /// byte inside an escape sequence is unusual but should not break
    /// out of the in-flight sequence). BEL (`0x07`) is recognised
    /// inside [`Screen::emit_codepoint`]; routing it through the
    /// decoder first means a BEL that arrives mid-sequence cancels
    /// the in-flight UTF-8 codepoint (emitting U+FFFD) before the
    /// bell rings — otherwise the next continuation byte would
    /// silently complete the broken sequence.
    pub fn feed(&mut self, byte: u8, out: &mut Vec<RenderCommand>) {
        match self.decoder.decode_byte(byte) {
            DecoderOutput::Pending => {}
            DecoderOutput::Codepoint(c) => self.emit_codepoint(c, out),
            DecoderOutput::Invalid => self.emit_codepoint(REPLACEMENT_CHARACTER, out),
            DecoderOutput::InvalidThenCodepoint(c) => {
                // The truncated in-flight sequence yields one U+FFFD,
                // and the offending byte was itself a complete ASCII
                // codepoint — emit both, preserving valid trailing
                // data the strict resync would otherwise drop.
                self.emit_codepoint(REPLACEMENT_CHARACTER, out);
                self.emit_codepoint(c, out);
            }
            DecoderOutput::InvalidThenPending => {
                // The truncated in-flight sequence yields one U+FFFD;
                // the offending byte has already begun a fresh
                // multi-byte sequence in the decoder.
                self.emit_codepoint(REPLACEMENT_CHARACTER, out);
            }
        }
    }

    /// Route a decoded codepoint either through the Phase 22b ANSI
    /// parser (when ASCII — every escape-sequence byte is ASCII) or
    /// directly into [`Screen::put_char`]. Codepoints >= 0x80 cannot
    /// occur inside an ANSI escape, so they bypass the parser; this
    /// also handles the U+FFFD replacement-character path from
    /// malformed UTF-8.
    ///
    /// BEL (U+0007) is mapped to a [`RenderCommand::Bell`] here.
    /// [`Screen::feed`] routes every byte (including 0x07) through
    /// the decoder first, so the BEL guard is the single mapping
    /// site: an isolated `0x07` byte decodes to `Codepoint(0x07)`
    /// and lands here; a `0x07` byte arriving mid-multi-byte
    /// sequence decodes to `InvalidThenCodepoint(0x07)`, in which
    /// case `feed` emits U+FFFD for the cancelled in-flight
    /// sequence before re-entering `emit_codepoint` with `0x07` and
    /// ringing the bell. Without this guard the ASCII path below
    /// would route 0x07 through the parser, which has no BEL
    /// handler and would emit a `PutGlyph` for the control byte.
    fn emit_codepoint(&mut self, codepoint: u32, out: &mut Vec<RenderCommand>) {
        if codepoint == 0x07 {
            out.push(RenderCommand::Bell);
            return;
        }
        if codepoint <= 0x7F {
            let ch = (codepoint as u8) as char;
            let cmd = self.parser.process_char(ch);
            self.dispatch_console_cmd(cmd, out);
        } else {
            self.put_char(codepoint, out);
        }
    }

    fn dispatch_console_cmd(&mut self, cmd: ConsoleCmd, out: &mut Vec<RenderCommand>) {
        match cmd {
            ConsoleCmd::Nop => {}
            ConsoleCmd::PutChar(codepoint) => self.put_char(codepoint, out),
            ConsoleCmd::CarriageReturn => {
                self.cursor_col = 0;
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::Newline => self.line_feed(out),
            ConsoleCmd::Backspace => {
                if self.cursor_col > 0 {
                    self.cursor_col -= 1;
                    self.blank_cell(self.cursor_row, self.cursor_col, out);
                    out.push(RenderCommand::MoveCursor {
                        row: self.cursor_row,
                        col: self.cursor_col,
                    });
                }
            }
            ConsoleCmd::Tab => {
                let next = ((self.cursor_col / 8) + 1) * 8;
                self.cursor_col = next.min(self.cols);
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorUp(n) => {
                self.cursor_row = self.cursor_row.saturating_sub(n);
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorDown(n) => {
                self.cursor_row = (self.cursor_row + n).min(self.rows.saturating_sub(1));
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorForward(n) => {
                self.cursor_col = (self.cursor_col + n).min(self.cols);
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorBack(n) => {
                self.cursor_col = self.cursor_col.saturating_sub(n);
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorHorizontalAbsolute(col_1based) => {
                let col = col_1based
                    .saturating_sub(1)
                    .min(self.cols.saturating_sub(1));
                self.cursor_col = col;
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::CursorPosition(r_1based, c_1based) => {
                let r = r_1based.saturating_sub(1).min(self.rows.saturating_sub(1));
                let c = c_1based.saturating_sub(1).min(self.cols.saturating_sub(1));
                self.cursor_row = r;
                self.cursor_col = c;
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::EraseDisplay(2) => {
                self.clear_buffer();
                self.cursor_row = 0;
                self.cursor_col = 0;
                out.push(RenderCommand::Clear);
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            // ED 0 (`ESC [ J` with no param or `ESC [ 0 J`): erase from
            // the cursor (inclusive) to the end of the display. ion
            // emits `\r\x1b[J<new line>` on every keystroke to redraw
            // the prompt, so dropping this leaves stale glyphs from any
            // longer prior content (the "backspace doesn't erase" /
            // "shorter history line leaves trailing chars" symptoms).
            ConsoleCmd::EraseDisplay(0) => self.erase_display_to_end(out),
            // ED 1: erase from the start of the display to the cursor
            // (inclusive). Less common in shells but cheap to handle
            // alongside ED 0.
            ConsoleCmd::EraseDisplay(1) => self.erase_display_to_cursor(out),
            ConsoleCmd::EraseDisplay(_) => { /* unsupported mode — keep the screen */ }
            ConsoleCmd::EraseLine(mode) => self.erase_line(mode, out),
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                // Phase 69 Track B — a single CSI may carry multiple
                // semicolon-separated codes (e.g. the terminfo `XM`
                // capability emits `\E[?1006;1000h`). Apply each one in
                // order so multi-mode toggles land atomically.
                for &code in &codes[..count.min(codes.len())] {
                    self.handle_dec_private(code, set, out);
                }
            }
            ConsoleCmd::CursorShape { shape } => {
                self.cursor_shape = CursorShape::from_code(shape);
            }
            ConsoleCmd::Sgr(sgr) => self.apply_sgr_ops(sgr.ops(), out),
            ConsoleCmd::SetScrollRegion { top, bottom } => {
                self.set_scroll_region(top, bottom, out);
            }
            ConsoleCmd::VerticalPositionAbsolute(n) => {
                let r = n.saturating_sub(1).min(self.rows.saturating_sub(1));
                self.cursor_row = r;
                out.push(RenderCommand::MoveCursor {
                    row: self.cursor_row,
                    col: self.cursor_col,
                });
            }
            ConsoleCmd::ScrollUp(n) => self.scroll_region_up(n.max(1), out),
            ConsoleCmd::ScrollDown(n) => self.scroll_region_down(n.max(1), out),
            ConsoleCmd::InsertLines(n) => self.insert_lines(n.max(1), out),
            ConsoleCmd::DeleteLines(n) => self.delete_lines(n.max(1), out),
            ConsoleCmd::InsertChars(n) => self.insert_chars(n.max(1), out),
            ConsoleCmd::DeleteChars(n) => self.delete_chars(n.max(1), out),
            ConsoleCmd::EraseChars(n) => self.erase_chars(n.max(1), out),
        }
    }

    fn handle_dec_private(&mut self, code: u16, set: bool, out: &mut Vec<RenderCommand>) {
        match code {
            // DECTCEM — cursor visibility. The renderer reads the
            // cursor shape directly today; visibility is policy and
            // gets a no-op here so a script that hides the cursor
            // does not crash.
            25 => {}
            // Alternate-screen buffer. Both `?1049` and the legacy
            // `?47` route through the same `switch_to_alt` /
            // `switch_to_primary` path: cursor + colours are saved
            // on enter and restored on exit. xterm historically
            // exposed `?47` as the unbuffered switch (no save), but
            // m3os-term ships a single buffered path so that
            // applications mixing the two codes see consistent
            // round-trip behaviour. A true `?47` (no save/restore)
            // path is deferred — see Phase 69 deferrals.
            1049 | 47 => {
                if set {
                    self.switch_to_alt(out);
                } else {
                    self.switch_to_primary(out);
                }
            }
            // Bracketed paste — owned by `Screen::feed` so future code
            // that consults `bracketed_paste_enabled()` always agrees
            // with the latest state machine update.
            2004 => {
                self.bracketed_paste_enabled = set;
            }
            // Mouse modes (?9 / ?1000 / ?1006). The MouseReporter
            // lives in the binary main loop; Screen forwards the
            // mode-change as an out-of-band RenderCommand so the
            // main loop can keep its reporter state in sync without
            // a back-channel.
            9 | 1000 | 1006 | 1002 | 1003 => {
                out.push(RenderCommand::SetMouseMode { code, set });
            }
            _ => {}
        }
    }

    fn put_char(&mut self, codepoint: u32, out: &mut Vec<RenderCommand>) {
        // Phase 69b Track F — wide-glyph accounting. A double-width
        // codepoint occupies `(row, col)` *and* `(row, col + 1)`. If
        // only one column remains on the current row, wrap to the next
        // row before placing the glyph (matches xterm's wrap policy).
        let width = width_of(codepoint).max(1);
        if self.cursor_col >= self.cols {
            self.line_feed(out);
            self.cursor_col = 0;
        }
        if width == 2 && self.cursor_col + 1 >= self.cols {
            // Not enough room on this row — leave the trailing cell
            // blank and wrap so the glyph lands intact on the next row.
            // The blank cell at `(row, cols-1)` keeps the cell grid
            // consistent for the renderer; we paint it explicitly so
            // the framebuffer drops any prior content there.
            let row = self.cursor_row;
            let col = self.cursor_col;
            self.blank_cell(row, col, out);
            self.line_feed(out);
            self.cursor_col = 0;
        }
        let row = self.cursor_row;
        let col = self.cursor_col;
        let cols = self.cols;
        let idx = row as usize * cols as usize + col as usize;
        let fg = self.fg;
        let bg = self.bg;
        // Phase 69b Track F — if this cell is currently the trailing
        // half of a wide glyph (`wide_continuation == true`), the
        // leader at `(row, col - 1)` is the surviving pixel source.
        // Blank the leader so the renderer drops its stale pixels
        // before we paint the new glyph here. The continuation flag
        // on this cell is cleared a few lines below when we write
        // the new `Cell` value.
        let target_is_wide_trail = self.active_buf()[idx].wide_continuation;
        if target_is_wide_trail && col > 0 {
            self.blank_cell(row, col - 1, out);
        }
        // Phase 69b Track F — if the cell currently at `(row, col+1)`
        // is the trailing half of an existing wide glyph at
        // `(row, col)`, the renderer's pixels there are stale once we
        // overwrite the leader. Blank the trail and clear its
        // continuation flag so the new glyph paints cleanly.
        let trailing_is_continuation = (col + 1) < cols
            && self.active_buf()[row as usize * cols as usize + (col + 1) as usize]
                .wide_continuation;
        if trailing_is_continuation {
            self.blank_cell(row, col + 1, out);
        }
        self.active_buf_mut()[idx] = Cell {
            codepoint,
            fg,
            bg,
            wide_continuation: false,
        };
        out.push(RenderCommand::PutGlyph {
            row,
            col,
            codepoint,
            fg,
            bg,
        });
        self.cursor_col += 1;
        if width == 2 {
            // Reserve the trailing cell as a continuation, AND emit a
            // blank `PutGlyph` for it. The leading `PutGlyph` only
            // paints a single 8×16 cell (the renderer has no
            // wide-glyph protocol), so the trail column's previous
            // pixels would otherwise stay visible underneath the
            // wide-cell pair. The blank render command paints the
            // background colour across the trail cell; the buffer
            // entry retains `wide_continuation: true` so wide-cell
            // accounting remains correct.
            let trail_col = self.cursor_col;
            if trail_col < self.cols {
                let trail_idx = row as usize * self.cols as usize + trail_col as usize;
                // If the cell about to be overwritten by this trail is
                // itself a wide *leader* (a real glyph with width 2
                // and no continuation flag), its own trailing cell at
                // `trail_col + 1` would otherwise remain as an orphan
                // `wide_continuation`. Blank that orphan first so the
                // grid stays consistent.
                let displaced = self.active_buf()[trail_idx];
                let displaced_is_wide_leader = !displaced.wide_continuation
                    && displaced.codepoint != 0
                    && width_of(displaced.codepoint) == 2;
                if displaced_is_wide_leader && (trail_col + 1) < self.cols {
                    self.blank_cell(row, trail_col + 1, out);
                }
                // Paint the trail cell as a blank space so previous
                // pixels are overwritten with bg.
                out.push(RenderCommand::PutGlyph {
                    row,
                    col: trail_col,
                    codepoint: b' ' as u32,
                    fg,
                    bg,
                });
                self.active_buf_mut()[trail_idx] = Cell {
                    codepoint: 0,
                    fg,
                    bg,
                    wide_continuation: true,
                };
            }
            self.cursor_col = self.cursor_col.saturating_add(1).min(self.cols);
        }
    }

    fn line_feed(&mut self, out: &mut Vec<RenderCommand>) {
        self.cursor_col = 0;
        // Honour the DECSTBM scroll region: when the cursor sits on
        // the bottom margin of the region, a line-feed scrolls the
        // region up by one and the cursor stays put.  Outside the
        // region (cursor below scroll_bottom) the cursor simply
        // advances until it hits the bottom of the surface.
        if self.cursor_row == self.scroll_bottom {
            self.scroll_region_up(1, out);
        } else if self.cursor_row + 1 < self.rows {
            self.cursor_row += 1;
        }
        out.push(RenderCommand::MoveCursor {
            row: self.cursor_row,
            col: self.cursor_col,
        });
    }

    /// Phase 69d-FU — scroll the active scroll region up by `n` lines.
    /// The top `n` rows of the region are lost (with scrollback eviction
    /// only when the region covers the full primary surface).  The
    /// bottom `n` rows of the region are blanked.
    fn scroll_region_up(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 || self.scroll_top > self.scroll_bottom {
            return;
        }
        let region_height = self.scroll_bottom - self.scroll_top + 1;
        let n = n.min(region_height);
        let top = self.scroll_top as usize;
        let bot = self.scroll_bottom as usize;
        let cols = self.cols as usize;
        let fg = self.fg;
        let bg = self.bg;
        let primary = matches!(self.active, ScreenSelect::Primary);
        let full_screen = self.scroll_top == 0 && self.scroll_bottom + 1 == self.rows;
        // Evict only when the region is the full primary surface — the
        // alternate screen and partial regions do not feed scrollback
        // per xterm.
        if primary && full_screen {
            for r in 0..n as usize {
                let evicted: Vec<Cell> = self.buf[r * cols..(r + 1) * cols].to_vec();
                if self.scrollback.len() >= SCROLLBACK_LINES {
                    self.scrollback.remove(0);
                }
                self.scrollback.push(evicted);
            }
        }
        let active = self.active_buf_mut();
        // Shift rows up within the region: row `r` <- row `r + n`.
        for r in top..=(bot - n as usize) {
            let src = r + n as usize;
            for c in 0..cols {
                active[r * cols + c] = active[src * cols + c];
            }
        }
        // Blank the bottom `n` rows of the region.
        for r in (bot + 1 - n as usize)..=bot {
            for c in 0..cols {
                active[r * cols + c] = Cell::blank(fg, bg);
            }
        }
        // Re-emit the affected cells.  The full-screen case can use
        // the fast `Scroll` framebuffer command; partial regions need
        // per-cell PutGlyph since the framebuffer scroll is whole-surface.
        if full_screen {
            out.push(RenderCommand::Scroll { amount: n as i16 });
        } else {
            self.emit_region(top, bot, out);
        }
    }

    /// Phase 69d-FU — scroll the active scroll region down by `n` lines.
    /// The bottom `n` rows of the region are lost; the top `n` rows are
    /// blanked.  Scrollback is never fed (xterm semantics).
    fn scroll_region_down(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 || self.scroll_top > self.scroll_bottom {
            return;
        }
        let region_height = self.scroll_bottom - self.scroll_top + 1;
        let n = n.min(region_height);
        let top = self.scroll_top as usize;
        let bot = self.scroll_bottom as usize;
        let cols = self.cols as usize;
        let fg = self.fg;
        let bg = self.bg;
        let active = self.active_buf_mut();
        // Shift rows down within the region: row `r` <- row `r - n`,
        // iterating from the bottom up so we don't clobber sources.
        for r in (top + n as usize..=bot).rev() {
            let src = r - n as usize;
            for c in 0..cols {
                active[r * cols + c] = active[src * cols + c];
            }
        }
        // Blank the top `n` rows of the region.
        for r in top..top + n as usize {
            for c in 0..cols {
                active[r * cols + c] = Cell::blank(fg, bg);
            }
        }
        self.emit_region(top, bot, out);
    }

    /// Phase 69d-FU — re-emit every cell in rows `[top, bot]` as PutGlyph
    /// commands so the renderer repaints the region after an in-buffer
    /// shift.  Used by partial-region scrolls and by IL / DL.
    fn emit_region(&self, top: usize, bot: usize, out: &mut Vec<RenderCommand>) {
        let cols = self.cols as usize;
        let buf = self.active_buf();
        for r in top..=bot {
            for c in 0..cols {
                let cell = buf[r * cols + c];
                out.push(RenderCommand::PutGlyph {
                    row: r as u16,
                    col: c as u16,
                    codepoint: cell.codepoint,
                    fg: cell.fg,
                    bg: cell.bg,
                });
            }
        }
    }

    /// Phase 69d-FU — re-emit one row's cells as PutGlyphs.  Used by
    /// ICH / DCH / ECH after an in-place row mutation.
    fn emit_row(&self, row: usize, out: &mut Vec<RenderCommand>) {
        let cols = self.cols as usize;
        let buf = self.active_buf();
        for c in 0..cols {
            let cell = buf[row * cols + c];
            out.push(RenderCommand::PutGlyph {
                row: row as u16,
                col: c as u16,
                codepoint: cell.codepoint,
                fg: cell.fg,
                bg: cell.bg,
            });
        }
    }

    /// Phase 69d-FU — set the DECSTBM scroll region.  Both bounds are
    /// 1-based in the wire protocol; passing `(0, 0)` resets to the
    /// full screen.  Bounds outside the screen are clamped; an
    /// inverted region (`top >= bottom` after clamping) also resets.
    fn set_scroll_region(&mut self, top: u16, bottom: u16, out: &mut Vec<RenderCommand>) {
        let (new_top, new_bottom) = if top == 0 && bottom == 0 {
            (0u16, self.rows.saturating_sub(1))
        } else {
            let t = top.saturating_sub(1).min(self.rows.saturating_sub(1));
            let b = bottom.saturating_sub(1).min(self.rows.saturating_sub(1));
            if t >= b {
                (0u16, self.rows.saturating_sub(1))
            } else {
                (t, b)
            }
        };
        self.scroll_top = new_top;
        self.scroll_bottom = new_bottom;
        // DECSTBM moves the cursor to (1,1) per the standard.
        self.cursor_row = 0;
        self.cursor_col = 0;
        out.push(RenderCommand::MoveCursor {
            row: self.cursor_row,
            col: self.cursor_col,
        });
    }

    /// Phase 69d-FU — IL: insert `n` blank lines at the cursor (or
    /// rather at `cursor_row`), shifting content within the scroll
    /// region downward.  Lines pushed past `scroll_bottom` are lost.
    /// No-op when the cursor is outside the scroll region.
    fn insert_lines(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 {
            return;
        }
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let top = self.cursor_row;
        let saved_top = self.scroll_top;
        self.scroll_top = top;
        self.scroll_region_down(n, out);
        self.scroll_top = saved_top;
    }

    /// Phase 69d-FU — DL: delete `n` lines starting at the cursor,
    /// shifting content within the scroll region upward.  Bottom `n`
    /// lines blanked.  No-op when the cursor is outside the scroll
    /// region.
    fn delete_lines(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 {
            return;
        }
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let top = self.cursor_row;
        let saved_top = self.scroll_top;
        self.scroll_top = top;
        self.scroll_region_up(n, out);
        self.scroll_top = saved_top;
    }

    /// Phase 69d-FU — ICH: insert `n` blank cells at the cursor,
    /// shifting cells right.  Cells pushed past the last column are
    /// lost.
    fn insert_chars(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 || self.cursor_col >= self.cols {
            return;
        }
        let n = n.min(self.cols - self.cursor_col);
        let row = self.cursor_row as usize;
        let cols = self.cols as usize;
        let start = self.cursor_col as usize;
        let fg = self.fg;
        let bg = self.bg;
        let buf = self.active_buf_mut();
        // Shift right: iterate from end so we don't clobber sources.
        for c in (start + n as usize..cols).rev() {
            buf[row * cols + c] = buf[row * cols + c - n as usize];
        }
        for c in start..start + n as usize {
            buf[row * cols + c] = Cell::blank(fg, bg);
        }
        self.emit_row(row, out);
    }

    /// Phase 69d-FU — DCH: delete `n` cells at the cursor, shifting
    /// the rest of the row left.  Last `n` cells of the row are blanked.
    fn delete_chars(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 || self.cursor_col >= self.cols {
            return;
        }
        let n = n.min(self.cols - self.cursor_col);
        let row = self.cursor_row as usize;
        let cols = self.cols as usize;
        let start = self.cursor_col as usize;
        let fg = self.fg;
        let bg = self.bg;
        let buf = self.active_buf_mut();
        for c in start..cols - n as usize {
            buf[row * cols + c] = buf[row * cols + c + n as usize];
        }
        for c in cols - n as usize..cols {
            buf[row * cols + c] = Cell::blank(fg, bg);
        }
        self.emit_row(row, out);
    }

    /// Phase 69d-FU — ECH: blank `n` cells starting at the cursor;
    /// cursor position unchanged.
    fn erase_chars(&mut self, n: u16, out: &mut Vec<RenderCommand>) {
        if n == 0 || self.cursor_col >= self.cols {
            return;
        }
        let n = n.min(self.cols - self.cursor_col);
        let row = self.cursor_row;
        let start_col = self.cursor_col;
        for c in start_col..start_col + n {
            self.blank_cell(row, c, out);
        }
    }

    /// ED 0: blank from `(cursor_row, cursor_col)` (inclusive) to the
    /// bottom-right corner of the grid. Cursor position is unchanged
    /// (per ANSI ED semantics — the shell positions the cursor before
    /// the erase, not after).
    fn erase_display_to_end(&mut self, out: &mut Vec<RenderCommand>) {
        if self.rows == 0 || self.cols == 0 || self.cursor_row >= self.rows {
            return;
        }
        let row = self.cursor_row;
        let start_col = self.cursor_col.min(self.cols);
        for col in start_col..self.cols {
            self.blank_cell(row, col, out);
        }
        for r in (row + 1)..self.rows {
            for col in 0..self.cols {
                self.blank_cell(r, col, out);
            }
        }
    }

    /// ED 1: blank from the top-left corner to `(cursor_row,
    /// cursor_col)` (inclusive). Cursor position is unchanged.
    fn erase_display_to_cursor(&mut self, out: &mut Vec<RenderCommand>) {
        if self.rows == 0 || self.cols == 0 || self.cursor_row >= self.rows {
            return;
        }
        let row = self.cursor_row;
        for r in 0..row {
            for col in 0..self.cols {
                self.blank_cell(r, col, out);
            }
        }
        let end = self.cursor_col.min(self.cols.saturating_sub(1));
        for col in 0..=end {
            self.blank_cell(row, col, out);
        }
    }

    fn erase_line(&mut self, mode: u16, out: &mut Vec<RenderCommand>) {
        if self.rows == 0 || self.cols == 0 || self.cursor_row >= self.rows {
            return;
        }
        let row = self.cursor_row;
        let cursor_col = self.cursor_col.min(self.cols);
        match mode {
            0 => {
                for col in cursor_col..self.cols {
                    self.blank_cell(row, col, out);
                }
            }
            1 => {
                let end = cursor_col.min(self.cols.saturating_sub(1));
                for col in 0..=end {
                    self.blank_cell(row, col, out);
                }
            }
            2 => {
                for col in 0..self.cols {
                    self.blank_cell(row, col, out);
                }
            }
            _ => {}
        }
    }

    fn blank_cell(&mut self, row: u16, col: u16, out: &mut Vec<RenderCommand>) {
        if row >= self.rows || col >= self.cols {
            return;
        }
        let idx = row as usize * self.cols as usize + col as usize;
        let fg = self.fg;
        let bg = self.bg;
        self.active_buf_mut()[idx] = Cell::blank(fg, bg);
        out.push(RenderCommand::PutGlyph {
            row,
            col,
            codepoint: b' ' as u32,
            fg,
            bg,
        });
    }

    fn clear_buffer(&mut self) {
        let fg = self.fg;
        let bg = self.bg;
        for cell in self.active_buf_mut().iter_mut() {
            *cell = Cell::blank(fg, bg);
        }
    }

    fn apply_sgr_ops<I>(&mut self, ops: I, out: &mut Vec<RenderCommand>)
    where
        I: IntoIterator<Item = SgrOp>,
    {
        let mut changed = false;
        for op in ops {
            let (new_fg, new_bg) = match op {
                SgrOp::Reset => (Some(DEFAULT_FG), Some(DEFAULT_BG)),
                SgrOp::FgDefault => (Some(DEFAULT_FG), None),
                SgrOp::BgDefault => (None, Some(DEFAULT_BG)),
                SgrOp::Fg8(i) => (Some(color_to_bgra(Color::Standard(i))), None),
                SgrOp::Bg8(i) => (None, Some(color_to_bgra(Color::Standard(i)))),
                SgrOp::FgBright8(i) => (Some(color_to_bgra(Color::Bright(i))), None),
                SgrOp::BgBright8(i) => (None, Some(color_to_bgra(Color::Bright(i)))),
                SgrOp::FgIndexed(i) => (Some(color_to_bgra(Color::Indexed(i))), None),
                SgrOp::BgIndexed(i) => (None, Some(color_to_bgra(Color::Indexed(i)))),
                SgrOp::FgRgb(r, g, b) => (Some(color_to_bgra(Color::Rgb(r, g, b))), None),
                SgrOp::BgRgb(r, g, b) => (None, Some(color_to_bgra(Color::Rgb(r, g, b)))),
                // Bold / underline / reverse are decorative-only today;
                // the renderer does not yet implement them, so we
                // record nothing and the call site rolls forward.
                _ => (None, None),
            };
            if let Some(fg) = new_fg {
                if self.fg != fg {
                    self.fg = fg;
                    changed = true;
                }
            }
            if let Some(bg) = new_bg {
                if self.bg != bg {
                    self.bg = bg;
                    changed = true;
                }
            }
        }
        if changed {
            out.push(RenderCommand::SetColor {
                fg: self.fg,
                bg: self.bg,
            });
        }
    }
}

impl Default for Screen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn feed_str(screen: &mut Screen, s: &str) -> Vec<RenderCommand> {
        let mut out = Vec::new();
        for b in s.as_bytes() {
            screen.feed(*b, &mut out);
        }
        out
    }

    /// Phase 57 G.4 acceptance: default geometry is the documented
    /// 80×25 grid.
    #[test]
    fn default_geometry_matches_constants() {
        let s = Screen::new();
        assert_eq!(s.cols(), DEFAULT_COLS);
        assert_eq!(s.rows(), DEFAULT_ROWS);
    }

    /// Phase 57 G.4 acceptance: feeding plain ASCII produces a
    /// `PutGlyph` per character with the active colours and advances
    /// the cursor by one column.
    #[test]
    fn put_glyph_advances_cursor() {
        let mut s = Screen::with_geometry(80, 25);
        let cmds = feed_str(&mut s, "Hi");
        assert_eq!(cmds.len(), 2);
        let first = cmds[0];
        match first {
            RenderCommand::PutGlyph {
                row,
                col,
                codepoint,
                ..
            } => {
                assert_eq!(row, 0);
                assert_eq!(col, 0);
                assert_eq!(codepoint, b'H' as u32);
            }
            _ => panic!("expected PutGlyph, got {:?}", first),
        }
        match cmds[1] {
            RenderCommand::PutGlyph {
                row,
                col,
                codepoint,
                ..
            } => {
                assert_eq!(row, 0);
                assert_eq!(col, 1);
                assert_eq!(codepoint, b'i' as u32);
            }
            _ => panic!("expected PutGlyph, got {:?}", cmds[1]),
        }
        assert_eq!(s.cursor(), (0, 2));
    }

    /// Phase 57 G.4 acceptance: BEL (0x07) maps to a single
    /// `RenderCommand::Bell` and never advances the cursor.
    #[test]
    fn bel_maps_to_bell_command() {
        let mut s = Screen::with_geometry(80, 25);
        let mut out = Vec::new();
        s.feed(0x07, &mut out);
        assert_eq!(out.as_slice(), &[RenderCommand::Bell]);
        assert_eq!(s.cursor(), (0, 0));
    }

    /// Phase 57 G.4 acceptance: newline advances to the next row.
    #[test]
    fn newline_advances_row() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "A\n");
        assert_eq!(s.cursor(), (1, 0));
    }

    /// Phase 57 G.4 acceptance: carriage return resets the column.
    #[test]
    fn carriage_return_resets_col() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "ABC\r");
        assert_eq!(s.cursor(), (0, 0));
    }

    #[test]
    fn backspace_erases_previous_cell() {
        let mut s = Screen::with_geometry(8, 2);
        let cmds = feed_str(&mut s, "AB\x08");

        assert_eq!(s.cursor(), (0, 1));
        assert_eq!(s.cell(0, 1).unwrap().codepoint, b' ' as u32);
        assert!(
            cmds.iter().any(|cmd| {
                matches!(
                    cmd,
                    RenderCommand::PutGlyph {
                        row: 0,
                        col: 1,
                        codepoint,
                        ..
                    } if *codepoint == b' ' as u32
                )
            }),
            "backspace must repaint the erased cell so stale glyphs disappear"
        );
    }

    #[test]
    fn erase_line_zero_clears_from_cursor_to_end() {
        let mut s = Screen::with_geometry(5, 2);
        let cmds = feed_str(&mut s, "ABCDE\r\x1b[2C\x1b[K");

        assert_eq!(s.cell(0, 0).unwrap().codepoint, b'A' as u32);
        assert_eq!(s.cell(0, 1).unwrap().codepoint, b'B' as u32);
        for col in 2..5 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b' ' as u32);
        }
        let erased = cmds
            .iter()
            .filter(|cmd| {
                matches!(
                    cmd,
                    RenderCommand::PutGlyph {
                        row: 0,
                        col: 2..=4,
                        codepoint,
                        ..
                    } if *codepoint == b' ' as u32
                )
            })
            .count();
        assert_eq!(erased, 3, "erase-line must repaint every cleared cell");
    }

    #[test]
    fn erase_line_one_clears_start_through_cursor() {
        let mut s = Screen::with_geometry(5, 2);
        let _ = feed_str(&mut s, "ABCDE\r\x1b[2C\x1b[1K");

        for col in 0..=2 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b' ' as u32);
        }
        assert_eq!(s.cell(0, 3).unwrap().codepoint, b'D' as u32);
        assert_eq!(s.cell(0, 4).unwrap().codepoint, b'E' as u32);
    }

    #[test]
    fn erase_line_two_clears_entire_line() {
        let mut s = Screen::with_geometry(5, 2);
        let _ = feed_str(&mut s, "ABCDE\r\x1b[2C\x1b[2K");

        for col in 0..5 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b' ' as u32);
        }
        assert_eq!(s.cursor(), (0, 2));
    }

    /// Phase 57 G.4 acceptance: writing past the right edge wraps to
    /// the next row.
    #[test]
    fn line_wrap_advances_row() {
        let mut s = Screen::with_geometry(4, 2);
        let _ = feed_str(&mut s, "ABCDE");
        // After 'A','B','C','D' cursor is (0, 4) which means "past
        // the right edge"; the next character ('E') wraps to (1, 1).
        let (row, col) = s.cursor();
        assert!(row >= 1, "expected row >= 1 after wrap, got {row}");
        let _ = col;
    }

    /// Phase 57 G.4 acceptance: writing past the last row scrolls and
    /// pushes the evicted row into scrollback.
    #[test]
    fn scroll_evicts_to_scrollback() {
        let mut s = Screen::with_geometry(4, 2);
        // Fill row 0 then row 1, then trigger another newline.
        let _ = feed_str(&mut s, "ABCD\nEFGH\n");
        // The second newline must scroll once.
        assert!(s.scrollback_len() >= 1);
    }

    /// Phase 57 G.4 acceptance: scrollback caps at SCROLLBACK_LINES;
    /// exceeding the cap drops the oldest line.
    #[test]
    fn scrollback_cap_drops_oldest() {
        let mut s = Screen::with_geometry(4, 2);
        // Force more lines than the cap.
        for _ in 0..(SCROLLBACK_LINES + 5) {
            let _ = feed_str(&mut s, "ABCD\n");
        }
        assert_eq!(s.scrollback_len(), SCROLLBACK_LINES);
    }

    /// Phase 57 G.4 acceptance: SGR 31 (red foreground) updates the
    /// active fg colour.
    #[test]
    fn sgr_red_changes_fg() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "\x1b[31m");
        let (fg, _bg) = s.colors();
        assert_ne!(fg, DEFAULT_FG, "fg must change after SGR 31");
    }

    /// Phase 57 G.4 acceptance: SGR 0 resets to defaults.
    #[test]
    fn sgr_reset_restores_defaults() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "\x1b[31m\x1b[0m");
        assert_eq!(s.colors(), (DEFAULT_FG, DEFAULT_BG));
    }

    /// Phase 57 G.4 acceptance: ED 2 (clear screen) emits a Clear
    /// command and the cursor is repositioned to the origin.
    #[test]
    fn ed_2_emits_clear() {
        let mut s = Screen::with_geometry(80, 25);
        let cmds = feed_str(&mut s, "ABC\x1b[2J");
        assert!(cmds.iter().any(|c| matches!(c, RenderCommand::Clear)));
    }

    /// Regression: ion redraws its prompt with `\r\x1b[J<new content>`
    /// on every keystroke. If we drop ED 0, characters past the new
    /// line's end stay visible — that produces the "backspace doesn't
    /// erase" and "shorter history line leaves trailing chars" bugs.
    #[test]
    fn ed_0_clears_from_cursor_to_end_of_display() {
        let mut s = Screen::with_geometry(5, 3);
        // Fill rows 0 and 1 with text, then position cursor at (0, 2).
        let _ = feed_str(&mut s, "ABCDE\nFGHIJ\r\x1b[A\x1b[3G");
        assert_eq!(s.cursor(), (0, 2));
        let _ = feed_str(&mut s, "\x1b[J");
        // Row 0: cells 0..2 unchanged, cells 2..5 blank.
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b'A' as u32);
        assert_eq!(s.cell(0, 1).unwrap().codepoint, b'B' as u32);
        for col in 2..5 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b' ' as u32);
        }
        // Row 1 fully blanked.
        for col in 0..5 {
            assert_eq!(s.cell(1, col).unwrap().codepoint, b' ' as u32);
        }
        // Cursor unchanged.
        assert_eq!(s.cursor(), (0, 2));
    }

    /// Direct regression for the ion redraw shape: the user types a
    /// long line, ion echoes it, then ion sends `\r\x1b[J<shorter>`.
    /// The trailing tail of the old line must be gone afterwards.
    #[test]
    fn ed_0_after_cr_clears_trailing_chars_from_prior_line() {
        let mut s = Screen::with_geometry(10, 2);
        let _ = feed_str(&mut s, "aaaaa");
        // Ion's redraw shape on backspace / history recall.
        let _ = feed_str(&mut s, "\r\x1b[Jbbb");
        for col in 0..3 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b'b' as u32);
        }
        for col in 3..5 {
            assert_eq!(
                s.cell(0, col).unwrap().codepoint,
                b' ' as u32,
                "trailing char at col {col} must be cleared"
            );
        }
    }

    /// ED 1 mirrors ED 0 from the other side: blank from the start of
    /// the display through the cursor (inclusive).
    #[test]
    fn ed_1_clears_from_start_to_cursor() {
        let mut s = Screen::with_geometry(5, 3);
        let _ = feed_str(&mut s, "ABCDE\nFGHIJ\r\x1b[3G");
        assert_eq!(s.cursor(), (1, 2));
        let _ = feed_str(&mut s, "\x1b[1J");
        // Row 0 fully blanked.
        for col in 0..5 {
            assert_eq!(s.cell(0, col).unwrap().codepoint, b' ' as u32);
        }
        // Row 1: cells 0..=2 blanked, cells 3..5 unchanged.
        for col in 0..=2 {
            assert_eq!(s.cell(1, col).unwrap().codepoint, b' ' as u32);
        }
        assert_eq!(s.cell(1, 3).unwrap().codepoint, b'I' as u32);
        assert_eq!(s.cell(1, 4).unwrap().codepoint, b'J' as u32);
    }

    /// Phase 57 G.4 acceptance: `out_of_bounds` cell access surfaces
    /// the typed `ScreenError::OutOfBounds`.
    #[test]
    fn out_of_bounds_cell_returns_error() {
        let s = Screen::with_geometry(4, 2);
        let err = s.cell(2, 0).expect_err("must error on out of bounds");
        assert_eq!(err, ScreenError::OutOfBounds);
    }

    /// Phase 69 Track B — `?1049h` activates the alternate screen
    /// without touching the primary buffer, and `?1049l` restores
    /// the primary content and saved cursor.
    #[test]
    fn alt_screen_preserves_primary_buffer() {
        let mut s = Screen::with_geometry(10, 4);
        let _ = feed_str(&mut s, "AB\n\x1b[31mC");
        assert_eq!(s.colors().0, 0xFFAA_0000); // red
        let (row, col) = s.cursor();
        // Enter alt screen.
        let _ = feed_str(&mut s, "\x1b[?1049h");
        assert_eq!(s.active(), ScreenSelect::Alt);
        // Write to alt — primary cells must be unchanged.
        let _ = feed_str(&mut s, "Z");
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b'Z' as u32);
        assert_eq!(s.cell_primary(0, 0).unwrap().codepoint, b'A' as u32);
        // Leave alt screen — primary cursor + colours restored.
        let _ = feed_str(&mut s, "\x1b[?1049l");
        assert_eq!(s.active(), ScreenSelect::Primary);
        assert_eq!(s.colors().0, 0xFFAA_0000);
        assert_eq!(s.cursor(), (row, col));
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b'A' as u32);
        assert_eq!(s.cell(0, 1).unwrap().codepoint, b'B' as u32);
    }

    /// Phase 69 Track B — nested `?1049h` while already on the
    /// alternate screen is a no-op (does not overwrite `SavedCursor`
    /// with the alt cursor).
    #[test]
    fn alt_screen_nested_entry_preserves_saved_cursor() {
        let mut s = Screen::with_geometry(10, 4);
        let _ = feed_str(&mut s, "XY");
        let _ = feed_str(&mut s, "\x1b[?1049h");
        let _ = feed_str(&mut s, "Q"); // alt cursor at (0,1)
        // Nested entry should be a no-op.
        let _ = feed_str(&mut s, "\x1b[?1049h");
        let _ = feed_str(&mut s, "\x1b[?1049l");
        // Cursor must restore to the primary's prior position (0,2).
        assert_eq!(s.cursor(), (0, 2));
    }

    /// Phase 69 Track B — `?1049l` while already on the primary is
    /// a no-op.
    #[test]
    fn alt_screen_exit_when_not_in_alt_is_noop() {
        let mut s = Screen::with_geometry(8, 2);
        let _ = feed_str(&mut s, "Hi");
        let (row, col) = s.cursor();
        let _ = feed_str(&mut s, "\x1b[?1049l");
        assert_eq!(s.cursor(), (row, col));
        assert_eq!(s.active(), ScreenSelect::Primary);
    }

    /// Phase 69 Track C — 256-color indexed SGR updates `fg`/`bg` via
    /// the xterm palette.
    #[test]
    fn sgr_indexed_color_updates_fg_to_palette_entry() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "\x1b[38;5;208m");
        let (fg, _) = s.colors();
        assert_eq!(fg, XTERM_256_PALETTE[208]);
    }

    /// Phase 69 Track C — truecolor RGB updates `fg`/`bg` to BGRA8888.
    #[test]
    fn sgr_truecolor_rgb_updates_fg() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "\x1b[38;2;10;20;30m");
        let (fg, _) = s.colors();
        assert_eq!(fg, 0xFF00_0000 | (10 << 16) | (20 << 8) | 30);
    }

    /// Phase 69 Track C — combined SGR `1;38;5;208;4m` parses
    /// correctly: the indexed-fg consumes exactly its parameters.
    #[test]
    fn sgr_mixed_codes_parse_correctly() {
        let mut s = Screen::with_geometry(80, 25);
        let _ = feed_str(&mut s, "\x1b[1;38;5;208;4m");
        let (fg, _) = s.colors();
        assert_eq!(fg, XTERM_256_PALETTE[208]);
    }

    /// Phase 69 Track F — DECSCUSR updates `cursor_shape`.
    #[test]
    fn decscusr_updates_cursor_shape() {
        let mut s = Screen::with_geometry(10, 4);
        assert_eq!(s.cursor_shape(), CursorShape::BlinkingBlock);
        let _ = feed_str(&mut s, "\x1b[2 q");
        assert_eq!(s.cursor_shape(), CursorShape::SteadyBlock);
        let _ = feed_str(&mut s, "\x1b[5 q");
        assert_eq!(s.cursor_shape(), CursorShape::BlinkingBar);
        // Out-of-range filtered at parser → no state change.
        let _ = feed_str(&mut s, "\x1b[7 q");
        assert_eq!(s.cursor_shape(), CursorShape::BlinkingBar);
    }

    /// Phase 69 Track G — `?2004h` toggles the bracketed-paste bit.
    #[test]
    fn bracketed_paste_mode_toggles() {
        let mut s = Screen::with_geometry(10, 4);
        assert!(!s.bracketed_paste_enabled());
        let _ = feed_str(&mut s, "\x1b[?2004h");
        assert!(s.bracketed_paste_enabled());
        let _ = feed_str(&mut s, "\x1b[?2004l");
        assert!(!s.bracketed_paste_enabled());
    }

    /// Phase 69 Track D — `Screen::resize` reallocates both grids,
    /// preserves cells in the overlap, and clamps the cursor.
    #[test]
    fn resize_clamps_cursor_and_preserves_overlap() {
        let mut s = Screen::with_geometry(8, 4);
        let _ = feed_str(&mut s, "abcdef");
        let mut out = Vec::new();
        s.resize(4, 2, &mut out);
        assert_eq!(s.cols(), 4);
        assert_eq!(s.rows(), 2);
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b'a' as u32);
        assert_eq!(s.cell(0, 3).unwrap().codepoint, b'd' as u32);
        let (r, c) = s.cursor();
        assert!(r < 2);
        assert!(c <= 4);
    }

    /// Phase 69b Track B.2 — a 3-byte UTF-8 sequence for U+2500
    /// lands as a single cell carrying the decoded codepoint.
    #[test]
    fn utf8_three_byte_box_drawing_decodes_to_single_cell() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // U+2500 ─ is E2 94 80.
        s.feed(0xE2, &mut out);
        // After the leading byte the decoder is `Pending` — no cell
        // has been written yet, cursor still at (0, 0).
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b' ' as u32);
        assert_eq!(s.cursor(), (0, 0));
        s.feed(0x94, &mut out);
        s.feed(0x80, &mut out);
        assert_eq!(s.cell(0, 0).unwrap().codepoint, 0x2500);
        assert_eq!(s.cursor().1, 1);
    }

    /// Phase 69b Track B.2 — a 2-byte UTF-8 sequence for U+00E9 (é)
    /// decodes into a single cell.
    #[test]
    fn utf8_two_byte_latin1_decodes_to_single_cell() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // U+00E9 é is C3 A9.
        s.feed(0xC3, &mut out);
        s.feed(0xA9, &mut out);
        assert_eq!(s.cell(0, 0).unwrap().codepoint, 0x00E9);
    }

    /// Phase 69b Track B.2 — a lone continuation byte yields exactly
    /// one U+FFFD cell.
    #[test]
    fn utf8_lone_continuation_yields_replacement_character() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // 0x80 is a continuation byte at the start of a sequence —
        // strictly invalid. The decoder emits Invalid, screen routes
        // it to U+FFFD.
        s.feed(0x80, &mut out);
        assert_eq!(s.cell(0, 0).unwrap().codepoint, REPLACEMENT_CHARACTER);
    }

    /// Phase 69b Track B.2 — BEL (0x07) routes through the UTF-8
    /// decoder and the `emit_codepoint` BEL guard. When the decoder
    /// is in the initial state, the result is identical to the
    /// Phase 57 contract: a single [`RenderCommand::Bell`] with no
    /// cell update.
    #[test]
    fn utf8_bel_byte_still_maps_to_bell_command() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        s.feed(0x07, &mut out);
        assert_eq!(out.as_slice(), &[RenderCommand::Bell]);
        // Cursor unchanged, no cell touched.
        assert_eq!(s.cursor(), (0, 0));
        assert_eq!(s.cell(0, 0).unwrap().codepoint, b' ' as u32);
    }

    /// Phase 69b Track B.2 — when a BEL byte arrives while the
    /// decoder is mid-sequence (e.g. one byte into a 2-byte UTF-8
    /// codepoint), the in-flight sequence must be cancelled and
    /// replaced by U+FFFD before the bell rings. The earlier
    /// implementation intercepted BEL before the decoder, leaving
    /// the pending state alive — so a following continuation byte
    /// would silently complete the broken sequence.
    #[test]
    fn utf8_bel_during_pending_cancels_in_flight_sequence() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // Start a 2-byte sequence (U+00E9 → C3 A9 form) then send BEL
        // before the continuation byte.
        s.feed(0xC2, &mut out);
        s.feed(0x07, &mut out);
        // First a PutGlyph for U+FFFD (the cancelled in-flight
        // sequence), then a Bell.
        assert!(
            matches!(out.first(), Some(RenderCommand::PutGlyph { codepoint, .. }) if *codepoint == REPLACEMENT_CHARACTER)
        );
        assert!(
            out.iter().any(|c| matches!(c, RenderCommand::Bell)),
            "BEL must be emitted after cancelling the pending sequence: {out:?}"
        );
        // Now feed a continuation byte that *would* have combined with
        // the stale 0xC2 if the decoder had not been reset. With the
        // fix, 0xA9 is a stray continuation → another U+FFFD, never
        // U+00A9.
        out.clear();
        s.feed(0xA9, &mut out);
        let last_glyph = out
            .iter()
            .filter_map(|c| match c {
                RenderCommand::PutGlyph { codepoint, .. } => Some(*codepoint),
                _ => None,
            })
            .last();
        assert_eq!(last_glyph, Some(REPLACEMENT_CHARACTER));
    }

    /// Phase 69b Track F — a width-2 codepoint (CJK U+4E2D = 中)
    /// occupies the leading cell plus the trailing cell as a
    /// wide-continuation. The renderer sees a `PutGlyph` for the
    /// leading cell *and* a blank `PutGlyph` for the trailing cell
    /// — the leader's draw covers only one 8×16 cell, so the trail
    /// needs its own command to overwrite any previous pixels there.
    #[test]
    fn utf8_wide_codepoint_reserves_two_cells() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // U+4E2D = E4 B8 AD.
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        // Leading cell.
        let lead = s.cell(0, 0).unwrap();
        assert_eq!(lead.codepoint, 0x4E2D);
        assert!(!lead.wide_continuation);
        // Trailing cell.
        let trail = s.cell(0, 1).unwrap();
        assert!(
            trail.wide_continuation,
            "trailing cell must be wide_continuation"
        );
        assert_eq!(trail.codepoint, 0);
        // Cursor advanced by 2.
        assert_eq!(s.cursor(), (0, 2));
        // Two PutGlyph commands: the leading cell with the wide
        // codepoint, and the trailing cell with a blank space so
        // previous pixels at the trail column are overwritten.
        let put_glyphs: Vec<_> = out
            .iter()
            .filter_map(|c| match c {
                RenderCommand::PutGlyph {
                    row,
                    col,
                    codepoint,
                    ..
                } => Some((*row, *col, *codepoint)),
                _ => None,
            })
            .collect();
        assert_eq!(put_glyphs.len(), 2);
        assert_eq!(put_glyphs[0], (0, 0, 0x4E2D));
        assert_eq!(put_glyphs[1], (0, 1, b' ' as u32));
    }

    /// Phase 69b Track F — overwriting the trailing cell of a wide
    /// glyph also blanks the leading half so the renderer drops the
    /// stale leading pixels.
    #[test]
    fn utf8_overwrite_trail_of_wide_cleans_lead() {
        let mut s = Screen::with_geometry(10, 2);
        let mut out = Vec::new();
        // Place 中 at (0, 0)/(0, 1).
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        // Move cursor back to (0, 1) — the trailing cell.
        let _ = feed_str(&mut s, "\x1b[1;2H");
        // Drain any out-of-band cursor moves.
        out.clear();
        // Now write 'X' at the trailing cell.
        s.feed(b'X', &mut out);
        // The leading cell (0, 0) must have been blanked.
        let lead = s.cell(0, 0).unwrap();
        assert_eq!(lead.codepoint, b' ' as u32);
        // The trailing cell now carries 'X'.
        let trail = s.cell(0, 1).unwrap();
        assert_eq!(trail.codepoint, b'X' as u32);
        assert!(!trail.wide_continuation);
    }

    /// Phase 69b Track F — when a new wide glyph's trailing cell falls
    /// on a column that currently holds a *leader* of another wide
    /// glyph, the displaced leader's own trail (one column further
    /// right) must be blanked so it does not remain as an orphan
    /// `wide_continuation`. Without that step, a subsequent overwrite
    /// at the orphan column would incorrectly blank an unrelated cell
    /// to its left.
    #[test]
    fn utf8_wide_glyph_displaces_existing_wide_leader_cleanly() {
        let mut s = Screen::with_geometry(6, 1);
        let mut out: Vec<RenderCommand> = Vec::new();
        // Place 'A' at (0, 0), then 中 at (0, 1)..=(0, 2).
        s.feed(b'A', &mut out);
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        // Move cursor back to (0, 0) via CR and overwrite columns
        // 0..=1 with a new wide 中.
        s.feed(b'\r', &mut out);
        out.clear();
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        // Column 0 — new leader.
        let new_lead = s.cell(0, 0).unwrap();
        assert_eq!(new_lead.codepoint, 0x4E2D);
        assert!(!new_lead.wide_continuation);
        // Column 1 — new trailing cell.
        let new_trail = s.cell(0, 1).unwrap();
        assert!(new_trail.wide_continuation);
        assert_eq!(new_trail.codepoint, 0);
        // Column 2 — the old wide glyph's trail. The leader it pointed
        // to (col 1) is now the new pair's trail, so column 2 must NOT
        // remain an orphan continuation.
        let displaced_trail = s.cell(0, 2).unwrap();
        assert!(
            !displaced_trail.wide_continuation,
            "column 2 must not be an orphan wide_continuation after a new wide leader is written at column 0"
        );
    }

    /// Phase 69b Track F — a wide glyph wrapping at the last column
    /// still emits a leader `PutGlyph` AND a blank trail `PutGlyph`
    /// on the destination row so the wrap target's previous pixels
    /// in both columns are overwritten.
    #[test]
    fn utf8_wide_codepoint_wrap_emits_trail_blank_on_destination_row() {
        let mut s = Screen::with_geometry(3, 2);
        let mut out = Vec::new();
        // Fill columns 0 and 1 on row 0 so the wide glyph cannot fit
        // there — it must wrap to row 1.
        s.feed(b'a', &mut out);
        s.feed(b'b', &mut out);
        out.clear();
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        let glyph_puts: Vec<_> = out
            .iter()
            .filter_map(|c| match c {
                RenderCommand::PutGlyph {
                    row,
                    col,
                    codepoint,
                    ..
                } => Some((*row, *col, *codepoint)),
                _ => None,
            })
            .collect();
        // The wrap path blanks the unused cell at (0, 2) before the
        // line feed; then the destination row gets the wide leader
        // at (1, 0) and the blank trail at (1, 1).
        assert!(
            glyph_puts.contains(&(1, 0, 0x4E2D)),
            "expected leader PutGlyph at (1,0): {glyph_puts:?}"
        );
        assert!(
            glyph_puts.contains(&(1, 1, b' ' as u32)),
            "expected blank trail PutGlyph at (1,1): {glyph_puts:?}"
        );
    }

    /// Phase 69b Track F — a width-2 glyph at the last column wraps
    /// to the next row so the wide cell pair stays adjacent.
    #[test]
    fn utf8_wide_codepoint_wraps_at_last_column() {
        let mut s = Screen::with_geometry(3, 2);
        let mut out = Vec::new();
        // Fill columns 0 and 1 with ASCII so the wide glyph cannot
        // fit at columns 1..=2 on row 0 — it must wrap.
        s.feed(b'a', &mut out);
        s.feed(b'b', &mut out);
        for &b in &[0xE4u8, 0xB8, 0xAD] {
            s.feed(b, &mut out);
        }
        // After the wide glyph: the glyph must land on row 1, cols
        // 0/1.
        let lead = s.cell(1, 0).unwrap();
        assert_eq!(lead.codepoint, 0x4E2D);
        let trail = s.cell(1, 1).unwrap();
        assert!(trail.wide_continuation);
    }

    /// Phase 69b Track B.2 — pure-ASCII escape sequences (every byte
    /// is < 0x80) still flow through the Phase 22b parser. Each byte
    /// completes a 1-byte UTF-8 codepoint and then routes to the
    /// parser, so the existing CSI vocabulary keeps working.
    #[test]
    fn utf8_ascii_escape_sequences_unaffected() {
        let mut s = Screen::with_geometry(10, 2);
        let _ = feed_str(&mut s, "\x1b[2J");
        // Cursor moved to origin by ED 2.
        assert_eq!(s.cursor(), (0, 0));
        // Colours stay at defaults.
        assert_eq!(s.colors(), (DEFAULT_FG, DEFAULT_BG));
    }

    /// Phase 57 G.4 property test: arbitrary ANSI byte sequences must
    /// not panic, must not produce out-of-bounds cursor positions, and
    /// must keep `scrollback_len` <= SCROLLBACK_LINES.  We use a
    /// hand-rolled fuzz loop because `proptest` is not a dev-dep of
    /// the term crate.
    #[test]
    fn property_arbitrary_bytes_dont_panic() {
        let mut s = Screen::with_geometry(8, 4);
        let mut out = Vec::new();
        // 4 KiB of pseudo-random bytes.
        let mut state = 0xCAFEBABEu32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let b = (state >> 16) as u8;
            s.feed(b, &mut out);
            // Cursor invariants.
            let (r, c) = s.cursor();
            assert!(r < s.rows());
            assert!(c <= s.cols());
            // Scrollback cap.
            assert!(s.scrollback_len() <= SCROLLBACK_LINES);
            // Drain command buffer between iterations so the property
            // does not OOM on a 4 KiB run.
            out.clear();
        }
    }
}
