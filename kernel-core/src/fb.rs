//! ANSI/VT100 escape sequence parser for the framebuffer console.
//!
//! This module lives in `kernel-core` so it can be unit-tested on the host
//! (`cargo test -p kernel-core`) without needing a real framebuffer or QEMU.
//!
//! The parser produces [`ConsoleCmd`] values that the kernel's `FbConsole`
//! executes against the real framebuffer.

/// Maximum number of CSI numeric parameters we track.
const MAX_PARAMS: usize = 8;

/// A command produced by the escape sequence parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleCmd {
    /// Print a visible character at the current cursor position.
    ///
    /// Phase 69b Track B.1 — payload widened from `char` to `u32` so
    /// callers feeding decoded UTF-8 codepoints (via
    /// [`crate::utf8::Utf8Decoder`]) can route any Unicode scalar
    /// straight into the parser without a lossy `char` round-trip.
    /// ASCII / Latin-1 call sites continue to compile by casting
    /// `byte as u32` or `c as u32`; existing renderers index their
    /// glyph tables on the same `u32` the cell buffer already stores.
    PutChar(u32),
    /// Carriage return — move cursor to column 0.
    CarriageReturn,
    /// Newline — advance to next line.
    Newline,
    /// Backspace — move cursor back one column, erase cell.
    Backspace,
    /// Tab — advance to next 8-column tab stop.
    Tab,
    /// Cursor Up by `n` rows.
    CursorUp(u16),
    /// Cursor Down by `n` rows.
    CursorDown(u16),
    /// Cursor Forward by `n` columns.
    CursorForward(u16),
    /// Cursor Back by `n` columns.
    CursorBack(u16),
    /// Cursor Horizontal Absolute — move to column `n` (1-based).
    CursorHorizontalAbsolute(u16),
    /// Cursor Position — move to (row, col), both 1-based.
    CursorPosition(u16, u16),
    /// Erase in Line: 0 = cursor to end, 1 = start to cursor, 2 = entire line.
    EraseLine(u16),
    /// Erase in Display: 0 = cursor to end, 1 = start to cursor, 2 = entire screen.
    EraseDisplay(u16),
    /// Phase 69 Track B — DEC private mode set/reset (`CSI ? <code> h`
    /// or `CSI ? <code> l`). Covers `?25` (DECTCEM cursor visibility),
    /// `?1049` / `?47` (alternate-screen buffer), `?9` / `?1000` /
    /// `?1006` (mouse reporting), and `?2004` (bracketed paste). Codes
    /// the consumer does not recognise are dropped silently.
    ///
    /// A single CSI may carry multiple semicolon-separated codes
    /// (e.g. `CSI ? 1006 ; 1000 h` from the terminfo `XM` capability);
    /// every parsed code is preserved in `codes[..count]` and consumers
    /// must iterate that slice so multi-mode toggles like the xterm
    /// `?1000 + ?1006` mouse-tracking idiom land as a single batched
    /// apply. Single-code call sites can construct payloads tersely via
    /// [`ConsoleCmd::dec_private_single`].
    DecPrivateMode {
        codes: [u16; MAX_PARAMS],
        count: usize,
        set: bool,
    },
    /// Phase 69 Track F — DECSCUSR cursor shape (`CSI <n> SP q`). `shape`
    /// is `0..=6` per the DEC vocabulary: 0/1 blinking block, 2 steady
    /// block, 3 blinking underline, 4 steady underline, 5 blinking bar,
    /// 6 steady bar. Out-of-range codes are filtered by the parser and
    /// do not produce this variant.
    CursorShape { shape: u16 },
    /// SGR — Set Graphic Rendition. Parameters stored as a slice reference
    /// isn't possible in a Copy enum, so we use a small inline array.
    Sgr(SgrParams),
    /// DECSTBM — Set Top and Bottom Margins (`CSI <top> ; <bot> r`).
    /// Both bounds are 1-based; when both are missing the consumer
    /// should reset to the full screen. Less and other TUI apps use
    /// this to limit subsequent scrolling to a sub-rectangle.
    SetScrollRegion { top: u16, bottom: u16 },
    /// VPA — Vertical Position Absolute (`CSI <n> d`). Moves the
    /// cursor to row `n` (1-based) without changing the column.  Less
    /// emits `vpa` heavily during incremental line repaints.
    VerticalPositionAbsolute(u16),
    /// IL — Insert N Lines (`CSI <n> L`). Inserts N blank lines at the
    /// cursor, scrolling content within the current scroll region
    /// downward.
    InsertLines(u16),
    /// DL — Delete N Lines (`CSI <n> M`). Deletes N lines starting at
    /// the cursor, scrolling content within the current scroll region
    /// upward.
    DeleteLines(u16),
    /// ICH — Insert N Characters (`CSI <n> @`). Inserts N blank cells
    /// at the cursor, shifting the rest of the row right.
    InsertChars(u16),
    /// DCH — Delete N Characters (`CSI <n> P`). Deletes N cells at the
    /// cursor, shifting the rest of the row left.
    DeleteChars(u16),
    /// ECH — Erase N Characters (`CSI <n> X`). Blanks N cells starting
    /// at the cursor without moving the cursor.
    EraseChars(u16),
    /// SU — Scroll Up by N (`CSI <n> S`). Shifts content up within the
    /// scroll region by N lines; top N lines are lost, bottom N
    /// blanked.
    ScrollUp(u16),
    /// SD — Scroll Down by N (`CSI <n> T`). Shifts content down within
    /// the scroll region by N lines; bottom N lines are lost, top N
    /// blanked.
    ScrollDown(u16),
    /// Unknown/unsupported sequence — silently ignored.
    Nop,
}

impl ConsoleCmd {
    /// Phase 69 Track B — build a single-code `DecPrivateMode` payload.
    /// Used by tests and by call sites that synthesize a private-mode
    /// command outside the parser.
    pub const fn dec_private_single(code: u16, set: bool) -> Self {
        let mut codes = [0u16; MAX_PARAMS];
        codes[0] = code;
        ConsoleCmd::DecPrivateMode {
            codes,
            count: 1,
            set,
        }
    }
}

/// Inline storage for SGR parameters (up to MAX_PARAMS values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SgrParams {
    pub params: [u16; MAX_PARAMS],
    pub count: usize,
}

/// Phase 69 Track C — typed SGR operation produced by
/// [`SgrParams::ops`]. Lets consumers walk a CSI `m` sequence
/// without re-implementing the 38/48 extended-color sub-grammar in
/// every renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgrOp {
    /// `SGR 0` — reset all attributes to default.
    Reset,
    /// `SGR 1` — bold.
    Bold,
    /// `SGR 4` — underline.
    Underline,
    /// `SGR 7` — reverse video.
    Reverse,
    /// `SGR 22` — clear bold.
    NoBold,
    /// `SGR 24` — clear underline.
    NoUnderline,
    /// `SGR 27` — clear reverse.
    NoReverse,
    /// `SGR 30..=37` — standard 8-color foreground (index 0..=7).
    Fg8(u8),
    /// `SGR 40..=47` — standard 8-color background (index 0..=7).
    Bg8(u8),
    /// `SGR 39` — restore default foreground.
    FgDefault,
    /// `SGR 49` — restore default background.
    BgDefault,
    /// `SGR 90..=97` — bright 8-color foreground (index 0..=7, palette
    /// indices 8..=15).
    FgBright8(u8),
    /// `SGR 100..=107` — bright 8-color background (index 0..=7,
    /// palette indices 8..=15).
    BgBright8(u8),
    /// `SGR 38;5;<n>` — 256-color indexed foreground.
    FgIndexed(u8),
    /// `SGR 48;5;<n>` — 256-color indexed background.
    BgIndexed(u8),
    /// `SGR 38;2;<r>;<g>;<b>` — 24-bit RGB foreground.
    FgRgb(u8, u8, u8),
    /// `SGR 48;2;<r>;<g>;<b>` — 24-bit RGB background.
    BgRgb(u8, u8, u8),
}

impl SgrParams {
    /// Phase 69 Track C — walk the raw SGR params and yield typed
    /// [`SgrOp`] values. Unrecognised codes are skipped silently.
    /// Truncated extended-color sequences (e.g. `38;5` with no
    /// trailing index) are also dropped silently so a malformed SGR
    /// cannot derail the rest of the sequence.
    pub fn ops(&self) -> SgrOpIter<'_> {
        SgrOpIter {
            params: &self.params[..self.count.min(self.params.len())],
            idx: 0,
        }
    }
}

/// Iterator returned by [`SgrParams::ops`].
#[derive(Debug, Clone)]
pub struct SgrOpIter<'a> {
    params: &'a [u16],
    idx: usize,
}

impl<'a> Iterator for SgrOpIter<'a> {
    type Item = SgrOp;

    fn next(&mut self) -> Option<SgrOp> {
        while self.idx < self.params.len() {
            let code = self.params[self.idx];
            self.idx += 1;
            match code {
                0 => return Some(SgrOp::Reset),
                1 => return Some(SgrOp::Bold),
                4 => return Some(SgrOp::Underline),
                7 => return Some(SgrOp::Reverse),
                22 => return Some(SgrOp::NoBold),
                24 => return Some(SgrOp::NoUnderline),
                27 => return Some(SgrOp::NoReverse),
                30..=37 => return Some(SgrOp::Fg8((code - 30) as u8)),
                39 => return Some(SgrOp::FgDefault),
                40..=47 => return Some(SgrOp::Bg8((code - 40) as u8)),
                49 => return Some(SgrOp::BgDefault),
                90..=97 => return Some(SgrOp::FgBright8((code - 90) as u8)),
                100..=107 => return Some(SgrOp::BgBright8((code - 100) as u8)),
                38 | 48 => {
                    let is_fg = code == 38;
                    let sub = match self.params.get(self.idx) {
                        Some(&s) => s,
                        None => return None,
                    };
                    self.idx += 1;
                    match sub {
                        5 => {
                            let n = match self.params.get(self.idx) {
                                Some(&v) => v,
                                None => return None,
                            };
                            self.idx += 1;
                            let n = (n & 0xff) as u8;
                            return Some(if is_fg {
                                SgrOp::FgIndexed(n)
                            } else {
                                SgrOp::BgIndexed(n)
                            });
                        }
                        2 => {
                            let r = self.params.get(self.idx).copied();
                            let g = self.params.get(self.idx + 1).copied();
                            let b = self.params.get(self.idx + 2).copied();
                            match (r, g, b) {
                                (Some(r), Some(g), Some(b)) => {
                                    self.idx += 3;
                                    let r = (r & 0xff) as u8;
                                    let g = (g & 0xff) as u8;
                                    let b = (b & 0xff) as u8;
                                    return Some(if is_fg {
                                        SgrOp::FgRgb(r, g, b)
                                    } else {
                                        SgrOp::BgRgb(r, g, b)
                                    });
                                }
                                _ => return None,
                            }
                        }
                        // Unknown sub-spec (e.g. 38;3 for CMY or 38;4 for
                        // CMYK) — skip silently and keep walking.
                        _ => continue,
                    }
                }
                // Codes we don't model yet (italic, blink, faint, …)
                // are skipped. They are not part of Phase 69's scope.
                _ => continue,
            }
        }
        None
    }
}

/// Parser state for the escape sequence state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscState {
    /// Normal text mode — characters are printed directly.
    Normal,
    /// Saw ESC (0x1B) — waiting for `[` or discarding.
    Escape,
    /// Inside a CSI sequence (ESC [) — accumulating parameters.
    Csi,
    /// Inside a CSI private sequence (ESC [ ?) — accumulating parameters.
    CsiPrivate,
    /// Phase 69 Track F — saw an intermediate space byte inside a CSI
    /// sequence (`ESC [ <n> SP ...`). DEC's DECSCUSR uses this form:
    /// `ESC [ <n> SP q`. We track the intermediate so the final byte
    /// `q` dispatches to a cursor-shape command rather than the
    /// existing repertoire.
    CsiIntermediate,
}

/// ANSI escape sequence parser.
///
/// Feed characters one at a time via [`process_char`]. Each call returns a
/// [`ConsoleCmd`] that the framebuffer console should execute.
#[derive(Debug, Clone)]
pub struct AnsiParser {
    state: EscState,
    params: [u16; MAX_PARAMS],
    param_count: usize,
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AnsiParser {
    /// Create a new parser in the Normal state.
    pub const fn new() -> Self {
        AnsiParser {
            state: EscState::Normal,
            params: [0; MAX_PARAMS],
            param_count: 0,
        }
    }

    /// Reset the parser to initial state.
    fn reset(&mut self) {
        self.state = EscState::Normal;
        self.params = [0; MAX_PARAMS];
        self.param_count = 0;
    }

    /// Reset CSI parameter accumulation.
    fn reset_params(&mut self) {
        self.params = [0; MAX_PARAMS];
        self.param_count = 0;
    }

    /// Get parameter at index with a default value if not provided.
    fn param(&self, idx: usize, default: u16) -> u16 {
        if idx < self.param_count {
            let v = self.params[idx];
            if v == 0 { default } else { v }
        } else {
            default
        }
    }

    /// Process a single character through the state machine.
    /// Returns the command the console should execute.
    pub fn process_char(&mut self, c: char) -> ConsoleCmd {
        match self.state {
            EscState::Normal => self.process_normal(c),
            EscState::Escape => self.process_escape(c),
            EscState::Csi => self.process_csi(c),
            EscState::CsiPrivate => self.process_csi_private(c),
            EscState::CsiIntermediate => self.process_csi_intermediate(c),
        }
    }

    fn process_normal(&mut self, c: char) -> ConsoleCmd {
        match c {
            '\x1b' => {
                self.state = EscState::Escape;
                ConsoleCmd::Nop
            }
            '\r' => ConsoleCmd::CarriageReturn,
            '\n' => ConsoleCmd::Newline,
            '\x08' => ConsoleCmd::Backspace,
            '\t' => ConsoleCmd::Tab,
            _ => ConsoleCmd::PutChar(c as u32),
        }
    }

    fn process_escape(&mut self, c: char) -> ConsoleCmd {
        match c {
            '[' => {
                self.state = EscState::Csi;
                self.reset_params();
                ConsoleCmd::Nop
            }
            _ => {
                // Unknown escape sequence — discard and return to normal.
                self.state = EscState::Normal;
                ConsoleCmd::Nop
            }
        }
    }

    fn process_csi(&mut self, c: char) -> ConsoleCmd {
        match c {
            '0'..='9' => {
                // Accumulate digit into current parameter.
                if self.param_count == 0 {
                    self.param_count = 1;
                }
                let idx = self.param_count - 1;
                if idx < MAX_PARAMS {
                    self.params[idx] = self.params[idx]
                        .saturating_mul(10)
                        .saturating_add(c as u16 - b'0' as u16);
                }
                ConsoleCmd::Nop
            }
            ';' => {
                // Advance to next parameter.
                if self.param_count == 0 {
                    self.param_count = 1; // First param was implicitly 0.
                }
                if self.param_count < MAX_PARAMS {
                    self.param_count += 1;
                }
                ConsoleCmd::Nop
            }
            '?' => {
                self.state = EscState::CsiPrivate;
                ConsoleCmd::Nop
            }
            // Phase 69 Track F — intermediate space byte: `ESC [ <n> SP <final>`.
            // DEC's DECSCUSR `ESC [ <n> SP q` cursor-shape sequence is the
            // only consumer today; future DEC private intermediates can
            // dispatch from `process_csi_intermediate`.
            ' ' => {
                self.state = EscState::CsiIntermediate;
                ConsoleCmd::Nop
            }
            // Final byte (0x40–0x7E) — dispatch the CSI sequence.
            c if (c as u32) >= 0x40 && (c as u32) <= 0x7E => {
                let cmd = self.dispatch_csi(c);
                self.state = EscState::Normal;
                cmd
            }
            _ => {
                // Malformed sequence — discard and return to normal.
                self.reset();
                ConsoleCmd::Nop
            }
        }
    }

    fn process_csi_intermediate(&mut self, c: char) -> ConsoleCmd {
        // We've already seen `ESC [ <params> SP`. The only final byte
        // we recognise is 'q' for DECSCUSR; everything else is dropped.
        let cmd = match c {
            'q' => {
                let shape = self.param(0, 0);
                if shape <= 6 {
                    ConsoleCmd::CursorShape { shape }
                } else {
                    ConsoleCmd::Nop
                }
            }
            _ => ConsoleCmd::Nop,
        };
        // Whether final byte was 'q' or something we don't recognise,
        // exit back to Normal — a malformed sequence still terminates
        // on the final byte. Only digits could extend the params, but
        // intermediates separate params from final bytes per ECMA-48.
        self.state = EscState::Normal;
        cmd
    }

    fn process_csi_private(&mut self, c: char) -> ConsoleCmd {
        match c {
            '0'..='9' => {
                if self.param_count == 0 {
                    self.param_count = 1;
                }
                let idx = self.param_count - 1;
                if idx < MAX_PARAMS {
                    self.params[idx] = self.params[idx]
                        .saturating_mul(10)
                        .saturating_add(c as u16 - b'0' as u16);
                }
                ConsoleCmd::Nop
            }
            ';' => {
                if self.param_count == 0 {
                    self.param_count = 1;
                }
                if self.param_count < MAX_PARAMS {
                    self.param_count += 1;
                }
                ConsoleCmd::Nop
            }
            c if (c as u32) >= 0x40 && (c as u32) <= 0x7E => {
                let cmd = self.dispatch_csi_private(c);
                self.state = EscState::Normal;
                cmd
            }
            _ => {
                self.reset();
                ConsoleCmd::Nop
            }
        }
    }

    fn dispatch_csi(&self, final_byte: char) -> ConsoleCmd {
        match final_byte {
            // CUU — Cursor Up
            'A' => ConsoleCmd::CursorUp(self.param(0, 1)),
            // CUD — Cursor Down
            'B' => ConsoleCmd::CursorDown(self.param(0, 1)),
            // CUF — Cursor Forward
            'C' => ConsoleCmd::CursorForward(self.param(0, 1)),
            // CUB — Cursor Back
            'D' => ConsoleCmd::CursorBack(self.param(0, 1)),
            // CHA — Cursor Horizontal Absolute
            'G' => ConsoleCmd::CursorHorizontalAbsolute(self.param(0, 1)),
            // CUP — Cursor Position
            'H' => ConsoleCmd::CursorPosition(self.param(0, 1), self.param(1, 1)),
            // ED — Erase in Display
            'J' => ConsoleCmd::EraseDisplay(self.param(0, 0)),
            // EL — Erase in Line
            'K' => ConsoleCmd::EraseLine(self.param(0, 0)),
            // VPA — Vertical Position Absolute (`CSI <n> d`).
            'd' => ConsoleCmd::VerticalPositionAbsolute(self.param(0, 1)),
            // DECSTBM — Set Top and Bottom Margins (`CSI <top> ; <bot> r`).
            // No params resets to full screen, encoded as (0, 0); the
            // screen state machine interprets the zero bounds as
            // "reset" rather than literal row zero.
            'r' => ConsoleCmd::SetScrollRegion {
                top: self.param(0, 0),
                bottom: self.param(1, 0),
            },
            // SU / SD — Scroll Up / Down within the scroll region.
            'S' => ConsoleCmd::ScrollUp(self.param(0, 1)),
            'T' => ConsoleCmd::ScrollDown(self.param(0, 1)),
            // IL / DL — Insert / Delete Lines at cursor.
            'L' => ConsoleCmd::InsertLines(self.param(0, 1)),
            'M' => ConsoleCmd::DeleteLines(self.param(0, 1)),
            // ICH / DCH / ECH — Insert / Delete / Erase Characters at cursor.
            '@' => ConsoleCmd::InsertChars(self.param(0, 1)),
            'P' => ConsoleCmd::DeleteChars(self.param(0, 1)),
            'X' => ConsoleCmd::EraseChars(self.param(0, 1)),
            // SGR — Select Graphic Rendition
            'm' => {
                let count = if self.param_count == 0 {
                    1
                } else {
                    self.param_count
                };
                let mut sgr = SgrParams {
                    params: [0; MAX_PARAMS],
                    count,
                };
                // If no params were given, treat as SGR 0 (reset).
                if self.param_count == 0 {
                    sgr.params[0] = 0;
                } else {
                    sgr.params[..self.param_count]
                        .copy_from_slice(&self.params[..self.param_count]);
                }
                ConsoleCmd::Sgr(sgr)
            }
            _ => ConsoleCmd::Nop,
        }
    }

    fn dispatch_csi_private(&self, final_byte: char) -> ConsoleCmd {
        // Phase 69 Track B — every `ESC [ ? <p1> [; <p2> ...] h/l`
        // dispatches to a single `DecPrivateMode` that carries all
        // semicolon-separated codes. Code 25 (DECTCEM) used to map to
        // a dedicated `SetCursorVisible`; that case is now folded into
        // `DecPrivateMode { codes, .. }` so the alternate-screen, mouse,
        // and bracketed-paste codes share a single uniform arm.
        //
        // Multi-code support matters for terminfo capabilities like
        // `XM=\E[?1006;1000h/l` which toggle SGR mouse encoding and
        // button-event tracking in one CSI; consumers iterate
        // `codes[..count]` to keep both halves in sync. Consumers that
        // do not recognise a code drop that code silently and continue
        // processing the rest.
        let set = match final_byte {
            'h' => true,
            'l' => false,
            _ => return ConsoleCmd::Nop,
        };
        let mut codes = [0u16; MAX_PARAMS];
        // `param_count` is 0 when the parser saw `\x1b[?h` with no
        // numeric body; treat that as a single zero so consumers get a
        // recognisable (and silently dropped) `code = 0` payload rather
        // than a zero-length slice.
        let count = if self.param_count == 0 {
            codes[0] = 0;
            1
        } else {
            let n = self.param_count.min(MAX_PARAMS);
            codes[..n].copy_from_slice(&self.params[..n]);
            n
        };
        ConsoleCmd::DecPrivateMode { codes, count, set }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_str(s: &str) -> alloc::vec::Vec<ConsoleCmd> {
        let mut parser = AnsiParser::new();
        s.chars().map(|c| parser.process_char(c)).collect()
    }

    fn parse_str_last(s: &str) -> ConsoleCmd {
        let mut parser = AnsiParser::new();
        let mut last = ConsoleCmd::Nop;
        for c in s.chars() {
            last = parser.process_char(c);
        }
        last
    }

    #[test]
    fn test_printable_chars() {
        let cmds = parse_str("AB");
        assert_eq!(
            cmds,
            &[
                ConsoleCmd::PutChar(b'A' as u32),
                ConsoleCmd::PutChar(b'B' as u32),
            ]
        );
    }

    #[test]
    fn test_control_chars() {
        assert_eq!(parse_str_last("\r"), ConsoleCmd::CarriageReturn);
        assert_eq!(parse_str_last("\n"), ConsoleCmd::Newline);
        assert_eq!(parse_str_last("\x08"), ConsoleCmd::Backspace);
        assert_eq!(parse_str_last("\t"), ConsoleCmd::Tab);
    }

    #[test]
    fn test_erase_display_2j() {
        let cmd = parse_str_last("\x1b[2J");
        assert_eq!(cmd, ConsoleCmd::EraseDisplay(2));
    }

    #[test]
    fn test_cursor_position() {
        let cmd = parse_str_last("\x1b[10;20H");
        assert_eq!(cmd, ConsoleCmd::CursorPosition(10, 20));
    }

    #[test]
    fn test_cursor_position_default() {
        let cmd = parse_str_last("\x1b[H");
        assert_eq!(cmd, ConsoleCmd::CursorPosition(1, 1));
    }

    #[test]
    fn test_dectcem_hide() {
        let cmd = parse_str_last("\x1b[?25l");
        assert_eq!(cmd, ConsoleCmd::dec_private_single(25, false));
    }

    #[test]
    fn test_dectcem_show() {
        let cmd = parse_str_last("\x1b[?25h");
        assert_eq!(cmd, ConsoleCmd::dec_private_single(25, true));
    }

    #[test]
    fn test_sgr_reset() {
        let cmd = parse_str_last("\x1b[m");
        if let ConsoleCmd::Sgr(sgr) = cmd {
            assert_eq!(sgr.count, 1);
            assert_eq!(sgr.params[0], 0);
        } else {
            panic!("Expected Sgr, got {:?}", cmd);
        }
    }

    #[test]
    fn test_sgr_explicit_reset() {
        let cmd = parse_str_last("\x1b[0m");
        if let ConsoleCmd::Sgr(sgr) = cmd {
            assert_eq!(sgr.count, 1);
            assert_eq!(sgr.params[0], 0);
        } else {
            panic!("Expected Sgr, got {:?}", cmd);
        }
    }

    #[test]
    fn test_sgr_color() {
        let cmd = parse_str_last("\x1b[31;42m");
        if let ConsoleCmd::Sgr(sgr) = cmd {
            assert_eq!(sgr.count, 2);
            assert_eq!(sgr.params[0], 31);
            assert_eq!(sgr.params[1], 42);
        } else {
            panic!("Expected Sgr, got {:?}", cmd);
        }
    }

    #[test]
    fn test_malformed_escape_recovery() {
        // ESC followed by a non-[ character should discard and return to normal.
        let cmds = parse_str("\x1bXA");
        assert_eq!(
            cmds,
            &[
                ConsoleCmd::Nop, // ESC
                ConsoleCmd::Nop, // 'X' — discarded, back to Normal
                ConsoleCmd::PutChar(b'A' as u32),
            ]
        );
    }

    #[test]
    fn test_cursor_movement() {
        assert_eq!(parse_str_last("\x1b[5A"), ConsoleCmd::CursorUp(5));
        assert_eq!(parse_str_last("\x1b[3B"), ConsoleCmd::CursorDown(3));
        assert_eq!(parse_str_last("\x1b[C"), ConsoleCmd::CursorForward(1));
        assert_eq!(parse_str_last("\x1b[2D"), ConsoleCmd::CursorBack(2));
        assert_eq!(
            parse_str_last("\x1b[15G"),
            ConsoleCmd::CursorHorizontalAbsolute(15)
        );
    }

    #[test]
    fn test_erase_in_line() {
        assert_eq!(parse_str_last("\x1b[K"), ConsoleCmd::EraseLine(0));
        assert_eq!(parse_str_last("\x1b[0K"), ConsoleCmd::EraseLine(0));
        assert_eq!(parse_str_last("\x1b[1K"), ConsoleCmd::EraseLine(1));
        assert_eq!(parse_str_last("\x1b[2K"), ConsoleCmd::EraseLine(2));
    }

    #[test]
    fn test_erase_in_display() {
        assert_eq!(parse_str_last("\x1b[J"), ConsoleCmd::EraseDisplay(0));
        assert_eq!(parse_str_last("\x1b[0J"), ConsoleCmd::EraseDisplay(0));
        assert_eq!(parse_str_last("\x1b[2J"), ConsoleCmd::EraseDisplay(2));
    }

    #[test]
    fn test_interleaved_text_and_escapes() {
        let cmds = parse_str("A\x1b[2JB");
        assert_eq!(
            cmds,
            &[
                ConsoleCmd::PutChar(b'A' as u32),
                ConsoleCmd::Nop,             // ESC
                ConsoleCmd::Nop,             // [
                ConsoleCmd::Nop,             // 2
                ConsoleCmd::EraseDisplay(2), // J
                ConsoleCmd::PutChar(b'B' as u32),
            ]
        );
    }

    #[test]
    fn test_unknown_csi_sequence() {
        // Unknown final byte 'z' — should produce Nop.
        assert_eq!(parse_str_last("\x1b[5z"), ConsoleCmd::Nop);
    }

    #[test]
    fn test_state_after_sequence() {
        // After a complete sequence, parser should be back in Normal.
        let mut parser = AnsiParser::new();
        for c in "\x1b[2J".chars() {
            parser.process_char(c);
        }
        assert_eq!(parser.state, EscState::Normal);
    }

    #[test]
    fn test_sgr_bright_fg_colors() {
        // Bright foreground colors: 90–97
        for code in 90..=97u16 {
            let seq = alloc::format!("\x1b[{}m", code);
            let cmd = parse_str_last(&seq);
            if let ConsoleCmd::Sgr(sgr) = cmd {
                assert_eq!(sgr.count, 1);
                assert_eq!(sgr.params[0], code);
            } else {
                panic!("Expected Sgr for code {}, got {:?}", code, cmd);
            }
        }
    }

    #[test]
    fn test_sgr_standard_fg_colors() {
        // Standard foreground colors: 30–37
        for code in 30..=37u16 {
            let seq = alloc::format!("\x1b[{}m", code);
            let cmd = parse_str_last(&seq);
            if let ConsoleCmd::Sgr(sgr) = cmd {
                assert_eq!(sgr.count, 1);
                assert_eq!(sgr.params[0], code);
            } else {
                panic!("Expected Sgr for code {}, got {:?}", code, cmd);
            }
        }
    }

    #[test]
    fn test_sgr_standard_bg_colors() {
        // Standard background colors: 40–47
        for code in 40..=47u16 {
            let seq = alloc::format!("\x1b[{}m", code);
            let cmd = parse_str_last(&seq);
            if let ConsoleCmd::Sgr(sgr) = cmd {
                assert_eq!(sgr.count, 1);
                assert_eq!(sgr.params[0], code);
            } else {
                panic!("Expected Sgr for code {}, got {:?}", code, cmd);
            }
        }
    }

    #[test]
    fn test_partial_escape_sequence() {
        // ESC alone produces Nop, parser should be in Escape state.
        let mut parser = AnsiParser::new();
        assert_eq!(parser.process_char('\x1b'), ConsoleCmd::Nop);
        assert_eq!(parser.state, EscState::Escape);
    }

    #[test]
    fn test_incomplete_csi_sequence() {
        // ESC [ with digits but no final byte — parser stays in Csi state.
        let mut parser = AnsiParser::new();
        assert_eq!(parser.process_char('\x1b'), ConsoleCmd::Nop);
        assert_eq!(parser.process_char('['), ConsoleCmd::Nop);
        assert_eq!(parser.process_char('3'), ConsoleCmd::Nop);
        assert_eq!(parser.state, EscState::Csi);
        // A normal character after an incomplete CSI is not a valid final byte
        // if it's outside 0x40-0x7E range; but 'A' (0x41) IS a final byte.
        // Let's verify the digit accumulation by completing with a final byte.
        assert_eq!(parser.process_char('A'), ConsoleCmd::CursorUp(3));
        assert_eq!(parser.state, EscState::Normal);
    }

    #[test]
    fn test_incomplete_csi_then_normal_text() {
        // Start CSI, then feed a character outside valid CSI range (< 0x20)
        // to trigger malformed discard, then normal text.
        let mut parser = AnsiParser::new();
        parser.process_char('\x1b');
        parser.process_char('[');
        // Feed a control char that's not a digit, semicolon, or final byte
        let cmd = parser.process_char('\x01');
        assert_eq!(cmd, ConsoleCmd::Nop); // malformed, discarded
        assert_eq!(parser.state, EscState::Normal);
        // Normal text works again
        assert_eq!(parser.process_char('X'), ConsoleCmd::PutChar(b'X' as u32));
    }

    #[test]
    fn test_unknown_escape_after_esc() {
        // ESC followed by something other than '[' discards and returns Nop.
        let mut parser = AnsiParser::new();
        parser.process_char('\x1b');
        let cmd = parser.process_char('O'); // e.g. SS3 — not supported
        assert_eq!(cmd, ConsoleCmd::Nop);
        assert_eq!(parser.state, EscState::Normal);
    }

    #[test]
    fn test_csi_private_unknown_param() {
        // Phase 69 Track B — non-25 codes still produce DecPrivateMode;
        // unrecognised codes are dropped at the consumer level.
        assert_eq!(
            parse_str_last("\x1b[?1h"),
            ConsoleCmd::dec_private_single(1, true)
        );
    }

    /// Phase 69 Track B — DECSET / DECRST for alternate-screen
    /// (`?1049`, `?47`), mouse reporting (`?9`, `?1000`, `?1006`),
    /// and bracketed paste (`?2004`).
    #[test]
    fn test_dec_private_mode_codes() {
        let cases: [(&str, u16, bool); 10] = [
            ("\x1b[?1049h", 1049, true),
            ("\x1b[?1049l", 1049, false),
            ("\x1b[?47h", 47, true),
            ("\x1b[?47l", 47, false),
            ("\x1b[?9h", 9, true),
            ("\x1b[?1000h", 1000, true),
            ("\x1b[?1006h", 1006, true),
            ("\x1b[?2004h", 2004, true),
            ("\x1b[?2004l", 2004, false),
            // Bogus / unrecognised code still parses successfully; the
            // consumer drops it silently.
            ("\x1b[?9999h", 9999, true),
        ];
        for (input, code, set) in cases {
            assert_eq!(
                parse_str_last(input),
                ConsoleCmd::dec_private_single(code, set),
                "unexpected parse for {:?}",
                input.as_bytes()
            );
        }
    }

    /// Phase 69 Track B — multi-parameter DEC private modes
    /// (e.g. terminfo `XM=\E[?1006;1000h`). Every code lands in
    /// `codes[..count]` so consumers can apply each one in order.
    #[test]
    fn test_dec_private_multi_param() {
        let cmd = parse_str_last("\x1b[?1006;1000h");
        match cmd {
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                assert!(set, "expected set=true for h-terminator");
                assert_eq!(count, 2);
                assert_eq!(&codes[..count], &[1006u16, 1000u16]);
            }
            other => panic!("expected DecPrivateMode, got {:?}", other),
        }

        // Same shape on reset (`l`).
        let cmd = parse_str_last("\x1b[?1006;1000l");
        match cmd {
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                assert!(!set);
                assert_eq!(count, 2);
                assert_eq!(&codes[..count], &[1006u16, 1000u16]);
            }
            other => panic!("expected DecPrivateMode, got {:?}", other),
        }

        // Three-param form — e.g. enable mouse + SGR + paste in one go.
        let cmd = parse_str_last("\x1b[?1000;1006;2004h");
        match cmd {
            ConsoleCmd::DecPrivateMode { codes, count, set } => {
                assert!(set);
                assert_eq!(count, 3);
                assert_eq!(&codes[..count], &[1000u16, 1006u16, 2004u16]);
            }
            other => panic!("expected DecPrivateMode, got {:?}", other),
        }
    }

    /// Phase 69 Track C — 256-color indexed SGR.
    #[test]
    fn test_sgr_256_indexed_fg_and_bg() {
        let cmd = parse_str_last("\x1b[38;5;208m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr, got {:?}", cmd)
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::FgIndexed(208)]);

        let cmd = parse_str_last("\x1b[48;5;0m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::BgIndexed(0)]);

        let cmd = parse_str_last("\x1b[38;5;255m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::FgIndexed(255)]);
    }

    /// Phase 69 Track C — 24-bit RGB SGR.
    #[test]
    fn test_sgr_truecolor_rgb() {
        let cmd = parse_str_last("\x1b[38;2;1;2;3m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::FgRgb(1, 2, 3)]);

        let cmd = parse_str_last("\x1b[48;2;0;0;255m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::BgRgb(0, 0, 255)]);
    }

    /// Phase 69 Track C — mixed SGR sequence combines bold + indexed-fg.
    #[test]
    fn test_sgr_mixed_bold_and_indexed_fg() {
        let cmd = parse_str_last("\x1b[1;38;5;208m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::Bold, SgrOp::FgIndexed(208)]);
    }

    /// Phase 69 Track C — bright 8-color foreground / background.
    #[test]
    fn test_sgr_bright_8color() {
        let cmd = parse_str_last("\x1b[93m");
        let sgr = if let ConsoleCmd::Sgr(s) = cmd {
            s
        } else {
            panic!("expected Sgr")
        };
        let ops: alloc::vec::Vec<_> = sgr.ops().collect();
        assert_eq!(ops, &[SgrOp::FgBright8(3)]);
    }

    /// Phase 69 Track F — DECSCUSR cursor-shape sequences.
    #[test]
    fn test_decscusr_cursor_shapes() {
        for shape in 0..=6u16 {
            let seq = alloc::format!("\x1b[{} q", shape);
            assert_eq!(
                parse_str_last(&seq),
                ConsoleCmd::CursorShape { shape },
                "DECSCUSR {} did not parse",
                shape
            );
        }
        // Out-of-range shape is dropped to Nop without changing parser
        // state.
        assert_eq!(parse_str_last("\x1b[7 q"), ConsoleCmd::Nop);
        // Default shape (no param) parses as shape 0.
        assert_eq!(
            parse_str_last("\x1b[ q"),
            ConsoleCmd::CursorShape { shape: 0 }
        );
    }

    #[test]
    fn test_csi_private_unknown_final_byte() {
        // CSI ? 25 with an unknown final byte should produce Nop.
        assert_eq!(parse_str_last("\x1b[?25z"), ConsoleCmd::Nop);
    }

    #[test]
    fn test_cursor_movement_defaults() {
        // All cursor movement with no param defaults to 1.
        assert_eq!(parse_str_last("\x1b[A"), ConsoleCmd::CursorUp(1));
        assert_eq!(parse_str_last("\x1b[B"), ConsoleCmd::CursorDown(1));
        assert_eq!(parse_str_last("\x1b[D"), ConsoleCmd::CursorBack(1));
        assert_eq!(
            parse_str_last("\x1b[G"),
            ConsoleCmd::CursorHorizontalAbsolute(1)
        );
    }
}
