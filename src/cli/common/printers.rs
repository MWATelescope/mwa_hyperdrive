// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/// Pretty printers for reporting information.
use std::{borrow::Cow, sync::Mutex};

const VERTICAL: char = '│';
const UP_AND_RIGHT: char = '└';
const VERTICAL_AND_RIGHT: char = '├';

lazy_static::lazy_static! {
    static ref WARNING_PRINTER: Mutex<WarningPrinter> = Mutex::new(WarningPrinter::new());
}

pub(crate) struct InfoPrinter {
    title: Cow<'static, str>,
    blocks: Vec<Vec<Cow<'static, str>>>,
}

impl InfoPrinter {
    pub(crate) fn new(title: Cow<'static, str>) -> Self {
        Self {
            title,
            blocks: vec![],
        }
    }

    pub(super) fn overwrite_title(&mut self, title: Cow<'static, str>) {
        self.title = title;
    }

    pub(crate) fn push_line(&mut self, line: Cow<'static, str>) {
        self.blocks.push(vec![line]);
    }

    pub(crate) fn push_block(&mut self, block: Vec<Cow<'static, str>>) {
        self.blocks.push(block);
    }

    pub(crate) fn display(self) {
        log::info!("{}", console::style(self.title).bold());
        let num_blocks = self.blocks.len();
        for (i_block, block) in self.blocks.into_iter().enumerate() {
            let num_lines = block.len();
            for (i_line, line) in block.into_iter().enumerate() {
                let symbol = match (i_line, i_line + 1 == num_lines, i_block + 1 == num_blocks) {
                    (0, false, _) => VERTICAL_AND_RIGHT,
                    (0, _, false) => VERTICAL_AND_RIGHT,
                    (0, true, true) => UP_AND_RIGHT,
                    _ => VERTICAL,
                };
                log::info!("{symbol} {line}");
            }
        }
        log::info!("");
    }
}

struct WarningPrinter {
    blocks: Vec<Vec<Cow<'static, str>>>,
}

impl WarningPrinter {
    fn new() -> Self {
        Self { blocks: vec![] }
    }

    fn push_line(&mut self, line: Cow<'static, str>) {
        self.blocks.push(vec![line]);
    }

    fn push_block(&mut self, block: Vec<Cow<'static, str>>) {
        self.blocks.push(block);
    }

    fn display(&mut self) {
        log::debug!("Displaying warnings");
        if self.blocks.is_empty() {
            return;
        }

        log::warn!("{}", console::style("Warnings").bold());
        let num_blocks = self.blocks.len();
        for (i_block, block) in self.blocks.iter().enumerate() {
            let num_lines = block.len();
            for (i_line, line) in block.iter().enumerate() {
                let symbol = match (i_line, i_line + 1 == num_lines, i_block + 1 == num_blocks) {
                    (0, false, _) => VERTICAL_AND_RIGHT,
                    (0, _, false) => VERTICAL_AND_RIGHT,
                    (0, true, true) => UP_AND_RIGHT,
                    _ => VERTICAL,
                };
                log::warn!("{symbol} {line}");
            }
        }
        log::warn!("");
        self.blocks.clear();
    }
}

pub(crate) trait Warn {
    fn warn(self);
}

impl Warn for &'static str {
    fn warn(self) {
        WARNING_PRINTER.lock().unwrap().push_line(self.into());
    }
}

impl Warn for String {
    fn warn(self) {
        WARNING_PRINTER.lock().unwrap().push_line(self.into());
    }
}

impl Warn for Cow<'static, str> {
    fn warn(self) {
        WARNING_PRINTER.lock().unwrap().push_line(self);
    }
}

impl Warn for Vec<Cow<'static, str>> {
    fn warn(self) {
        WARNING_PRINTER.lock().unwrap().push_block(self);
    }
}

impl<const N: usize> Warn for [Cow<'static, str>; N] {
    fn warn(self) {
        WARNING_PRINTER.lock().unwrap().push_block(self.to_vec());
    }
}

/// Print out any warnings that have been collected as CLI arguments have been
/// parsed. This should only be called once before all arguments have been
/// parsed into parameters.
pub(crate) fn display_warnings() {
    WARNING_PRINTER.lock().unwrap().display();
}
