// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to carry the processing history of input visibilities over to output
//! visibilities.
//!
//! uvfits files keep their history as `COMMENT` and `HISTORY` cards in the
//! primary header; measurement sets keep theirs as rows of the `HISTORY`
//! table. `Marlu` writes hyperdrive's own entry when it creates an output
//! file; the code here reads the input's entries and appends them to output
//! files after they have been written.

use std::{ffi::CString, path::Path};

use fitsio::FitsFile;
use hifitime::Epoch;
use log::debug;
use marlu::rubbl_casatables::{Table, TableOpenMode};

use super::read::fits::FitsError;

/// The prefixes `Marlu` gives to the three parts of a history entry when it
/// writes them as uvfits `COMMENT`s.
const APPLICATION_PREFIX: &str = "Created by ";
const CMD_LINE_PREFIX: &str = "CmdLine: ";
const MESSAGE_PREFIX: &str = "Msg: ";

/// A `COMMENT` that goes before the input's history in a uvfits output, so
/// that it can be told apart from the entry of the program that wrote the
/// file.
const UVFITS_MARKER: &str = "--- History of the input data follows ---";

/// The `COMMENT`s cfitsio puts in every primary header; these aren't history.
const CFITSIO_BOILERPLATE: [&str; 2] = [
    "FITS (Flexible Image Transport System) format is defined in 'Astronomy",
    "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H",
];

/// The longest text that fits on one `COMMENT` or `HISTORY` card.
const CARD_TEXT_LEN: usize = 72;

/// One piece of an input file's processing history, kept in the form it was
/// read so that writing it back to the same format changes nothing.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum HistoryEntry {
    /// A `COMMENT` or `HISTORY` card from a uvfits file's primary header.
    /// cfitsio splits long text over several cards, so one card isn't
    /// necessarily a whole line of history.
    UvfitsCard {
        /// Is this a `HISTORY` card? Otherwise it's a `COMMENT`.
        is_history: bool,
        /// The card's text, without the keyword or trailing spaces.
        text: String,
    },

    /// A row of a measurement set's `HISTORY` table.
    MsRow(MsHistoryRow),
}

/// A row of a measurement set's `HISTORY` table.
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct MsHistoryRow {
    /// When the entry was made [MJD UTC seconds].
    pub(crate) time: f64,
    pub(crate) application: String,
    pub(crate) message: String,
    pub(crate) cli_command: Vec<String>,
    pub(crate) app_params: Vec<String>,
    pub(crate) priority: String,
    pub(crate) origin: String,
}

/// Read the history out of a uvfits file's primary header. The cards that
/// every file has (cfitsio's description of FITS, and the `AIPS WTSCAL`
/// `HISTORY` card that uvfits readers want) are left out.
pub(crate) fn read_uvfits_history(fptr: &mut FitsFile) -> Result<Vec<HistoryEntry>, FitsError> {
    let to_fits_error = |e: fitsio::errors::Error, fptr: &FitsFile| FitsError::Fitsio {
        fits_error: Box::new(e),
        fits_filename: fptr.file_path().into(),
        hdu_description: "0".to_string().into_boxed_str(),
        source_file: file!(),
        source_line: line!(),
        source_column: column!(),
    };
    // Make the primary HDU the current one.
    fptr.hdu(0).map_err(|e| to_fits_error(e, fptr))?;

    let mut num_cards = 0;
    let mut status = 0;
    unsafe {
        // ffghsp = fits_get_hdrspace
        fitsio_sys::ffghsp(
            fptr.as_raw(),
            &mut num_cards,
            std::ptr::null_mut(),
            &mut status,
        );
    }
    fitsio::errors::check_status(status).map_err(|e| to_fits_error(e, fptr))?;

    let mut entries = vec![];
    // A card is 80 characters, and cfitsio null-terminates it.
    let mut card = [0; 81];
    for i_card in 1..=num_cards {
        unsafe {
            // ffgrec = fits_read_record
            fitsio_sys::ffgrec(fptr.as_raw(), i_card, card.as_mut_ptr(), &mut status);
        }
        fitsio::errors::check_status(status).map_err(|e| to_fits_error(e, fptr))?;
        let card: String = card
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8 as char)
            .collect();
        let is_history = match card.get(..8) {
            Some("HISTORY ") => true,
            Some("COMMENT ") => false,
            _ => continue,
        };
        let text = card[8..].trim_end();
        if text.trim().is_empty() {
            continue;
        }
        let boilerplate = if is_history {
            text.trim_start().starts_with("AIPS WTSCAL")
        } else {
            CFITSIO_BOILERPLATE.contains(&text.trim())
        };
        if !boilerplate {
            entries.push(HistoryEntry::UvfitsCard {
                is_history,
                text: text.to_string(),
            });
        }
    }
    Ok(entries)
}

/// Read the rows of a measurement set's `HISTORY` table. A measurement set
/// doesn't need to have one, and those written by other software can lack
/// columns or have empty cells, so anything that can't be read is left out or
/// empty rather than being an error.
pub(crate) fn read_ms_history(ms: &Path) -> Vec<HistoryEntry> {
    let mut table = match Table::open(ms.join("HISTORY"), TableOpenMode::Read) {
        Ok(t) => t,
        Err(e) => {
            debug!("Couldn't open the HISTORY table of {}: {e}", ms.display());
            return vec![];
        }
    };
    (0..table.n_rows())
        .map(|row| {
            HistoryEntry::MsRow(MsHistoryRow {
                time: table.get_cell("TIME", row).unwrap_or_default(),
                application: table.get_cell("APPLICATION", row).unwrap_or_default(),
                message: table.get_cell("MESSAGE", row).unwrap_or_default(),
                cli_command: table
                    .get_cell_as_vec("CLI_COMMAND", row)
                    .unwrap_or_default(),
                app_params: table.get_cell_as_vec("APP_PARAMS", row).unwrap_or_default(),
                priority: table.get_cell("PRIORITY", row).unwrap_or_default(),
                origin: table.get_cell("ORIGIN", row).unwrap_or_default(),
            })
        })
        .collect()
}

/// Turn history entries into uvfits cards (is it a `HISTORY` card, and its
/// text). uvfits cards are passed through; measurement set rows are written
/// the way `Marlu` writes a history entry to a uvfits file.
fn to_uvfits_cards(entries: &[HistoryEntry]) -> Vec<(bool, String)> {
    let mut cards = vec![];
    for entry in entries {
        match entry {
            HistoryEntry::UvfitsCard { is_history, text } => {
                cards.push((*is_history, text.clone()))
            }
            HistoryEntry::MsRow(row) => {
                let cli_command = row.cli_command.join(" ");
                for (prefix, text) in [
                    (APPLICATION_PREFIX, row.application.as_str()),
                    (CMD_LINE_PREFIX, cli_command.as_str()),
                    (MESSAGE_PREFIX, row.message.as_str()),
                ] {
                    if text.trim().is_empty() {
                        continue;
                    }
                    // Split the text over as many cards as it needs, as
                    // cfitsio would, but on character boundaries. Anything
                    // that isn't printable ASCII isn't allowed in a card.
                    let line: Vec<char> = format!("{prefix}{text}")
                        .chars()
                        .map(|c| {
                            if c.is_ascii() && !c.is_ascii_control() {
                                c
                            } else {
                                '?'
                            }
                        })
                        .collect();
                    for chunk in line.chunks(CARD_TEXT_LEN) {
                        cards.push((false, chunk.iter().collect()));
                    }
                }
            }
        }
    }
    cards
}

/// Turn history entries into measurement set `HISTORY` rows. Rows are passed
/// through. uvfits cards are grouped into rows: a card starting with "Created
/// by" starts a new row, and the cards after it that continue its command line
/// or message (cfitsio splits long text over cards) are joined back together.
/// Cards that don't follow that pattern become the row's message, one line per
/// card. uvfits cards have no times, so their rows get `time_for_cards`.
fn to_ms_rows(entries: &[HistoryEntry], time_for_cards: f64) -> Vec<MsHistoryRow> {
    #[derive(Clone, Copy, PartialEq)]
    enum Part {
        CmdLine,
        Message,
        Other,
    }

    let mut rows: Vec<MsHistoryRow> = vec![];
    // The row being built from uvfits cards, its command line, and which part
    // the last card belonged to.
    let mut current: Option<(MsHistoryRow, String, Part)> = None;
    let finish = |current: &mut Option<(MsHistoryRow, String, Part)>,
                  rows: &mut Vec<MsHistoryRow>| {
        if let Some((mut row, cmd_line, _)) = current.take() {
            if !cmd_line.is_empty() {
                row.cli_command = vec![cmd_line];
            }
            rows.push(row);
        }
    };
    let new_row = || MsHistoryRow {
        time: time_for_cards,
        priority: "NORMAL".to_string(),
        origin: "standalone".to_string(),
        ..Default::default()
    };

    for entry in entries {
        match entry {
            HistoryEntry::MsRow(row) => {
                finish(&mut current, &mut rows);
                rows.push(row.clone());
            }
            HistoryEntry::UvfitsCard { text, .. } => {
                if text == UVFITS_MARKER {
                    finish(&mut current, &mut rows);
                } else if let Some(application) = text.strip_prefix(APPLICATION_PREFIX) {
                    finish(&mut current, &mut rows);
                    let mut row = new_row();
                    row.application = application.to_string();
                    current = Some((row, String::new(), Part::Other));
                } else {
                    let (row, cmd_line, part) =
                        current.get_or_insert_with(|| (new_row(), String::new(), Part::Other));
                    if let Some(c) = text.strip_prefix(CMD_LINE_PREFIX) {
                        cmd_line.push_str(c);
                        *part = Part::CmdLine;
                    } else if let Some(m) = text.strip_prefix(MESSAGE_PREFIX) {
                        if !row.message.is_empty() {
                            row.message.push('\n');
                        }
                        row.message.push_str(m);
                        *part = Part::Message;
                    } else {
                        match *part {
                            // A continuation of a long line.
                            Part::CmdLine => cmd_line.push_str(text),
                            Part::Message => row.message.push_str(text),
                            Part::Other => {
                                if !row.message.is_empty() {
                                    row.message.push('\n');
                                }
                                row.message.push_str(text);
                            }
                        }
                    }
                }
            }
        }
    }
    finish(&mut current, &mut rows);
    rows
}

/// Append history entries to the primary header of a uvfits file that has
/// been completely written and closed. If the header has no room left for the
/// cards, cfitsio makes room by moving everything after it, which takes a
/// while for a big file.
pub(crate) fn append_uvfits_history(
    uvfits: &Path,
    entries: &[HistoryEntry],
) -> Result<(), fitsio::errors::Error> {
    let cards = to_uvfits_cards(entries);
    if cards.is_empty() {
        return Ok(());
    }

    let mut fptr = FitsFile::edit(uvfits)?;
    fptr.hdu(0)?;
    let mut status = 0;
    for (is_history, text) in std::iter::once((false, UVFITS_MARKER.to_string())).chain(cards) {
        // The text came from a card or was made ASCII above, so it has no
        // nul bytes.
        let text = CString::new(text).expect("no nul bytes in history text");
        unsafe {
            if is_history {
                // ffphis = fits_write_history
                fitsio_sys::ffphis(fptr.as_raw(), text.as_ptr(), &mut status);
            } else {
                // ffpcom = fits_write_comment
                fitsio_sys::ffpcom(fptr.as_raw(), text.as_ptr(), &mut status);
            }
        }
        fitsio::errors::check_status(status)?;
    }
    Ok(())
}

/// Append history entries to the `HISTORY` table of a measurement set that
/// has been completely written and closed.
pub(crate) fn append_ms_history(
    ms: &Path,
    entries: &[HistoryEntry],
) -> Result<(), marlu::rubbl_casatables::TableError> {
    let now = Epoch::now()
        .map(|e| e.to_mjd_utc_seconds())
        .unwrap_or_default();
    let rows = to_ms_rows(entries, now);
    if rows.is_empty() {
        return Ok(());
    }

    let mut table = Table::open(ms.join("HISTORY"), TableOpenMode::ReadWrite)?;
    let first_new_row = table.n_rows();
    table.add_rows(rows.len())?;
    for (i_row, row) in (first_new_row..).zip(rows) {
        table.put_cell("TIME", i_row, &row.time)?;
        table.put_cell("OBSERVATION_ID", i_row, &0_i32)?;
        table.put_cell("MESSAGE", i_row, &row.message)?;
        table.put_cell("APPLICATION", i_row, &row.application)?;
        table.put_cell("PRIORITY", i_row, &row.priority)?;
        table.put_cell("ORIGIN", i_row, &row.origin)?;
        table.put_cell("APP_PARAMS", i_row, &row.app_params)?;
        table.put_cell("CLI_COMMAND", i_row, &row.cli_command)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(text: &str) -> HistoryEntry {
        HistoryEntry::UvfitsCard {
            is_history: false,
            text: text.to_string(),
        }
    }

    #[test]
    fn ms_rows_become_marlu_style_comments() {
        let long_arg = "x".repeat(100);
        let entries = [HistoryEntry::MsRow(MsHistoryRow {
            time: 5e9,
            application: "birli 0.15.1".to_string(),
            message: "cable length corrections".to_string(),
            cli_command: vec!["birli".to_string(), long_arg.clone()],
            ..Default::default()
        })];
        let cards = to_uvfits_cards(&entries);
        let cmd_line = format!("CmdLine: birli {long_arg}");
        assert_eq!(
            cards,
            vec![
                (false, "Created by birli 0.15.1".to_string()),
                (false, cmd_line[..72].to_string()),
                (false, cmd_line[72..].to_string()),
                (false, "Msg: cable length corrections".to_string()),
            ]
        );
        // No card is too long.
        assert!(cards.iter().all(|(_, text)| text.len() <= CARD_TEXT_LEN));
    }

    #[test]
    fn uvfits_cards_become_ms_rows() {
        let long_arg = "x".repeat(100);
        let cmd_line = format!("CmdLine: birli {long_arg}");
        let entries = [
            // Something before the first "Created by".
            card("a stray comment"),
            card("Created by birli 0.15.1"),
            card(&cmd_line[..72]),
            card(&cmd_line[72..]),
            card("Msg: cable length corrections, digital gains, pfb gains, aoflagging with"),
            card(" something"),
            card(UVFITS_MARKER),
            card("Created by cotter"),
        ];
        let rows = to_ms_rows(&entries, 123.0);
        let expected_row = |application: &str, message: &str, cli: Option<String>| MsHistoryRow {
            time: 123.0,
            application: application.to_string(),
            message: message.to_string(),
            cli_command: cli.into_iter().collect(),
            priority: "NORMAL".to_string(),
            origin: "standalone".to_string(),
            ..Default::default()
        };
        assert_eq!(
            rows,
            vec![
                expected_row("", "a stray comment", None),
                expected_row(
                    "birli 0.15.1",
                    "cable length corrections, digital gains, pfb gains, aoflagging with something",
                    Some(format!("birli {long_arg}"))
                ),
                expected_row("cotter", "", None),
            ]
        );
    }

    #[test]
    fn same_format_entries_pass_through() {
        let row = MsHistoryRow {
            time: 5e9,
            application: "app".to_string(),
            message: "msg".to_string(),
            cli_command: vec!["a".to_string(), "b".to_string()],
            app_params: vec!["p".to_string()],
            priority: "INFO".to_string(),
            origin: "somewhere".to_string(),
        };
        assert_eq!(
            to_ms_rows(&[HistoryEntry::MsRow(row.clone())], 0.0),
            vec![row]
        );

        let entries = [
            HistoryEntry::UvfitsCard {
                is_history: true,
                text: "some history".to_string(),
            },
            card("some comment"),
        ];
        assert_eq!(
            to_uvfits_cards(&entries),
            vec![
                (true, "some history".to_string()),
                (false, "some comment".to_string())
            ]
        );
    }
}
