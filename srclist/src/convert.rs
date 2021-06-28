// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code to convert between sky-model source list files.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

use log::{info, trace, warn};

use crate::{
    ao, hyperdrive, rts, woden, HyperdriveFileType, SourceListType, SrclistError,
    WriteSourceListError,
};

pub fn convert<T: AsRef<Path>, S: AsRef<str>>(
    input_path: T,
    output_path: T,
    input_type: Option<S>,
    output_type: Option<S>,
) -> Result<(), SrclistError> {
    let input_path = input_path.as_ref().to_path_buf();
    let output_path = output_path.as_ref().to_path_buf();
    let input_type = input_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());
    let output_type = output_type.and_then(|t| SourceListType::from_str(t.as_ref()).ok());

    // When writing out hyperdrive-style source lists, we must know the output
    // file format. This can either be explicit from the user input, or the file
    // extension.
    let output_ext = output_path.extension().and_then(|e| e.to_str());
    let output_file_type = output_ext.and_then(|e| HyperdriveFileType::from_str(&e).ok());
    if output_type.is_none() && output_file_type.is_some() {
        warn!("Assuming that the output file type is 'hyperdrive'");
    }

    // Read the input source list.
    let (sl, sl_type) = crate::read::read_source_list_file(&input_path, input_type)?;
    if input_type.is_none() {
        info!(
            "Successfully read {} as a {}-style source list",
            input_path.display(),
            sl_type
        );
    }

    // Write the output source list.
    trace!("Attempting to write output source list");
    let mut f = BufWriter::new(File::create(&output_path)?);

    match (output_type, output_file_type) {
        (_, Some(HyperdriveFileType::Yaml)) => {
            hyperdrive::source_list_to_yaml(&mut f, &sl)?;
            info!(
                "Wrote hyperdrive-style source list to {}",
                output_path.display()
            );
        }
        (_, Some(HyperdriveFileType::Json)) => {
            hyperdrive::source_list_to_json(&mut f, &sl)?;
            info!(
                "Wrote hyperdrive-style source list to {}",
                output_path.display()
            );
        }
        (Some(SourceListType::Hyperdrive), None) => {
            return Err(WriteSourceListError::InvalidHyperdriveFormat(
                output_ext.unwrap_or("<no extension>").to_string(),
            )
            .into())
        }
        (Some(SourceListType::Rts), _) => {
            rts::write_source_list(&mut f, &sl)?;
            info!("Wrote rts-style source list to {}", output_path.display());
        }
        (Some(SourceListType::AO), _) => {
            ao::write_source_list(&mut f, &sl)?;
            info!("Wrote ao-style source list to {}", output_path.display());
        }
        (Some(SourceListType::Woden), _) => {
            woden::write_source_list(&mut f, &sl)?;
            info!("Wrote woden-style source list to {}", output_path.display());
        }
        (None, None) => return Err(WriteSourceListError::NotEnoughInfo.into()),
    }

    f.flush()?;

    Ok(())
}
