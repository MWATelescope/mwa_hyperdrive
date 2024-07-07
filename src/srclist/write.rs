// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    str::FromStr,
};

use log::{info, trace};

use super::{
    ao, fits, hyperdrive, rts, woden, HyperdriveFileType, SourceList, SourceListType,
    WriteSourceListError,
};

pub(crate) fn write_source_list(
    sl: &SourceList,
    path: &Path,
    input_srclist_type: SourceListType,
    output_srclist_type: Option<SourceListType>,
    num_sources: Option<usize>,
) -> Result<(), WriteSourceListError> {
    trace!("Attempting to write output source list");
    let mut f = BufWriter::new(File::create(path)?);
    let output_ext = path.extension().and_then(|e| e.to_str());
    let hyp_file_type = output_ext.and_then(|e| HyperdriveFileType::from_str(e).ok());

    let output_srclist_type = match (output_srclist_type, hyp_file_type) {
        (Some(t), _) => t,

        (None, Some(_)) => SourceListType::Hyperdrive,

        // Use the input source list type as the output type.
        (None, None) => input_srclist_type,
    };

    match (output_srclist_type, hyp_file_type) {
        (SourceListType::Hyperdrive, None) => {
            return Err(WriteSourceListError::InvalidHyperdriveFormat(
                output_ext.unwrap_or("<no extension>").to_string(),
            ))
        }
        (SourceListType::Fits, _) => fits::write_source_list_jack(path, sl, num_sources)?,
        (SourceListType::Rts, _) => {
            rts::write_source_list(&mut f, sl, num_sources)?;
            info!("Wrote rts-style source list to {}", path.display());
        }
        (SourceListType::AO, _) => {
            ao::write_source_list(&mut f, sl, num_sources)?;
            info!("Wrote ao-style source list to {}", path.display());
        }
        (SourceListType::Woden, _) => {
            woden::write_source_list(&mut f, sl, num_sources)?;
            info!("Wrote woden-style source list to {}", path.display());
        }
        (_, Some(HyperdriveFileType::Yaml)) => {
            hyperdrive::source_list_to_yaml(&mut f, sl, num_sources)?;
            info!("Wrote hyperdrive-style source list to {}", path.display());
        }
        (_, Some(HyperdriveFileType::Json)) => {
            hyperdrive::source_list_to_json(&mut f, sl, num_sources)?;
            info!("Wrote hyperdrive-style source list to {}", path.display());
        }
    }
    f.flush()?;

    Ok(())
}
