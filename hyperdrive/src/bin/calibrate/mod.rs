// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use anyhow::{bail, ensure};
use log::debug;

use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;
use mwa_hyperdrive::*;

pub(crate) fn calibrate(
    cli_args: CalibrateUserArgs,
    param_file: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), anyhow::Error> {
    let params = mwa_hyperdrive::calibrate::args::merge_cli_and_file_args(cli_args, param_file)?;
    info!("Using metafits: {}", params.metafits.display());
    info!("Using gpubox files: {:?}", params.gpuboxes);

    debug!("Creating mwalib context");
    let context = match mwalibContext::new(&params.metafits, &params.gpuboxes) {
        Ok(c) => c,
        Err(e) => bail!("mwalib error: {}", e),
    };

    let flags = match &params.mwafs {
        Some(m) => Some({
            info!("Using mwaf files: {:?}", &m);
            let mut f = CotterFlags::new_from_mwafs(&m)?;

            // The cotter flags are available for all times. Make them match
            // only those we'll use according to mwalib.
            f.trim(&context);

            // Ensure that there is a mwaf file for each specified gpubox file.
            for cc in &context.coarse_channels {
                ensure!(
                    f.gpubox_nums.contains(&(cc.gpubox_number as u8)),
                    "gpubox file {} does not have a corresponding mwaf file specified",
                    cc.gpubox_number
                );
            }

            f
        }),
        None => {
            warn!("No cotter flags files specified");
            None
        }
    };

    if dry_run {
        println!("mwalib context:\n{}", context);
        if let Some(f) = flags {
            println!("cotter flags:\n{}", f);
        }
        return Ok(());
    }

    mwa_hyperdrive::calibrate::calibrate(&params)?;

    Ok(())
}
