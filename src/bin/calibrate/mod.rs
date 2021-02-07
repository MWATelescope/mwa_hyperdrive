// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use log::info;

use mwa_hyperdrive::calibrate::args::CalibrateUserArgs;

pub(crate) fn calibrate(
    cli_args: CalibrateUserArgs,
    args_file: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), anyhow::Error> {
    let params = cli_args.merge(args_file)?.to_params()?;

    if dry_run {
        println!("mwalib context:\n{}", params.context);
        if let Some(f) = params.cotter_flags {
            println!("cotter flags:\n{}", f);
        }

        return Ok(());
    }

    info!("Using metafits: {}", params.context.metafits_filename);

    {
        let mut gpubox_filenames = vec![];
        for batch in &params.context.gpubox_batches {
            for gpubox_file in &batch.gpubox_files {
                gpubox_filenames.push(&gpubox_file.filename);
            }
        }
        info!("Using gpubox files: {:#?}", gpubox_filenames);
    }
    // TODO: Show what files were actually used, not just specified as arguments
    // (could be a glob)
    // match &args.mwafs {
    //     Some(mwafs) => info!("Using mwaf files: {:#?}", &mwafs),
    //     None => (),
    // }

    mwa_hyperdrive::calibrate::calibrate(params)?;

    Ok(())
}
