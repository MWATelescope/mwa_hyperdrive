// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::*;
use mwa_hyperdrive::sourcelist::read::parse_source_list;
use mwa_hyperdrive::sourcelist::source::Source;

pub(crate) fn verify_srclist(source_lists: Vec<PathBuf>) -> Result<(), anyhow::Error> {
    if source_lists.is_empty() {
        bail!("No source lists were supplied!");
    }

    for source_list in source_lists {
        let sources: Vec<Source> = {
            let mut f = std::fs::File::open(&source_list)?;
            let mut contents = String::new();
            f.read_to_string(&mut contents)?;
            parse_source_list(&contents)?
        };

        println!("{}:", source_list.to_string_lossy());
        println!(
            "{} sources, {} components\n",
            sources.len(),
            sources.iter().map(|s| s.components.len()).sum::<usize>()
        );
    }

    Ok(())
}
