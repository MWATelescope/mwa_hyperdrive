// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::fs::File;

use flate2::read::GzDecoder;
use tar::Archive;
use tempfile::TempDir;

use super::*;

#[test]
fn test_rts_solutions_correctly_read() {
    let temp_dir = TempDir::new().unwrap();
    // https://rust-lang-nursery.github.io/rust-cookbook/compression/tar.html#decompress-a-tarball-while-removing-a-prefix-from-the-paths
    let file = File::open("test_files/1088284872/1088284872_rts_sols.tar.gz").unwrap();
    let mut archive = Archive::new(GzDecoder::new(file));

    archive
        .entries()
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|mut entry| -> Result<PathBuf, std::io::Error> {
            let path = temp_dir.path().join(entry.path().unwrap());
            entry.unpack(&path).unwrap();
            Ok(path)
        })
        .filter_map(|e| e.ok())
        .for_each(|x| println!("> {}", x.display()));

    let sols = super::read(temp_dir.path(), "test_files/1088284872/1088284872.metafits");
    assert!(sols.is_ok(), "{:?}", sols.err());
    let sols = sols.unwrap();

    let flagged_tiles = [3, 112, 113, 114, 115, 116, 117, 118, 119, 123];
    for i_tile in 0..128 {
        if flagged_tiles.contains(&i_tile) {
            assert!(sols.flagged_tiles.contains(&i_tile), "{i_tile}");
        } else {
            assert!(!sols.flagged_tiles.contains(&i_tile), "{i_tile}");
        }
    }

    // 0, 1, 16, 30, 31 are standard flagged channels per coarse channel.
    let flagged_chans = [0, 1, 16, 30, 31];
    for i_chan in 0..768 {
        if flagged_chans.contains(&(i_chan % 32)) {
            assert!(sols.flagged_chanblocks.contains(&i_chan), "{i_chan}");
        } else {
            assert!(!sols.flagged_chanblocks.contains(&i_chan), "{i_chan}");
        }
    }

    assert_eq!(sols.obsid, Some(1088284872));

    assert_eq!(
        sols.di_jones[(0, 0, 2)],
        Jones::from([
            -8.62824812664832e-2,
            4.6851042422797773e-1,
            -1.961825514843333e-2,
            3.6931913974824733e-3,
            -1.5295973447010728e-2,
            7.09072986052293e-3,
            1.273485221868868e-1,
            4.978064938648525e-1
        ])
    );
}
