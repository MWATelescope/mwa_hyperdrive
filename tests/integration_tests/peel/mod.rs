use tempfile::TempDir;

use crate::{get_cmd_output, hyperdrive};

#[test]
fn test_1090008640_peel() {
    let tmp_dir = TempDir::new().expect("couldn't make tmp dir");
    let out_path = tmp_dir.path().join("peeled.uvfits");

    // Reading from a uvfits file without a metafits file should fail because
    // there's no beam information.
    let cmd = hyperdrive()
        .arg("peel")
        .arg("--data")
        .arg("test_files/1090008640_WODEN/output_band01.uvfits")
        .arg("test_files/1090008640_WODEN/1090008640.metafits")
        .arg("--source-list=test_files/1090008640_WODEN/srclist_3x3_grid.txt")
        .arg(format!("--outputs={}", out_path.display()))
        .arg("--iono-sub=1")
        .arg("--num-passes=2")
        .arg("--num-loops=1")
        .arg("--iono-time-average=8s")
        .arg("--iono-freq-average=1280kHz")
        .arg("--uvw-min=50lambda")
        .arg("--uvw-max=300lambda")
        .arg("--short-baseline-sigma=40")
        .arg("--convergence=0.9")
        .arg("--no-progress-bars")
        .ok();
    assert!(cmd.is_ok(), "{:?}", get_cmd_output(cmd));
}
