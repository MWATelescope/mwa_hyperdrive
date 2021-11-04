// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use mwa_rust_core::{
    math::cross_correlation_baseline_to_tiles, time::gps_to_epoch, HADec, Jones, RADec,
    XyzGeodetic, UVW,
};
use ndarray::prelude::*;
use rayon::prelude::*;
use tempfile::NamedTempFile;

use super::*;
use crate::math::TileBaselineMaps;
use mwa_hyperdrive_beam::Delays;

// TODO: Have in mwa_rust_core
/// Convert [XyzGeodetic] tile coordinates to [UVW] baseline coordinates without
/// having to form [XyzGeodetic] baselines first. This function performs
/// calculations in parallel. Cross-correlation baselines only.
pub fn xyzs_to_cross_uvws_parallel(xyzs: &[XyzGeodetic], phase_centre: HADec) -> Vec<UVW> {
    let (s_ha, c_ha) = phase_centre.ha.sin_cos();
    let (s_dec, c_dec) = phase_centre.dec.sin_cos();
    // Get a UVW for each tile.
    let tile_uvws: Vec<UVW> = xyzs
        .par_iter()
        .map(|&xyz| UVW::from_xyz_inner(xyz, s_ha, c_ha, s_dec, c_dec))
        .collect();
    // Take the difference of every pair of UVWs.
    let num_tiles = xyzs.len();
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    (0..num_baselines)
        .into_par_iter()
        .map(|i_bl| {
            let (i, j) = cross_correlation_baseline_to_tiles(num_tiles, i_bl);
            tile_uvws[i] - tile_uvws[j]
        })
        .collect()
}

#[test]
fn test_get_truncated_date_str() {
    let mjd = 56580.575370370374;
    let mjd_seconds = mjd * 24.0 * 3600.0;
    // The number of seconds between 1858-11-17T00:00:00 (MJD epoch) and
    // 1900-01-01T00:00:00 (TAI epoch) is 1297728000.
    let epoch_diff = 1297728000.0;
    let epoch = Epoch::from_tai_seconds(mjd_seconds - epoch_diff);
    assert_eq!(get_truncated_date_string(epoch), "2013-10-15T00:00:00.0");
}

#[test]
fn test_encode_uvfits_baseline() {
    assert_eq!(encode_uvfits_baseline(1, 1), 257);
    // TODO: Test the other part of the if statement.
}

#[test]
fn test_decode_uvfits_baseline() {
    assert_eq!(decode_uvfits_baseline(257), (1, 1));
    // TODO: Test the other part of the if statement.
}

#[test]
fn test_synthetic_data() {
    let result = uvfits::read::Uvfits::new(
        &"test_files/1196175296/1196175296.uvfits",
        None,
        &mut Delays::NotNecessary,
    );
    assert!(
        result.is_ok(),
        "Failed to read test_files/1196175296/1196175296.uvfits"
    );
    let uvfits = result.unwrap();
    let num_timesteps = uvfits.get_obs_context().timesteps.len();
    let num_chans = uvfits.get_freq_context().fine_chan_freqs.len();
    let flagged_chans = HashSet::new();
    let total_num_tiles = uvfits.get_obs_context().tile_xyzs.len();
    let tile_flags_set: HashSet<usize> = uvfits
        .get_obs_context()
        .tile_flags
        .iter()
        .cloned()
        .collect();
    let num_flagged_tiles = tile_flags_set.len();
    let num_tiles = total_num_tiles - num_flagged_tiles;
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let maps = crate::math::TileBaselineMaps::new(total_num_tiles, &tile_flags_set);

    let flagged_cross_rows = vec![
        10, 59, 61, 62, 79, 86, 90, 92, 94, 110, 136, 185, 187, 188, 205, 212, 216, 218, 220, 236,
        261, 310, 312, 313, 330, 337, 341, 343, 345, 361, 385, 434, 436, 437, 454, 461, 465, 467,
        469, 485, 508, 557, 559, 560, 577, 584, 588, 590, 592, 608, 630, 679, 681, 682, 699, 706,
        710, 712, 714, 730, 751, 800, 802, 803, 820, 827, 831, 833, 835, 851, 871, 920, 922, 923,
        940, 947, 951, 953, 955, 971, 990, 1039, 1041, 1042, 1059, 1066, 1070, 1072, 1074, 1090,
        1108, 1157, 1159, 1160, 1177, 1184, 1188, 1190, 1192, 1208, 1225, 1274, 1276, 1277, 1294,
        1301, 1305, 1307, 1309, 1325, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351,
        1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366,
        1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381,
        1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396,
        1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411,
        1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426,
        1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441,
        1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456,
        1457, 1505, 1507, 1508, 1525, 1532, 1536, 1538, 1540, 1556, 1619, 1621, 1622, 1639, 1646,
        1650, 1652, 1654, 1670, 1732, 1734, 1735, 1752, 1759, 1763, 1765, 1767, 1783, 1844, 1846,
        1847, 1864, 1871, 1875, 1877, 1879, 1895, 1955, 1957, 1958, 1975, 1982, 1986, 1988, 1990,
        2006, 2065, 2067, 2068, 2085, 2092, 2096, 2098, 2100, 2116, 2174, 2176, 2177, 2194, 2201,
        2205, 2207, 2209, 2225, 2282, 2284, 2285, 2302, 2309, 2313, 2315, 2317, 2333, 2389, 2391,
        2392, 2409, 2416, 2420, 2422, 2424, 2440, 2495, 2497, 2498, 2515, 2522, 2526, 2528, 2530,
        2546, 2600, 2602, 2603, 2620, 2627, 2631, 2633, 2635, 2651, 2704, 2706, 2707, 2724, 2731,
        2735, 2737, 2739, 2755, 2807, 2809, 2810, 2827, 2834, 2838, 2840, 2842, 2858, 2909, 2911,
        2912, 2929, 2936, 2940, 2942, 2944, 2960, 3010, 3012, 3013, 3030, 3037, 3041, 3043, 3045,
        3061, 3110, 3112, 3113, 3130, 3137, 3141, 3143, 3145, 3161, 3209, 3211, 3212, 3229, 3236,
        3240, 3242, 3244, 3260, 3307, 3309, 3310, 3327, 3334, 3338, 3340, 3342, 3358, 3404, 3406,
        3407, 3424, 3431, 3435, 3437, 3439, 3455, 3500, 3502, 3503, 3520, 3527, 3531, 3533, 3535,
        3551, 3595, 3597, 3598, 3615, 3622, 3626, 3628, 3630, 3646, 3689, 3691, 3692, 3709, 3716,
        3720, 3722, 3724, 3740, 3782, 3784, 3785, 3802, 3809, 3813, 3815, 3817, 3833, 3874, 3876,
        3877, 3894, 3901, 3905, 3907, 3909, 3925, 3965, 3967, 3968, 3985, 3992, 3996, 3998, 4000,
        4016, 4055, 4057, 4058, 4075, 4082, 4086, 4088, 4090, 4106, 4144, 4146, 4147, 4164, 4171,
        4175, 4177, 4179, 4195, 4232, 4234, 4235, 4252, 4259, 4263, 4265, 4267, 4283, 4319, 4321,
        4322, 4339, 4346, 4350, 4352, 4354, 4370, 4405, 4407, 4408, 4425, 4432, 4436, 4438, 4440,
        4456, 4490, 4492, 4493, 4510, 4517, 4521, 4523, 4525, 4541, 4574, 4576, 4577, 4594, 4601,
        4605, 4607, 4609, 4625, 4657, 4659, 4660, 4677, 4684, 4688, 4690, 4692, 4708, 4739, 4741,
        4742, 4759, 4766, 4770, 4772, 4774, 4790, 4820, 4822, 4823, 4840, 4847, 4851, 4853, 4855,
        4871, 4900, 4902, 4903, 4920, 4927, 4931, 4933, 4935, 4951, 4979, 4981, 4982, 4999, 5006,
        5010, 5012, 5014, 5030, 5057, 5059, 5060, 5077, 5084, 5088, 5090, 5092, 5108, 5134, 5136,
        5137, 5154, 5161, 5165, 5167, 5169, 5185, 5210, 5212, 5213, 5230, 5237, 5241, 5243, 5245,
        5261, 5285, 5287, 5288, 5305, 5312, 5316, 5318, 5320, 5336, 5359, 5361, 5362, 5379, 5386,
        5390, 5392, 5394, 5410, 5432, 5434, 5435, 5452, 5459, 5463, 5465, 5467, 5483, 5504, 5506,
        5507, 5524, 5531, 5535, 5537, 5539, 5555, 5575, 5577, 5578, 5595, 5602, 5606, 5608, 5610,
        5626, 5645, 5647, 5648, 5665, 5672, 5676, 5678, 5680, 5696, 5714, 5716, 5717, 5734, 5741,
        5745, 5747, 5749, 5765, 5782, 5784, 5785, 5802, 5809, 5813, 5815, 5817, 5833, 5850, 5851,
        5852, 5853, 5854, 5855, 5856, 5857, 5858, 5859, 5860, 5861, 5862, 5863, 5864, 5865, 5866,
        5867, 5868, 5869, 5870, 5871, 5872, 5873, 5874, 5875, 5876, 5877, 5878, 5879, 5880, 5881,
        5882, 5883, 5884, 5885, 5886, 5887, 5888, 5889, 5890, 5891, 5892, 5893, 5894, 5895, 5896,
        5897, 5898, 5899, 5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907, 5908, 5909, 5910, 5911,
        5912, 5913, 5914, 5915, 5916, 5917, 5918, 5935, 5942, 5946, 5948, 5950, 5966, 5983, 5984,
        5985, 5986, 5987, 5988, 5989, 5990, 5991, 5992, 5993, 5994, 5995, 5996, 5997, 5998, 5999,
        6000, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010, 6011, 6012, 6013, 6014,
        6015, 6016, 6017, 6018, 6019, 6020, 6021, 6022, 6023, 6024, 6025, 6026, 6027, 6028, 6029,
        6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043, 6044,
        6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6057, 6058, 6059,
        6060, 6061, 6062, 6063, 6064, 6065, 6066, 6067, 6068, 6069, 6070, 6071, 6072, 6073, 6074,
        6075, 6076, 6077, 6078, 6079, 6080, 6081, 6082, 6083, 6084, 6085, 6086, 6087, 6088, 6089,
        6090, 6091, 6092, 6093, 6094, 6095, 6096, 6097, 6098, 6099, 6100, 6101, 6102, 6103, 6104,
        6105, 6106, 6107, 6108, 6109, 6110, 6111, 6127, 6134, 6138, 6140, 6142, 6158, 6189, 6196,
        6200, 6202, 6204, 6220, 6250, 6257, 6261, 6263, 6265, 6281, 6310, 6317, 6321, 6323, 6325,
        6341, 6369, 6376, 6380, 6382, 6384, 6400, 6427, 6434, 6438, 6440, 6442, 6458, 6484, 6491,
        6495, 6497, 6499, 6515, 6540, 6547, 6551, 6553, 6555, 6571, 6595, 6602, 6606, 6608, 6610,
        6626, 6649, 6656, 6660, 6662, 6664, 6680, 6702, 6709, 6713, 6715, 6717, 6733, 6754, 6761,
        6765, 6767, 6769, 6785, 6805, 6812, 6816, 6818, 6820, 6836, 6855, 6862, 6866, 6868, 6870,
        6886, 6904, 6911, 6915, 6917, 6919, 6935, 6952, 6959, 6963, 6965, 6967, 6983, 7000, 7001,
        7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016,
        7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031,
        7032, 7033, 7034, 7035, 7036, 7037, 7038, 7039, 7040, 7041, 7042, 7043, 7044, 7045, 7046,
        7052, 7056, 7058, 7060, 7076, 7097, 7101, 7103, 7105, 7121, 7141, 7145, 7147, 7149, 7165,
        7184, 7188, 7190, 7192, 7208, 7226, 7230, 7232, 7234, 7250, 7267, 7271, 7273, 7275, 7291,
        7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7321, 7322,
        7323, 7324, 7325, 7326, 7327, 7328, 7329, 7330, 7331, 7332, 7333, 7334, 7335, 7336, 7337,
        7338, 7339, 7340, 7341, 7342, 7343, 7344, 7345, 7346, 7347, 7350, 7352, 7354, 7370, 7388,
        7390, 7392, 7408, 7425, 7427, 7429, 7445, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469,
        7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484,
        7485, 7486, 7487, 7488, 7489, 7490, 7491, 7492, 7493, 7494, 7495, 7496, 7497, 7498, 7500,
        7516, 7533, 7534, 7535, 7536, 7537, 7538, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546,
        7547, 7548, 7549, 7550, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560, 7561,
        7562, 7563, 7564, 7565, 7566, 7567, 7583, 7600, 7601, 7602, 7603, 7604, 7605, 7606, 7607,
        7608, 7609, 7610, 7611, 7612, 7613, 7614, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622,
        7623, 7624, 7625, 7626, 7627, 7628, 7629, 7630, 7631, 7646, 7676, 7705, 7733, 7760, 7786,
        7811, 7835, 7858, 7880, 7901, 7921, 7940, 7958, 7975, 7992, 7993, 7994, 7995, 7996, 7997,
        7998, 7999, 8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007,
    ];

    let mut cross_vis = Array2::zeros((num_cross_baselines, num_chans));
    let mut cross_weights = Array2::zeros((num_cross_baselines, num_chans));
    for timestep in (0..num_timesteps).into_iter() {
        let result = uvfits.read_crosses(
            cross_vis.view_mut(),
            cross_weights.view_mut(),
            timestep,
            &maps.tile_to_unflagged_baseline_map,
            &flagged_chans,
        );
        assert!(
            result.is_ok(),
            "Failed to read crosses: {:?}",
            result.unwrap_err()
        );
        result.unwrap();
        assert_abs_diff_eq!(cross_vis, cross_vis);
        for i in 0..cross_vis.dim().0 {
            let factor = if flagged_cross_rows.contains(&i) {
                -1.0
            } else {
                1.0
            } * 32.0;

            assert_abs_diff_eq!(
                cross_weights.slice(s![i, ..]),
                Array1::ones(cross_vis.dim().1) * factor
            );
        }

        let flagged_auto_rows = vec![11, 60, 62, 63, 80, 87, 91, 93, 95, 111];
        let mut auto_vis = Array2::zeros((num_tiles, num_chans));
        let mut auto_weights = Array2::zeros((num_tiles, num_chans));
        for timestep in (0..num_timesteps).into_iter() {
            let result = uvfits.read_autos(
                auto_vis.view_mut(),
                auto_weights.view_mut(),
                timestep,
                &tile_flags_set,
                &flagged_chans,
            );
            assert!(
                result.is_ok(),
                "Failed to read autos: {:?}",
                result.unwrap_err()
            );
            result.unwrap();
            assert_abs_diff_eq!(auto_vis, auto_vis);
            for i in 0..auto_vis.dim().0 {
                let factor = if flagged_auto_rows.contains(&i) {
                    -1.0
                } else {
                    1.0
                } * 32.0;
                assert_abs_diff_eq!(
                    auto_weights.slice(s![i, ..]),
                    Array1::ones(auto_vis.dim().1) * factor
                );
            }
        }
    }
    // TODO: Assert the visibility values properly.
    todo!();
}

#[test]
// Make a tiny uvfits file. The result has been verified by CASA's
// "importuvfits" function.
fn test_new_uvfits_is_sensible() {
    let tmp_uvfits_file = NamedTempFile::new().unwrap();
    let num_timesteps = 1;
    let num_tiles = 3;
    let num_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let num_chans = 2;
    let obsid = 1065880128.0;
    let start_epoch = gps_to_epoch(obsid);
    let maps = TileBaselineMaps::new(num_tiles, &HashSet::new());
    let chan_flags = HashSet::new();

    let mut u = UvfitsWriter::new(
        tmp_uvfits_file.path(),
        num_timesteps,
        num_baselines,
        num_chans,
        false,
        start_epoch,
        Some(40e3),
        170e6,
        3,
        RADec::new_degrees(0.0, 60.0),
        Some("test"),
        &maps.unflagged_baseline_to_tile_map,
        &chan_flags,
    )
    .unwrap();

    let mut f = u.open().unwrap();
    let mut row = vec![0.0; 5];
    row.append(&mut (0..num_chans).into_iter().map(|i| i as f32).collect());
    for _timestep_index in 0..num_timesteps {
        for baseline_index in 0..num_baselines {
            let (tile1, tile2) = maps.unflagged_baseline_to_tile_map[&baseline_index];
            u.write_vis(&mut f, UVW::default(), tile1, tile2, start_epoch, &mut row)
                .unwrap();
        }
    }

    let names = ["Tile1", "Tile2", "Tile3"];
    let positions: Vec<XyzGeodetic> = (0..names.len())
        .into_iter()
        .map(|i| XyzGeodetic {
            x: i as f64,
            y: i as f64 * 2.0,
            z: i as f64 * 3.0,
        })
        .collect();
    u.write_uvfits_antenna_table(&names, &positions).unwrap();
}

fn write_then_read_uvfits(autos: bool) {
    let output = NamedTempFile::new().expect("Couldn't create temporary file");
    let phase_centre = RADec::new_degrees(0.0, -27.0);
    let lst_rad = 0.0;
    let timesteps = [gps_to_epoch(1065880128.0)];
    let num_timesteps = timesteps.len();
    let num_tiles = 128;
    let autocorrelations_present = autos;
    let fine_chan_width_hz = 80000.0;
    let num_chans = 16;
    let fine_chan_freqs_hz: Vec<f64> = (0..num_chans)
        .into_iter()
        .map(|i| 150e6 + fine_chan_width_hz * i as f64)
        .collect();

    let (tile_names, xyzs): (Vec<String>, Vec<XyzGeodetic>) = (0..num_tiles)
        .into_iter()
        .map(|i| {
            (
                format!("Tile{}", i),
                XyzGeodetic {
                    x: 1.0 * i as f64,
                    y: 2.0 * i as f64,
                    z: 3.0 * i as f64,
                },
            )
        })
        .unzip();
    let num_cross_baselines = (num_tiles * (num_tiles - 1)) / 2;
    let uvws = xyzs_to_cross_uvws_parallel(&xyzs, phase_centre.to_hadec(lst_rad));
    let num_baselines = if autocorrelations_present {
        (num_tiles * (num_tiles + 1)) / 2
    } else {
        num_cross_baselines
    };

    let flagged_tiles = HashSet::new();
    let flagged_fine_chans = HashSet::new();
    let maps = TileBaselineMaps::new(num_tiles, &flagged_tiles);

    // Just in case this gets accidentally changed.
    assert_eq!(
        num_timesteps, 1,
        "num_timesteps should always be 1 for this test"
    );

    let result = UvfitsWriter::new(
        output.path(),
        num_timesteps,
        num_baselines,
        num_chans,
        autocorrelations_present,
        *timesteps.first().unwrap(),
        Some(fine_chan_width_hz),
        fine_chan_freqs_hz[num_chans / 2],
        num_chans / 2,
        phase_centre,
        None,
        &maps.unflagged_baseline_to_tile_map,
        &flagged_fine_chans,
    );
    assert!(result.is_ok(), "Failed to create new uvfits file");
    let mut output_writer = result.unwrap();

    let mut cross_vis = Array2::from_elem((num_cross_baselines, num_chans), Jones::identity());
    cross_vis
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v *= i as f32);
    let mut auto_vis = Array2::from_elem((num_tiles, num_chans), Jones::identity());
    auto_vis
        .iter_mut()
        .enumerate()
        .for_each(|(i, v)| *v *= i as f32);

    for timestep in timesteps {
        let result = if autocorrelations_present {
            output_writer.write_cross_and_auto_timestep_vis(
                cross_vis.view(),
                Array2::ones(cross_vis.dim()).view(),
                auto_vis.view(),
                Array2::ones(auto_vis.dim()).view(),
                &uvws,
                timestep,
            )
        } else {
            output_writer.write_cross_timestep_vis(
                cross_vis.view(),
                Array2::ones(cross_vis.dim()).view(),
                &uvws,
                timestep,
            )
        };
        assert!(
            result.is_ok(),
            "Failed to write visibilities to uvfits file: {:?}",
            result.unwrap_err()
        );
        result.unwrap();
    }

    let result = output_writer.write_uvfits_antenna_table(&tile_names, &xyzs);
    assert!(
        result.is_ok(),
        "Failed to finish writing uvfits file: {:?}",
        result.unwrap_err()
    );
    result.unwrap();

    // Inspect the file for sanity's sake!
    let result = uvfits::read::Uvfits::new(&output.path(), None, &mut Delays::NotNecessary);
    assert!(
        result.is_ok(),
        "Failed to read the just-created uvfits file"
    );
    let uvfits = result.unwrap();

    let mut cross_vis_read = Array2::zeros((num_cross_baselines, num_chans));
    let mut cross_weights_read = Array2::zeros((num_cross_baselines, num_chans));
    let mut auto_vis_read = Array2::zeros((num_tiles, num_chans));
    let mut auto_weights_read = Array2::zeros((num_tiles, num_chans));
    for (timestep, _) in timesteps.iter().enumerate() {
        let result = uvfits.read_crosses(
            cross_vis_read.view_mut(),
            cross_weights_read.view_mut(),
            timestep,
            &maps.tile_to_unflagged_baseline_map,
            &flagged_fine_chans,
        );
        assert!(
            result.is_ok(),
            "Failed to read crosses from the just-created uvfits file: {:?}",
            result.unwrap_err()
        );
        result.unwrap();
        assert_abs_diff_eq!(cross_vis_read, cross_vis);
        assert_abs_diff_eq!(cross_weights_read, Array2::ones(cross_vis.dim()));

        if autocorrelations_present {
            let result = uvfits.read_autos(
                auto_vis_read.view_mut(),
                auto_weights_read.view_mut(),
                timestep,
                &flagged_tiles,
                &flagged_fine_chans,
            );
            assert!(
                result.is_ok(),
                "Failed to read autos from the just-created uvfits file: {:?}",
                result.unwrap_err()
            );
            result.unwrap();

            assert_abs_diff_eq!(auto_vis_read, auto_vis);
            assert_abs_diff_eq!(auto_weights_read, Array2::ones(auto_vis.dim()));
        }
    }
}

#[test]
fn uvfits_io_works_for_cross_correlations() {
    write_then_read_uvfits(false)
}

#[test]
fn uvfits_io_works_for_auto_correlations() {
    write_then_read_uvfits(true)
}

// TODO: Test with some flagging.
