// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Build the C shim around EveryBeam, and link against EveryBeam.
//!
//! Without the `vendored` feature, an installed EveryBeam (>= 0.9) and casacore
//! are used. They are searched for in the prefixes given by these environment
//! variables (`:`-separated), then `$CONDA_PREFIX`, `/opt/everybeam`,
//! `/opt/casacore`, `/usr/local` and `/usr`:
//! - `EVERYBEAM_DIR`: EveryBeam's installation prefix(es). Headers are expected
//!   in `include` and `include/EveryBeam`, libraries in `lib` or `lib64`.
//! - `CASACORE_DIR`: casacore's installation prefix(es).
//!
//! Other environment variables:
//! - `EVERYBEAM_CXXFLAGS`: Extra (whitespace-separated) flags for the C++
//!   compiler.
//! - `EVERYBEAM_LIBS`: Extra (whitespace-separated) libraries to link.
//!
//! With the `vendored` feature, casacore and EveryBeam (and their header-only
//! dependencies) are downloaded into `OUT_DIR`, built, and linked statically.
//! Downloads are verified (by SHA-256 for tarballs, by commit for git
//! repositories). For offline builds, the source of any dependency can be
//! supplied as an extracted directory with `EVERYBEAM_SYS_<NAME>_SRC`, where
//! `<NAME>` is one of `CASACORE`, `EVERYBEAM`, `AOCOMMON`, `SKA_SDP_FUNC`,
//! `XTL`, `XTENSOR` or `EIGEN`.

use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/shim.cpp");
    println!("cargo:rerun-if-changed=shim/shim.h");

    #[cfg(feature = "vendored")]
    vendored::build();
    #[cfg(not(feature = "vendored"))]
    system::build();
}

/// Compile the shim with the supplied include directories (which must make
/// `<EveryBeam/...>` headers available).
fn compile_shim(include_dirs: &[PathBuf], extra_flags: &[String]) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .file("shim/shim.cpp")
        // Warnings from EveryBeam's and casacore's headers aren't ours to fix.
        .flag_if_supported("-Wno-unused-parameter");
    for dir in include_dirs {
        // Don't add the compiler's default directories; doing so breaks
        // #include_next in the C++ standard library.
        if dir.exists()
            && dir != Path::new("/usr/include")
            && dir != Path::new("/usr/local/include")
        {
            // Use -isystem so that warnings from these headers are ignored.
            build
                .flag("-isystem")
                .flag(dir.to_str().expect("path is UTF-8"));
        }
    }
    for flag in extra_flags {
        build.flag(flag);
    }
    build.compile("everybeam_shim");
}

fn env_flags(var: &str) -> Vec<String> {
    println!("cargo:rerun-if-env-changed={var}");
    env::var(var)
        .map(|v| v.split_whitespace().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

#[cfg(not(feature = "vendored"))]
mod system {
    use super::*;

    /// Get installation prefixes from an environment variable (`:`
    /// separated), followed by common defaults.
    fn get_prefixes(var: &str, extra_default: &str) -> Vec<PathBuf> {
        println!("cargo:rerun-if-env-changed={var}");
        println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
        let mut prefixes: Vec<PathBuf> = match env::var_os(var) {
            Some(v) if !v.is_empty() => env::split_paths(&v).collect(),
            _ => vec![],
        };
        if let Some(conda) = env::var_os("CONDA_PREFIX") {
            prefixes.push(PathBuf::from(conda));
        }
        for p in [extra_default, "/usr/local", "/usr"] {
            prefixes.push(PathBuf::from(p));
        }
        prefixes.dedup();
        prefixes
    }

    pub(super) fn build() {
        let everybeam_prefixes = get_prefixes("EVERYBEAM_DIR", "/opt/everybeam");
        let casacore_prefixes = get_prefixes("CASACORE_DIR", "/opt/casacore");

        // Find EveryBeam's headers, so that we can give a helpful error.
        let everybeam_prefix = everybeam_prefixes
            .iter()
            .find(|p| p.join("include/EveryBeam/load.h").exists())
            .unwrap_or_else(|| {
                panic!(
                    "Couldn't find EveryBeam (looked for include/EveryBeam/load.h in {everybeam_prefixes:?}). \
                     Set EVERYBEAM_DIR to EveryBeam's installation prefix, or use the 'vendored' feature \
                     (hyperdrive's 'everybeam-vendored' feature) to build it automatically."
                )
            });

        let mut include_dirs = vec![
            everybeam_prefix.join("include"),
            everybeam_prefix.join("include/EveryBeam"),
        ];
        for prefix in &casacore_prefixes {
            include_dirs.push(prefix.join("include"));
            include_dirs.push(prefix.join("include/casacore"));
        }
        compile_shim(&include_dirs, &env_flags("EVERYBEAM_CXXFLAGS"));

        // Link. Export the library directories, so that dependents can embed
        // an rpath to them (rustc-link-arg doesn't propagate to dependents).
        let mut lib_dirs = vec![];
        for prefix in std::iter::once(everybeam_prefix).chain(casacore_prefixes.iter()) {
            for lib_dir in ["lib", "lib64"] {
                let dir = prefix.join(lib_dir);
                if dir.exists() && !lib_dirs.contains(&dir) {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                    lib_dirs.push(dir);
                }
            }
        }
        println!("cargo:rustc-link-lib=dylib=everybeam");
        for lib in env_flags("EVERYBEAM_LIBS") {
            println!("cargo:rustc-link-lib={lib}");
        }
        // System directories don't need an rpath.
        let rpath: Vec<String> = lib_dirs
            .iter()
            .filter(|d| !d.starts_with("/usr/lib") && !d.starts_with("/usr/lib64"))
            .map(|d| d.display().to_string())
            .collect();
        println!("cargo:rpath={}", rpath.join(":"));
    }
}

#[cfg(feature = "vendored")]
mod vendored {
    use std::{fs, io::Read, process::Command};

    use sha2::{Digest, Sha256};

    use super::*;

    enum Fetch {
        /// A tarball, verified by its SHA-256.
        Tarball {
            url: &'static str,
            sha256: &'static str,
        },
        /// A git repository at a specific commit.
        Git {
            url: &'static str,
            commit: &'static str,
        },
    }

    struct Source {
        /// Used for the `EVERYBEAM_SYS_<NAME>_SRC` override.
        name: &'static str,
        /// The top-level directory inside the tarball.
        dir: &'static str,
        fetch: Fetch,
    }

    const CASACORE: Source = Source {
        name: "CASACORE",
        dir: "casacore-3.8.2",
        fetch: Fetch::Tarball {
            url: "https://github.com/casacore/casacore/archive/v3.8.2.tar.gz",
            sha256: "aa6cf40aaadc71b85d10718d5140479cce6f889f506f8fdf1a7a4b90fb7dfee5",
        },
    };
    // The PyPI sdist contains EveryBeam's full C++ source (without git
    // submodules, which are fetched separately).
    const EVERYBEAM: Source = Source {
        name: "EVERYBEAM",
        dir: "everybeam-0.9.0",
        fetch: Fetch::Tarball {
            url:
                "https://files.pythonhosted.org/packages/source/e/everybeam/everybeam-0.9.0.tar.gz",
            sha256: "1da7fe2c58c5a1b2712f0462ee97433e2987ffe1218b2a03af6c6230eefe426a",
        },
    };
    const AOCOMMON: Source = Source {
        name: "AOCOMMON",
        dir: "aocommon",
        fetch: Fetch::Git {
            url: "https://gitlab.com/aroffringa/aocommon.git",
            commit: "97bbc3e3b6dcd9b392271ec45d0c528a62e530ab",
        },
    };
    // The commit used by EveryBeam 0.9.0.
    const SKA_SDP_FUNC: Source = Source {
        name: "SKA_SDP_FUNC",
        dir: "ska-sdp-func",
        fetch: Fetch::Git {
            url: "https://gitlab.com/ska-telescope/sdp/ska-sdp-func.git",
            commit: "7f691cb376883c234383712c206b1ac8a7a3e58f",
        },
    };
    const XTL: Source = Source {
        name: "XTL",
        dir: "xtl-0.8.2",
        fetch: Fetch::Tarball {
            url: "https://github.com/xtensor-stack/xtl/archive/0.8.2.tar.gz",
            sha256: "8fb38d6a5856aab5740d2ccb3d791d289f648d4cc506b94a1338fe5fce100c11",
        },
    };
    const XTENSOR: Source = Source {
        name: "XTENSOR",
        dir: "xtensor-0.27.1",
        fetch: Fetch::Tarball {
            url: "https://github.com/xtensor-stack/xtensor/archive/0.27.1.tar.gz",
            sha256: "117c192ae3b7c37c0156dedaa88038e0599a6b264666c3c6c2553154b500fe23",
        },
    };
    // Eigen 3.4.0.
    const EIGEN: Source = Source {
        name: "EIGEN",
        dir: "eigen",
        fetch: Fetch::Git {
            url: "https://gitlab.com/libeigen/eigen.git",
            commit: "3147391d946bb4b6c68edd901f2add6ac1f31f8c",
        },
    };

    /// EveryBeam's source files, excluding aterms (which aren't needed for
    /// beam responses, and need extra dependencies). This is
    /// `EVERYBEAM_FILENAMES` in EveryBeam's cpp/CMakeLists.txt, plus the
    /// element-response libraries.
    const EVERYBEAM_FILES: &[&str] = &[
        "antenna.cc",
        "beamformer.cc",
        "beamformer_fastmath.cc",
        "beamformeridenticalantennas.cc",
        "beamformerlofar.cc",
        "beamformerlofarhba.cc",
        "beamformerlofarlba.cc",
        "beammode.cc",
        "beamnormalisationmode.cc",
        "circularsymmetric/atcacoefficients.cc",
        "circularsymmetric/gmrtcoefficients.cc",
        "circularsymmetric/meerkatcoefficients.cc",
        "circularsymmetric/vlacoefficients.cc",
        "circularsymmetric/voltagepattern.cc",
        "common/directionlist.cc",
        "common/sphericalharmonics.cc",
        "common/fftresampler.cc",
        "coords/itrfconverter.cc",
        "coords/itrfdirection.cc",
        "eep/eepgrid.cc",
        "elementmodel/dishresponse.cc",
        "elementmodel/dipoleelementmodel.cc",
        "elementmodel/meerkatelementmodel.cc",
        "element.cc",
        "elementhamaker.cc",
        "elementresponse.cc",
        "elementresponsefactory.cc",
        "everybeam.cc",
        "griddedresponse/aartfaacgrid.cc",
        "griddedresponse/airygrid.cc",
        "griddedresponse/dishgrid.cc",
        "griddedresponse/griddedresponse.cc",
        "griddedresponse/mwagrid.cc",
        "griddedresponse/phasedarraygrid.cc",
        "griddedresponse/skamidgrid.cc",
        "h5.cc",
        "load/load_new.cc",
        "load/loadlofar.cc",
        "load/loadmeerkat.cc",
        "load.cc",
        "lobes/lobeselementresponse.cc",
        "lwa/lwaelementresponse.cc",
        "msreadutils.cc",
        "mwabeam/tilebeam2016.cc",
        "mwabeam/beam2016implementation.cc",
        "options.cc",
        "phasedarrayresponse.cc",
        "pointresponse/airypoint.cc",
        "pointresponse/dishpoint.cc",
        "pointresponse/mwapoint.cc",
        "pointresponse/phasedarraypoint.cc",
        "pointresponse/skamidpoint.cc",
        "response/multistationresponse.cc",
        "response/singlestationresponse.cc",
        "sphericalharmonicsresponse.cc",
        "sphericalharmonicsresponsefixeddirection.cc",
        "station.cc",
        "telescope/alma.cc",
        "telescope/dish.cc",
        "telescope/dsa110.cc",
        "telescope/lofar.cc",
        "telescope/lwa.cc",
        "telescope/mwa.cc",
        "telescope/phasedarray.cc",
        "telescope/oskar.cc",
        "telescope/skamid.cc",
        "telescope.cc",
        // Element-response libraries.
        "hamaker/hamakerelementresponse.cc",
        "hamaker/hamakercoeff.cc",
        "oskar/oskarelementresponse.cc",
        "oskar/oskardatafile.cc",
        "oskar/oskardataset.cc",
        "skalowelementbeam/skalowelementresponse.cc",
        "skalowelementbeam/skalowfekocoefficients.cc",
        "skamidbeam/skamidanalyticalresponse.cc",
    ];

    /// The parts of ska-sdp-func that EveryBeam uses.
    const SKA_SDP_FUNC_FILES: &[&str] = &[
        "station_beam/sdp_element_dipole.cpp",
        "station_beam/sdp_element_spherical_wave_feko.cpp",
        "station_beam/sdp_element_spherical_wave_harp.cpp",
        "utility/sdp_mem.cpp",
        "utility/sdp_device_wrapper.cpp",
        "utility/sdp_logging.c",
    ];

    /// casacore's static libraries, in link order.
    const CASACORE_LIBS: &[&str] = &[
        "casa_ms",
        "casa_derivedmscal",
        "casa_meas",
        "casa_measures",
        "casa_scimath",
        "casa_scimath_f",
        "casa_tables",
        "casa_casa",
    ];

    fn sha256_of(path: &Path) -> String {
        let mut file = fs::File::open(path).unwrap();
        let mut hasher = Sha256::new();
        let mut buf = vec![0; 1 << 16];
        loop {
            let n = file.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    fn run(cmd: &mut Command, what: &str) {
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("Couldn't run {what} ({cmd:?}): {e}"));
        if !status.success() {
            panic!("{what} failed ({cmd:?}): {status}");
        }
    }

    /// Get the source directory for a dependency, downloading and extracting
    /// it if necessary.
    fn get_source(source: &Source, out_dir: &Path) -> PathBuf {
        let var = format!("EVERYBEAM_SYS_{}_SRC", source.name);
        println!("cargo:rerun-if-env-changed={var}");
        if let Some(dir) = env::var_os(&var) {
            return PathBuf::from(dir);
        }

        let src_root = out_dir.join("src");
        let dir = src_root.join(source.dir);
        let done_marker = src_root.join(format!("{}.done", source.dir));
        if done_marker.exists() {
            return dir;
        }
        fs::create_dir_all(&src_root).unwrap();
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }

        match source.fetch {
            Fetch::Tarball { url, sha256 } => {
                let downloads = out_dir.join("downloads");
                fs::create_dir_all(&downloads).unwrap();
                let tarball = downloads.join(format!("{}.tar.gz", source.dir));
                if !tarball.exists() || sha256_of(&tarball) != sha256 {
                    run(
                        Command::new("curl")
                            .args(["--fail", "--location", "--silent", "--show-error"])
                            .args(["--retry", "3", "--output"])
                            .arg(&tarball)
                            .arg(url),
                        &format!("Downloading {url} (set {var} to use a local copy)"),
                    );
                }
                let got = sha256_of(&tarball);
                if got != sha256 {
                    panic!("SHA-256 of {url} is {got}, but expected {sha256}");
                }
                let gz = flate2::read::GzDecoder::new(fs::File::open(&tarball).unwrap());
                tar::Archive::new(gz).unpack(&src_root).unwrap();
            }

            Fetch::Git { url, commit } => {
                fs::create_dir_all(&dir).unwrap();
                let git = |args: &[&str]| {
                    run(
                        Command::new("git").arg("-C").arg(&dir).args(args),
                        &format!("Fetching {url} (set {var} to use a local copy)"),
                    )
                };
                git(&["init", "--quiet"]);
                git(&["fetch", "--quiet", "--depth", "1", url, commit]);
                git(&[
                    "-c",
                    "advice.detachedHead=false",
                    "checkout",
                    "--quiet",
                    commit,
                ]);
            }
        }
        assert!(dir.exists(), "Expected {} to exist", dir.display());
        fs::write(&done_marker, "").unwrap();
        dir
    }

    fn copy_dir(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let dest = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &dest);
            } else {
                fs::copy(entry.path(), dest).unwrap();
            }
        }
    }

    /// The directory containing the Fortran runtime library (libgfortran).
    fn fortran_runtime_dir() -> Option<PathBuf> {
        println!("cargo:rerun-if-env-changed=FC");
        let fc = env::var("FC").unwrap_or_else(|_| "gfortran".to_string());
        let out = Command::new(fc)
            .arg("-print-file-name=libgfortran.so")
            .output()
            .ok()?;
        let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        // If the library isn't found, the bare file name is printed.
        if path.is_absolute() && path.exists() {
            path.parent().map(|p| p.to_path_buf())
        } else {
            None
        }
    }

    /// The directory containing Boost.DateTime's library, if it's installed.
    fn boost_date_time_dir() -> Option<PathBuf> {
        println!("cargo:rerun-if-env-changed=BOOST_ROOT");
        let mut dirs: Vec<PathBuf> = ["BOOST_ROOT", "CONDA_PREFIX"]
            .iter()
            .filter_map(env::var_os)
            .map(|p| PathBuf::from(p).join("lib"))
            .collect();
        dirs.extend(
            [
                "/usr/lib64",
                "/usr/lib",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",
                "/usr/local/lib",
            ]
            .map(PathBuf::from),
        );
        dirs.into_iter()
            .find(|d| d.join("libboost_date_time.so").exists())
    }

    /// Find HDF5's include directory and library directory.
    fn find_hdf5() -> (Vec<PathBuf>, Vec<PathBuf>) {
        println!("cargo:rerun-if-env-changed=HDF5_DIR");
        if let Some(dir) = env::var_os("HDF5_DIR") {
            let dir = PathBuf::from(dir);
            return (vec![dir.join("include")], vec![dir.join("lib")]);
        }
        let mut includes = vec![];
        let mut libs = vec![];
        if let Ok(out) = Command::new("pkg-config")
            .args(["--cflags-only-I", "--libs-only-L", "hdf5"])
            .output()
        {
            for token in String::from_utf8_lossy(&out.stdout).split_whitespace() {
                if let Some(p) = token.strip_prefix("-I") {
                    includes.push(PathBuf::from(p));
                } else if let Some(p) = token.strip_prefix("-L") {
                    libs.push(PathBuf::from(p));
                }
            }
        }
        // Debian/Ubuntu put the serial HDF5 headers here.
        includes.push(PathBuf::from("/usr/include/hdf5/serial"));
        (includes, libs)
    }

    fn build_casacore(src: &Path, out_dir: &Path) -> PathBuf {
        let dst = cmake::Config::new(src)
            .out_dir(out_dir.join("casacore"))
            // Always optimise; debug casacore is very slow.
            .profile("Release")
            .define("MODULE", "ms")
            .define("ENABLE_SHARED", "OFF")
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .define("BUILD_PYTHON", "OFF")
            .define("BUILD_PYTHON3", "OFF")
            .define("BUILD_TESTING", "OFF")
            .define("BUILD_SISCO", "OFF")
            .define("USE_OPENMP", "OFF")
            .define("USE_HDF5", "OFF")
            .define("USE_FFTW3", "OFF")
            .define("USE_READLINE", "OFF")
            .define("USE_THREADS", "ON")
            // casacore's measures data are found at runtime (see .casarc).
            .define("DATA_DIR", "")
            .build();
        dst
    }

    /// Write EveryBeam's generated headers, and patch the source so that the
    /// data directory can be set at runtime (see `eb_set_data_dir`).
    /// Whether the C++ compiler (with the user's flags, e.g. `-march`) enables
    /// aocommon's AVX matrices; EveryBeam's config.h must agree with it.
    fn compiler_has_avx_matrix() -> bool {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let probe = out_dir.join("avx_probe.cpp");
        fs::write(
            &probe,
            "#if defined(__AVX2__) && defined(__FMA__)\nHYPERDRIVE_AVX_MATRIX\n#endif\n",
        )
        .unwrap();
        let expanded = cc::Build::new().cpp(true).file(&probe).expand();
        String::from_utf8_lossy(&expanded).contains("HYPERDRIVE_AVX_MATRIX")
    }

    fn prepare_everybeam(src: &Path) {
        let cpp = src.join("cpp");
        let coeffs = src.join("coeffs");
        let config = fs::read_to_string(src.join("CMake/config.h.in"))
            .unwrap()
            .replace("@EVERYBEAM_DATADIR@", "share/everybeam")
            .replace(
                "@EVERYBEAM_ABSOLUTE_DATADIR@",
                coeffs.to_str().expect("path is UTF-8"),
            )
            .replace(
                "@COMPILED_WITH_AVX_MATRIX@",
                if compiler_has_avx_matrix() { "1" } else { "0" },
            );
        // Blank any remaining (test-only) substitutions.
        let config: String = config
            .lines()
            .map(|l| match (l.find('@'), l.rfind('@')) {
                (Some(a), Some(b)) if a < b => format!("{}{}", &l[..a], &l[b + 1..]),
                _ => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(cpp.join("config.h"), config).unwrap();
        let version = fs::read_to_string(src.join("CMake/version.h.in"))
            .unwrap()
            .replace("@EVERYBEAM_VERSION@", "0.9.0")
            .replace("@EVERYBEAM_VERSION_MAJOR@", "0")
            .replace("@EVERYBEAM_VERSION_MINOR@", "9")
            .replace("@EVERYBEAM_VERSION_PATCH@", "0");
        fs::write(cpp.join("version.h"), version).unwrap();

        let options_cc = cpp.join("options.cc");
        let options = fs::read_to_string(&options_cc).unwrap();
        let marker = "std::filesystem::path GetDataDirectory() {";
        if !options.contains("hyperdrive_everybeam_data_dir") {
            assert!(
                options.contains(marker),
                "EveryBeam's options.cc has changed; update the patch in build.rs"
            );
            let patched = options.replace(
                marker,
                &format!(
                    "extern \"C\" const char* hyperdrive_everybeam_data_dir();\n{marker}\n  \
                     if (const char* dir = hyperdrive_everybeam_data_dir())\n    \
                     return std::filesystem::path(dir);"
                ),
            );
            fs::write(&options_cc, patched).unwrap();
        }

        // libc++ (macOS) lacks the C++17 special math functions; EveryBeam
        // already uses Boost's version elsewhere.
        if cfg!(target_os = "macos") {
            let vp_cc = cpp.join("circularsymmetric/voltagepattern.cc");
            let vp = fs::read_to_string(&vp_cc).unwrap();
            if vp.contains("std::cyl_bessel_j") {
                let patched = vp
                    .replace("std::cyl_bessel_j", "boost::math::cyl_bessel_j")
                    .replacen(
                        "#include <cmath>",
                        "#include <cmath>\n#include <boost/math/special_functions/bessel.hpp>",
                        1,
                    );
                fs::write(&vp_cc, patched).unwrap();
            }
        }
    }

    /// Generate Rust code embedding EveryBeam's data files.
    fn embed_data(src: &Path, out_dir: &Path) {
        let coeffs = src.join("coeffs");
        let mut files = vec![];
        for name in ["oskar.h5", "HamakerHBACoeff.h5", "HamakerLBACoeff.h5"] {
            files.push(name.to_string());
        }
        let mut skalow: Vec<String> = fs::read_dir(coeffs.join("skalow"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".h5"))
            .collect();
        skalow.sort();
        files.extend(skalow.into_iter().map(|n| format!("skalow/{n}")));

        let mut code = String::from(
            "/// EveryBeam's version.\npub const EVERYBEAM_VERSION: &str = \"0.9.0\";\n\n\
             /// The data directory used at build time. This may not exist at runtime.\n",
        );
        code += &format!(
            "pub const BUILD_DATA_DIR: &str = {:?};\n\n",
            coeffs.display().to_string()
        );
        code += "/// The data files, as (relative path, contents).\npub static FILES: &[(&str, &[u8])] = &[\n";
        for f in &files {
            let path = coeffs.join(f);
            assert!(path.exists(), "Expected {} to exist", path.display());
            code += &format!(
                "    ({f:?}, include_bytes!({:?})),\n",
                path.display().to_string()
            );
        }
        code += "];\n";
        fs::write(out_dir.join("everybeam_data.rs"), code).unwrap();
    }

    pub(super) fn build() {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

        let casacore_src = get_source(&CASACORE, &out_dir);
        // EveryBeam's source is patched, so never modify a user-supplied copy.
        let everybeam_src = {
            let src = get_source(&EVERYBEAM, &out_dir);
            if src.starts_with(&out_dir) {
                src
            } else {
                let copy = out_dir.join("src").join(EVERYBEAM.dir);
                if !copy.exists() {
                    copy_dir(&src, &copy);
                }
                copy
            }
        };
        let aocommon = get_source(&AOCOMMON, &out_dir);
        let ska_sdp_func = get_source(&SKA_SDP_FUNC, &out_dir);
        let xtl = get_source(&XTL, &out_dir);
        let xtensor = get_source(&XTENSOR, &out_dir);
        let eigen = get_source(&EIGEN, &out_dir);

        // casacore.
        let casacore = build_casacore(&casacore_src, &out_dir);
        let casacore_include = casacore.join("include");
        let casacore_lib_dirs: Vec<PathBuf> = ["lib", "lib64"]
            .iter()
            .map(|d| casacore.join(d))
            .filter(|d| d.exists())
            .collect();

        // EveryBeam.
        prepare_everybeam(&everybeam_src);
        let cpp = everybeam_src.join("cpp");
        // Make the headers available as <EveryBeam/...>, as when installed.
        let include = out_dir.join("include");
        fs::create_dir_all(&include).unwrap();
        let link = include.join("EveryBeam");
        if fs::symlink_metadata(&link).is_err() {
            std::os::unix::fs::symlink(&cpp, &link).unwrap();
        }
        let (hdf5_includes, hdf5_lib_dirs) = find_hdf5();
        let sdp_src = ska_sdp_func.join("src");
        let mut include_dirs = vec![
            cpp.clone(),
            aocommon.join("include"),
            xtl.join("include"),
            xtensor.join("include"),
            eigen.clone(),
            casacore_include.clone(),
            casacore_include.join("casacore"),
            sdp_src.clone(),
        ];
        include_dirs.extend(hdf5_includes);
        for sub in ["hamaker", "oskar", "skalowelementbeam", "skamidbeam"] {
            include_dirs.push(cpp.join(sub));
        }

        // Static libraries are linked in the order that they're emitted, and a
        // library must come before the libraries it depends on: the shim,
        // then EveryBeam, then ska-sdp-func, then casacore.
        let mut shim_includes = vec![include.clone()];
        shim_includes.extend(include_dirs.iter().cloned());
        compile_shim(&shim_includes, &env_flags("EVERYBEAM_CXXFLAGS"));

        let mut eb = cc::Build::new();
        eb.cpp(true)
            .std("c++20")
            .opt_level(2)
            .pic(true)
            .warnings(false)
            // As EveryBeam's CMakeLists.txt does; HDF5 >= 1.12 otherwise
            // defaults to a newer, incompatible API.
            .define("H5_USE_110_API", None)
            .flag_if_supported("-fopenmp")
            .flag_if_supported("-w")
            // aocommon's MC2x2Base converting constructor narrows double to
            // float; GCC only warns, clang errors.
            .flag_if_supported("-Wno-c++11-narrowing");
        for dir in &include_dirs {
            eb.include(dir);
        }
        for f in EVERYBEAM_FILES {
            let path = cpp.join(f);
            println!("cargo:rerun-if-changed={}", path.display());
            eb.file(path);
        }
        eb.compile("everybeam_vendored");

        let mut sdp_cpp = cc::Build::new();
        sdp_cpp
            .cpp(true)
            .std("c++17")
            .opt_level(2)
            .pic(true)
            .warnings(false)
            .include(&sdp_src);
        let mut sdp_c = cc::Build::new();
        sdp_c
            .opt_level(2)
            .pic(true)
            .warnings(false)
            .include(&sdp_src);
        for f in SKA_SDP_FUNC_FILES {
            let path = sdp_src.join("ska-sdp-func").join(f);
            if f.ends_with(".c") {
                sdp_c.file(path);
            } else {
                sdp_cpp.file(path);
            }
        }
        sdp_cpp.compile("everybeam_sdp_cpp");
        sdp_c.compile("everybeam_sdp_c");

        // Link casacore statically, then system libraries dynamically.
        for dir in &casacore_lib_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        for lib in CASACORE_LIBS {
            println!("cargo:rustc-link-lib=static={lib}");
        }
        for dir in &hdf5_lib_dirs {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        // Debian/Ubuntu put the serial HDF5 C++ library here.
        for dir in [
            "/usr/lib/x86_64-linux-gnu/hdf5/serial",
            "/usr/lib/aarch64-linux-gnu/hdf5/serial",
        ] {
            if Path::new(dir).exists() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
        // casacore's Fortran code needs the Fortran runtime. Ask the Fortran
        // compiler where it is, because it isn't always on the linker's
        // default search path (e.g. with versioned compilers like gfortran-10).
        if let Some(dir) = fortran_runtime_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        // GSL is used by casacore's Dysco storage manager, which is needed to
        // open Dysco-compressed measurement sets.
        for lib in [
            "hdf5_cpp", "hdf5", "fftw3f", "gsl", "gslcblas", "lapack", "blas", "gfortran",
        ] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
        // Boost.DateTime is header-only from Boost 1.73, but older Boosts (e.g.
        // RHEL 8's) need its library.
        if let Some(dir) = boost_date_time_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=dylib=boost_date_time");
        }
        for lib in env_flags("EVERYBEAM_LIBS") {
            println!("cargo:rustc-link-lib={lib}");
        }
        // Nothing to add to the rpath; EveryBeam and casacore are static.
        println!("cargo:rpath=");
        // Some GSLs (e.g. RHEL's) leave their CBLAS symbols for the program to
        // provide, but nothing else links to gslcblas, so the linker's
        // --as-needed would drop it. Dependents must link it with
        // --no-as-needed (link-arg directives don't propagate from here).
        println!("cargo:no_as_needed=gslcblas");

        embed_data(&everybeam_src, &out_dir);
    }
}
