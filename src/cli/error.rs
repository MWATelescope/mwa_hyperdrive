// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all hyperdrive-related errors. This should be the *only*
//! error enum that is publicly visible.

use thiserror::Error;

use super::{
    common::InputVisArgsError,
    di_calibrate::DiCalArgsError,
    peel::PeelArgsError,
    solutions::{SolutionsApplyArgsError, SolutionsPlotError},
    srclist::SrclistByBeamError,
    vis_convert::VisConvertArgsError,
    vis_simulate::VisSimulateArgsError,
    vis_subtract::VisSubtractArgsError,
};
use crate::{
    beam::BeamError,
    io::{
        read::VisReadError,
        write::{FileWriteError, VisWriteError},
        GlobError,
    },
    model::ModelError,
    params::{DiCalibrateError, PeelError, VisConvertError, VisSimulateError, VisSubtractError},
    solutions::{SolutionsReadError, SolutionsWriteError},
    srclist::{ReadSourceListError, SrclistError, WriteSourceListError},
};

const URL: &str = "https://MWATelescope.github.io/mwa_hyperdrive";

/// The *only* publicly visible error from hyperdrive. Each error message should
/// include the URL, unless it's "generic".
#[derive(Error, Debug)]
pub enum HyperdriveError {
    /// An error related to di-calibrate.
    #[error("{0}\n\nSee for more info: {URL}/user/di_cal/intro.html")]
    DiCalibrate(String),

    /// An error related to peeling.
    #[error("{0}\n\nSee for more info: {URL}/*****.html")]
    Peel(String),

    /// An error related to solutions-apply.
    #[error("{0}\n\nSee for more info: {URL}/user/solutions_apply/intro.html")]
    SolutionsApply(String),

    /// An error related to solutions-plot.
    #[error("{0}\n\nSee for more info: {URL}/user/plotting.html")]
    SolutionsPlot(String),

    /// An error related to vis-simulate.
    #[error("{0}\n\nSee for more info: {URL}/user/vis_simulate/intro.html")]
    VisSimulate(String),

    /// An error related to vis-subtract.
    #[error("{0}\n\nSee for more info: {URL}/user/vis_subtract/intro.html")]
    VisSubtract(String),

    /// Generic error surrounding source lists.
    #[error("{0}\n\nSee for more info: {URL}/defs/source_lists.html")]
    Srclist(String),

    /// Generic error surrounding calibration solutions.
    #[error("{0}\n\nSee for more info: {URL}/defs/cal_sols.html")]
    Solutions(String),

    /// Error specific to hyperdrive calibration solutions.
    #[error("{0}\n\nSee for more info: {URL}/defs/cal_sols_hyp.html")]
    SolutionsHyp(String),

    /// Error specific to AO calibration solutions.
    #[error("{0}\n\nSee for more info: {URL}/defs/cal_sols_ao.html")]
    SolutionsAO(String),

    /// Error specific to RTS calibration solutions.
    #[error("{0}\n\nSee for more info: {URL}/defs/cal_sols_rts.html")]
    SolutionsRts(String),

    /// An error related to reading visibilities.
    #[error("{0}\n\nSee for more info: {URL}/defs/vis_formats_read.html")]
    VisRead(String),

    /// An error related to reading visibilities.
    #[error("{0}\n\nSee for more info: {URL}/defs/vis_formats_write.html")]
    VisWrite(String),

    /// An error related to averaging.
    #[error("{0}\n\nSee for more info: {URL}/defs/vis_formats_write.html#visibility-averaging")]
    Averaging(String),

    /// An error related to raw MWA data corrections.
    #[error("{0}\n\nSee for more info: {URL}/defs/mwa/corrections.html")]
    RawDataCorrections(String),

    /// An error related to metafits files.
    #[error("{0}\n\nSee for more info: {URL}/defs/mwa/metafits.html")]
    Metafits(String),

    /// An error related to dipole delays.
    #[error("{0}\n\nYou may be able to fix this by supplying a metafits file or manually specifying the MWA dipole delays.\n\nSee for more info: {URL}/defs/mwa/delays.html")]
    Delays(String),

    /// An error related to mwaf files.
    #[error("{0}\n\nSee for more info: {URL}/defs/mwa/mwaf.html")]
    Mwaf(String),

    /// An error related to mwalib.
    #[error("{0}\n\nSee for more info: {URL}/defs/mwa/mwalib.html")]
    Mwalib(String),

    /// An error related to beam code.
    #[error("{0}\n\nSee for more info: {URL}/defs/beam.html")]
    Beam(String),

    /// An error related to argument files.
    #[error("{0}\n\nSee for more info: {URL}/defs/arg_file.html")]
    ArgFile(String),

    /// A cfitsio error. Because these are usually quite spartan, some
    /// suggestions are provided here.
    #[error("cfitsio error: {0}\n\nIf you don't know what this means, try turning up verbosity (-v or -vv) and maybe disabling progress bars.")]
    Cfitsio(String),

    /// A generic error that can't be clarified further with documentation, e.g.
    /// IO errors.
    #[error("{0}")]
    Generic(String),
}

// When changing the error propagation below, ensure `Self::from(e)` uses the
// correct `e`!

// Binary sub-command errors.

impl From<DiCalArgsError> for HyperdriveError {
    fn from(e: DiCalArgsError) -> Self {
        match e {
            DiCalArgsError::NoOutput
            | DiCalArgsError::AllBaselinesFlaggedFromUvwCutoffs
            | DiCalArgsError::ParseUvwMin(_)
            | DiCalArgsError::ParseUvwMax(_) => Self::DiCalibrate(e.to_string()),
            DiCalArgsError::CalibrationOutputFile { .. } => Self::Solutions(e.to_string()),
            DiCalArgsError::ParseCalTimeAverageFactor(_)
            | DiCalArgsError::CalTimeFactorNotInteger
            | DiCalArgsError::CalTimeResNotMultiple { .. }
            | DiCalArgsError::CalTimeFactorZero => Self::Averaging(e.to_string()),
            DiCalArgsError::IO(e) => Self::from(e),
        }
    }
}

impl From<DiCalibrateError> for HyperdriveError {
    fn from(e: DiCalibrateError) -> Self {
        let s = e.to_string();
        match e {
            DiCalibrateError::InsufficientMemory { .. } => Self::DiCalibrate(s),
            DiCalibrateError::SolutionsRead(_) | DiCalibrateError::SolutionsWrite(_) => {
                Self::Solutions(s)
            }
            DiCalibrateError::Fitsio(_) => Self::Cfitsio(s),
            DiCalibrateError::VisRead(e) => Self::from(e),
            DiCalibrateError::VisWrite(_) => Self::VisWrite(s),
            DiCalibrateError::Model(_) | DiCalibrateError::IO(_) => Self::Generic(s),
        }
    }
}

impl From<PeelArgsError> for HyperdriveError {
    fn from(e: PeelArgsError) -> Self {
        match e {
            PeelArgsError::NoOutput
            | PeelArgsError::NoChannels
            | PeelArgsError::ZeroPasses
            | PeelArgsError::ParseIonoTimeAverageFactor(_)
            | PeelArgsError::ParseIonoFreqAverageFactor(_)
            | PeelArgsError::IonoTimeFactorNotInteger
            | PeelArgsError::IonoFreqFactorNotInteger
            | PeelArgsError::IonoTimeResNotMultiple { .. }
            | PeelArgsError::IonoFreqResNotMultiple { .. }
            | PeelArgsError::IonoTimeFactorZero
            | PeelArgsError::IonoFreqFactorZero
            | PeelArgsError::ParseUvwMin(_)
            | PeelArgsError::ParseUvwMax(_) => Self::Generic(e.to_string()),
            PeelArgsError::Glob(e) => Self::from(e),
            PeelArgsError::VisRead(e) => Self::from(e),
            PeelArgsError::FileWrite(e) => Self::from(e),
            PeelArgsError::SourceList(e) => Self::from(e),
            PeelArgsError::Beam(e) => Self::from(e),
            PeelArgsError::Model(e) => Self::from(e),
            PeelArgsError::IO(e) => Self::from(e),
            #[cfg(any(feature = "cuda", feature = "hip"))]
            PeelArgsError::Gpu(e) => Self::from(e),
        }
    }
}

impl From<PeelError> for HyperdriveError {
    fn from(e: PeelError) -> Self {
        match e {
            PeelError::VisRead(e) => Self::from(e),
            PeelError::VisWrite(e) => Self::from(e),
            PeelError::FileWrite(e) => Self::from(e),
            PeelError::Beam(e) => Self::from(e),
            PeelError::Model(e) => Self::from(e),
            PeelError::IO(e) => Self::from(e),
            #[cfg(any(feature = "cuda", feature = "hip"))]
            PeelError::Gpu(e) => Self::from(e),
        }
    }
}

impl From<SolutionsApplyArgsError> for HyperdriveError {
    fn from(e: SolutionsApplyArgsError) -> Self {
        let s = e.to_string();
        match e {
            SolutionsApplyArgsError::NoSolutions => Self::SolutionsApply(s),
        }
    }
}

impl From<SolutionsPlotError> for HyperdriveError {
    fn from(e: SolutionsPlotError) -> Self {
        let s = e.to_string();
        match e {
            #[cfg(not(feature = "plotting"))]
            SolutionsPlotError::NoPlottingFeature => Self::SolutionsPlot(s),
            SolutionsPlotError::SolutionsRead(_) => Self::Solutions(s),
            SolutionsPlotError::Mwalib(_) => Self::Mwalib(s),
            SolutionsPlotError::IO(_) => Self::Generic(s),
            #[cfg(feature = "plotting")]
            SolutionsPlotError::MetafitsNoAntennaNames => Self::Metafits(s),
            #[cfg(feature = "plotting")]
            SolutionsPlotError::Draw(_)
            | SolutionsPlotError::NoInputs
            | SolutionsPlotError::InvalidSolsFormat(_) => Self::Generic(s),
        }
    }
}

impl From<VisConvertArgsError> for HyperdriveError {
    fn from(e: VisConvertArgsError) -> Self {
        let s = e.to_string();
        match e {
            VisConvertArgsError::NoOutputs => Self::VisWrite(s),
        }
    }
}

impl From<VisConvertError> for HyperdriveError {
    fn from(e: VisConvertError) -> Self {
        match e {
            VisConvertError::VisRead(e) => Self::from(e),
            VisConvertError::VisWrite(e) => Self::from(e),
            VisConvertError::IO(e) => Self::from(e),
        }
    }
}

impl From<VisSimulateArgsError> for HyperdriveError {
    fn from(e: VisSimulateArgsError) -> Self {
        let s = e.to_string();
        match e {
            VisSimulateArgsError::NoMetafits
            | VisSimulateArgsError::MetafitsDoesntExist(_)
            | VisSimulateArgsError::RaInvalid
            | VisSimulateArgsError::DecInvalid
            | VisSimulateArgsError::OnlyOneRAOrDec
            | VisSimulateArgsError::FineChansZero
            | VisSimulateArgsError::FineChansWidthTooSmall
            | VisSimulateArgsError::ZeroTimeSteps
            | VisSimulateArgsError::BadArrayPosition { .. } => Self::VisSimulate(s),
        }
    }
}

impl From<VisSimulateError> for HyperdriveError {
    fn from(e: VisSimulateError) -> Self {
        match e {
            VisSimulateError::VisWrite(e) => Self::from(e),
            VisSimulateError::Model(e) => Self::from(e),
            VisSimulateError::IO(e) => Self::from(e),
        }
    }
}

impl From<VisSubtractArgsError> for HyperdriveError {
    fn from(e: VisSubtractArgsError) -> Self {
        let s = e.to_string();
        match e {
            VisSubtractArgsError::MissingSource { .. }
            | VisSubtractArgsError::NoSources
            | VisSubtractArgsError::AllSourcesFiltered => Self::VisSubtract(s),
        }
    }
}

impl From<VisSubtractError> for HyperdriveError {
    fn from(e: VisSubtractError) -> Self {
        match e {
            VisSubtractError::VisRead(e) => Self::from(e),
            VisSubtractError::VisWrite(e) => Self::from(e),
            VisSubtractError::Model(e) => Self::from(e),
            VisSubtractError::IO(e) => Self::from(e),
            #[cfg(any(feature = "cuda", feature = "hip"))]
            VisSubtractError::Gpu(e) => Self::from(e),
        }
    }
}

impl From<SrclistByBeamError> for HyperdriveError {
    fn from(e: SrclistByBeamError) -> Self {
        match e {
            SrclistByBeamError::NoPhaseCentre => todo!(),
            SrclistByBeamError::NoLst => todo!(),
            SrclistByBeamError::NoFreqs => todo!(),
            SrclistByBeamError::ReadSourceList(e) => Self::from(e),
            SrclistByBeamError::WriteSourceList(e) => Self::from(e),
            SrclistByBeamError::Beam(e) => Self::from(e),
            SrclistByBeamError::Mwalib(e) => Self::from(e),
            SrclistByBeamError::IO(e) => Self::from(e),
        }
    }
}

// Library code errors.

impl From<InputVisArgsError> for HyperdriveError {
    fn from(e: InputVisArgsError) -> Self {
        let s = e.to_string();
        match e {
            InputVisArgsError::Raw(
                crate::io::read::RawReadError::MwafFlagsMissingForTimestep { .. }
                | crate::io::read::RawReadError::MwafMerge(_),
            ) => Self::Mwaf(s),
            InputVisArgsError::PfbParse(_) => Self::RawDataCorrections(s),
            InputVisArgsError::DoesNotExist(_)
            | InputVisArgsError::CouldNotRead(_)
            | InputVisArgsError::PpdMetafitsUnsupported(_)
            | InputVisArgsError::NotRecognised(_)
            | InputVisArgsError::NoInputData
            | InputVisArgsError::MultipleMetafits(_)
            | InputVisArgsError::MultipleMeasurementSets(_)
            | InputVisArgsError::MultipleUvfits(_)
            | InputVisArgsError::MultipleSolutions(_)
            | InputVisArgsError::InvalidDataInput(_)
            | InputVisArgsError::BadArrayPosition { .. }
            | InputVisArgsError::NoTimesteps
            | InputVisArgsError::DuplicateTimesteps
            | InputVisArgsError::UnavailableTimestep { .. }
            | InputVisArgsError::NoTiles
            | InputVisArgsError::BadTileIndexForFlagging { .. }
            | InputVisArgsError::BadTileNameForFlagging(_)
            | InputVisArgsError::NoChannels
            | InputVisArgsError::FineChanFlagTooBig { .. }
            | InputVisArgsError::FineChanFlagPerCoarseChanTooBig { .. }
            | InputVisArgsError::Raw(_)
            | InputVisArgsError::Ms(_)
            | InputVisArgsError::Uvfits(_) => Self::VisRead(s),
            InputVisArgsError::TileCountMismatch { .. } | InputVisArgsError::Solutions(_) => {
                Self::Solutions(s)
            }
            InputVisArgsError::ParseTimeAverageFactor(_)
            | InputVisArgsError::TimeFactorNotInteger
            | InputVisArgsError::TimeResNotMultiple { .. }
            | InputVisArgsError::ParseFreqAverageFactor(_)
            | InputVisArgsError::FreqFactorNotInteger
            | InputVisArgsError::FreqResNotMultiple { .. } => Self::Averaging(s),
            InputVisArgsError::Glob(_) | InputVisArgsError::IO(_, _) => Self::Generic(s),
        }
    }
}

impl From<VisReadError> for HyperdriveError {
    fn from(e: VisReadError) -> Self {
        let s = e.to_string();
        match e {
            VisReadError::Raw(_) | VisReadError::MS(_) | VisReadError::Uvfits(_) => {
                Self::VisRead(s)
            }
            VisReadError::BadArraySize { .. } => Self::Generic(s),
        }
    }
}

impl From<VisWriteError> for HyperdriveError {
    fn from(e: VisWriteError) -> Self {
        Self::VisWrite(e.to_string())
    }
}

impl From<FileWriteError> for HyperdriveError {
    fn from(e: FileWriteError) -> Self {
        Self::VisWrite(e.to_string())
    }
}

impl From<ReadSourceListError> for HyperdriveError {
    fn from(e: ReadSourceListError) -> Self {
        let s = e.to_string();
        match e {
            ReadSourceListError::IO(_) => Self::Generic(s),
            _ => Self::Srclist(s),
        }
    }
}

impl From<WriteSourceListError> for HyperdriveError {
    fn from(e: WriteSourceListError) -> Self {
        let s = e.to_string();
        match e {
            WriteSourceListError::UnsupportedComponentType { .. }
            | WriteSourceListError::UnsupportedFluxDensityType { .. }
            | WriteSourceListError::InvalidHyperdriveFormat(_)
            | WriteSourceListError::Sexagesimal(_) => Self::Srclist(s),
            WriteSourceListError::IO(e) => Self::from(e),
            WriteSourceListError::Yaml(_)
            | WriteSourceListError::Json(_)
            | WriteSourceListError::Fitsio(_)
            | WriteSourceListError::Fits(_) => Self::Generic(s),
        }
    }
}

impl From<SrclistError> for HyperdriveError {
    fn from(e: SrclistError) -> Self {
        let s = e.to_string();
        match e {
            SrclistError::ReadSourceList(e) => Self::from(e),
            SrclistError::Beam(e) => Self::from(e),
            SrclistError::WriteSourceList(_) => Self::Srclist(s),
            SrclistError::MissingMetafits => Self::Metafits(s),
            SrclistError::Mwalib(_) => Self::Mwalib(s),
            SrclistError::IO(e) => Self::from(e),
        }
    }
}

impl From<SolutionsReadError> for HyperdriveError {
    fn from(e: SolutionsReadError) -> Self {
        let s = e.to_string();
        match e {
            SolutionsReadError::UnsupportedExt { .. } => Self::Solutions(s),
            SolutionsReadError::BadShape { .. } | SolutionsReadError::ParsePfbFlavour(_) => {
                Self::SolutionsHyp(s)
            }
            SolutionsReadError::AndreBinaryStr { .. }
            | SolutionsReadError::AndreBinaryVal { .. } => Self::SolutionsAO(s),
            SolutionsReadError::RtsMetafitsRequired | SolutionsReadError::Rts(_) => {
                Self::SolutionsRts(s)
            }
            SolutionsReadError::Fits(_) | SolutionsReadError::Fitsio(_) => Self::Cfitsio(s),
            SolutionsReadError::IO(e) => Self::from(e),
        }
    }
}

impl From<SolutionsWriteError> for HyperdriveError {
    fn from(e: SolutionsWriteError) -> Self {
        let s = e.to_string();
        match e {
            SolutionsWriteError::UnsupportedExt { .. } => Self::Solutions(s),
            SolutionsWriteError::Fits(_) | SolutionsWriteError::Fitsio(_) => Self::Cfitsio(s),
            SolutionsWriteError::IO(e) => Self::from(e),
        }
    }
}

impl From<BeamError> for HyperdriveError {
    fn from(e: BeamError) -> Self {
        let s = e.to_string();
        match e {
            BeamError::NoDelays(_)
            | BeamError::BadDelays
            | BeamError::InconsistentDelays { .. }
            | BeamError::DelayGainsDimensionMismatch { .. } => Self::Delays(s),
            BeamError::Unrecognised(_)
            | BeamError::BadTileIndex { .. }
            | BeamError::Hyperbeam(_)
            | BeamError::HyperbeamInit(_) => Self::Beam(s),
            #[cfg(any(feature = "cuda", feature = "hip"))]
            BeamError::Gpu(_) => Self::Beam(s),
        }
    }
}

impl From<ModelError> for HyperdriveError {
    fn from(e: ModelError) -> Self {
        match e {
            ModelError::Beam(e) => Self::from(e),

            #[cfg(any(feature = "cuda", feature = "hip"))]
            ModelError::Gpu(e) => Self::from(e),
        }
    }
}

impl From<GlobError> for HyperdriveError {
    fn from(e: GlobError) -> Self {
        Self::Generic(e.to_string())
    }
}

impl From<std::io::Error> for HyperdriveError {
    fn from(e: std::io::Error) -> Self {
        Self::Generic(e.to_string())
    }
}

impl From<mwalib::MwalibError> for HyperdriveError {
    fn from(e: mwalib::MwalibError) -> Self {
        Self::Mwalib(e.to_string())
    }
}

#[cfg(any(feature = "cuda", feature = "hip"))]
impl From<crate::gpu::GpuError> for HyperdriveError {
    fn from(e: crate::gpu::GpuError) -> Self {
        Self::Generic(e.to_string())
    }
}
