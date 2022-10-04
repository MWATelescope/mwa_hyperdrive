// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Error type for all hyperdrive-related errors. This should be the *only*
//! error enum that is publicly visible.

use thiserror::Error;

use crate::{
    cli::{
        di_calibrate::DiCalArgsError,
        solutions::{apply::SolutionsApplyError, plot::SolutionsPlotError},
        vis_utils::{simulate::VisSimulateError, subtract::VisSubtractError},
    },
    di_calibrate::DiCalibrateError,
    filenames::InputFileError,
    solutions::{SolutionsReadError, SolutionsWriteError},
    srclist::SrclistError,
    vis_io::read::VisReadError,
};

const URL: &str = "https://MWATelescope.github.io/mwa_hyperdrive";

/// The *only* publicly visible error from hyperdrive. Each error message should
/// include the URL, unless it's "generic".
#[derive(Error, Debug)]
pub enum HyperdriveError {
    /// An error related to di-calibrate.
    #[error("{0}\n\nSee for more info: {URL}/user/di_cal/intro.html")]
    DiCalibrate(String),

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
    #[error("{0}\n\nSee for more info: {URL}/defs/mwa/delays.html")]
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

impl From<DiCalibrateError> for HyperdriveError {
    fn from(e: DiCalibrateError) -> Self {
        let s = e.to_string();
        match e {
            DiCalibrateError::InsufficientMemory { .. }
            | DiCalibrateError::TimestepUnavailable { .. } => Self::DiCalibrate(s),
            DiCalibrateError::DiCalArgs(e) => Self::from(e),
            DiCalibrateError::SolutionsRead(_) | DiCalibrateError::SolutionsWrite(_) => {
                Self::Solutions(s)
            }
            DiCalibrateError::Fitsio(_) => Self::Cfitsio(s),
            DiCalibrateError::VisRead(e) => Self::from(e),
            DiCalibrateError::VisWrite(_) => Self::VisWrite(s),
            DiCalibrateError::Beam(_) | DiCalibrateError::IO(_) => Self::Generic(s),
        }
    }
}

impl From<SolutionsApplyError> for HyperdriveError {
    fn from(e: SolutionsApplyError) -> Self {
        let s = e.to_string();
        match e {
            SolutionsApplyError::NoInputData | SolutionsApplyError::TileCountMismatch { .. } => {
                Self::SolutionsApply(s)
            }
            SolutionsApplyError::MultipleMetafits(_)
            | SolutionsApplyError::MultipleMeasurementSets(_)
            | SolutionsApplyError::MultipleUvfits(_)
            | SolutionsApplyError::InvalidDataInput(_) => Self::VisRead(s),
            SolutionsApplyError::InvalidOutputFormat(_) | SolutionsApplyError::NoOutput => {
                Self::VisWrite(s)
            }
            SolutionsApplyError::NoTiles
            | SolutionsApplyError::TileFlag(_)
            | SolutionsApplyError::NoTimesteps
            | SolutionsApplyError::DuplicateTimesteps
            | SolutionsApplyError::UnavailableTimestep { .. }
            | SolutionsApplyError::BadArrayPosition { .. } => Self::Generic(s),
            SolutionsApplyError::ParsePfbFlavour(_) => Self::RawDataCorrections(s),
            SolutionsApplyError::ParseOutputVisTimeAverageFactor(_)
            | SolutionsApplyError::ParseOutputVisFreqAverageFactor(_)
            | SolutionsApplyError::OutputVisTimeFactorNotInteger
            | SolutionsApplyError::OutputVisFreqFactorNotInteger
            | SolutionsApplyError::OutputVisTimeAverageFactorZero
            | SolutionsApplyError::OutputVisFreqAverageFactorZero
            | SolutionsApplyError::OutputVisTimeResNotMultiple { .. }
            | SolutionsApplyError::OutputVisFreqResNotMultiple { .. } => Self::Averaging(s),
            SolutionsApplyError::SolutionsRead(_) => Self::Solutions(s),
            SolutionsApplyError::VisRead(e) => Self::from(e),
            SolutionsApplyError::FileWrite(_) | SolutionsApplyError::VisWrite(_) => {
                Self::VisWrite(s)
            }
            SolutionsApplyError::IO(_) => Self::Generic(s),
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

impl From<VisSimulateError> for HyperdriveError {
    fn from(e: VisSimulateError) -> Self {
        let s = e.to_string();
        match e {
            VisSimulateError::RaInvalid
            | VisSimulateError::DecInvalid
            | VisSimulateError::OnlyOneRAOrDec
            | VisSimulateError::FineChansZero
            | VisSimulateError::FineChansWidthTooSmall
            | VisSimulateError::ZeroTimeSteps
            | VisSimulateError::BadArrayPosition { .. } => Self::VisSimulate(s),
            VisSimulateError::BadDelays => Self::Delays(s),
            VisSimulateError::SourceList(_) | VisSimulateError::Veto(_) => Self::Srclist(s),
            VisSimulateError::Beam(_) => Self::Beam(s),
            VisSimulateError::Mwalib(_) => Self::Mwalib(s),
            VisSimulateError::InvalidOutputFormat(_) | VisSimulateError::VisWrite(_) => {
                Self::VisWrite(s)
            }
            VisSimulateError::AverageFactor(_) => Self::Averaging(s),
            VisSimulateError::Glob(_)
            | VisSimulateError::FileWrite(_)
            | VisSimulateError::IO(_) => Self::Generic(s),
            #[cfg(feature = "cuda")]
            VisSimulateError::CudaError(_) => Self::Generic(s),
        }
    }
}

impl From<VisSubtractError> for HyperdriveError {
    fn from(e: VisSubtractError) -> Self {
        let s = e.to_string();
        match e {
            VisSubtractError::MissingSource { .. }
            | VisSubtractError::NoSourcesAfterVeto
            | VisSubtractError::NoSources
            | VisSubtractError::AllSourcesFiltered
            | VisSubtractError::NoTimesteps
            | VisSubtractError::DuplicateTimesteps
            | VisSubtractError::UnavailableTimestep { .. }
            | VisSubtractError::NoInputData
            | VisSubtractError::InvalidDataInput(_)
            | VisSubtractError::BadArrayPosition { .. }
            | VisSubtractError::MultipleMetafits(_)
            | VisSubtractError::MultipleMeasurementSets(_)
            | VisSubtractError::MultipleUvfits(_) => Self::VisSubtract(s),
            VisSubtractError::NoDelays | VisSubtractError::BadDelays => Self::Delays(s),
            VisSubtractError::VisWrite(_) | VisSubtractError::InvalidOutputFormat(_) => {
                Self::VisWrite(s)
            }
            VisSubtractError::VisRead(e) => Self::from(e),
            VisSubtractError::SourceList(_) | VisSubtractError::Veto(_) => Self::Srclist(s),
            VisSubtractError::Beam(_) => Self::Beam(s),
            VisSubtractError::ParseOutputVisTimeAverageFactor(_)
            | VisSubtractError::ParseOutputVisFreqAverageFactor(_)
            | VisSubtractError::OutputVisTimeFactorNotInteger
            | VisSubtractError::OutputVisFreqFactorNotInteger
            | VisSubtractError::OutputVisTimeAverageFactorZero
            | VisSubtractError::OutputVisFreqAverageFactorZero
            | VisSubtractError::OutputVisTimeResNotMultiple { .. }
            | VisSubtractError::OutputVisFreqResNotMultiple { .. } => Self::Averaging(s),
            VisSubtractError::Glob(_)
            | VisSubtractError::FileWrite(_)
            | VisSubtractError::IO(_) => Self::Generic(s),
            #[cfg(feature = "cuda")]
            VisSubtractError::CudaError(_) => Self::Generic(s),
        }
    }
}

// Library code errors.

impl From<SrclistError> for HyperdriveError {
    fn from(e: SrclistError) -> Self {
        let s = e.to_string();
        match e {
            SrclistError::NoSourcesAfterVeto
            | SrclistError::ReadSourceList(_)
            | SrclistError::WriteSourceList(_)
            | SrclistError::Veto(_) => Self::Srclist(s),
            SrclistError::MissingMetafits => Self::Metafits(s),
            SrclistError::Beam(_) => Self::Beam(s),
            SrclistError::Mwalib(_) => Self::Mwalib(s),
            SrclistError::IO(_) => Self::Generic(s),
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
            SolutionsReadError::IO(_) => Self::Generic(s),
        }
    }
}

impl From<SolutionsWriteError> for HyperdriveError {
    fn from(e: SolutionsWriteError) -> Self {
        let s = e.to_string();
        match e {
            SolutionsWriteError::UnsupportedExt { .. } => Self::Solutions(s),
            SolutionsWriteError::Fits(_) | SolutionsWriteError::Fitsio(_) => Self::Cfitsio(s),
            SolutionsWriteError::IO(_) => Self::Generic(s),
        }
    }
}

impl From<InputFileError> for HyperdriveError {
    fn from(e: InputFileError) -> Self {
        let s = e.to_string();
        match e {
            InputFileError::PpdMetafitsUnsupported(_) => Self::Metafits(s),
            InputFileError::NotRecognised(_)
            | InputFileError::DoesNotExist(_)
            | InputFileError::CouldNotRead(_)
            | InputFileError::Glob(_)
            | InputFileError::IO(_, _) => Self::VisRead(s),
        }
    }
}

impl From<VisReadError> for HyperdriveError {
    fn from(e: VisReadError) -> Self {
        let s = e.to_string();
        match e {
            VisReadError::InputFile(e) => Self::from(e),
            VisReadError::Raw(_)
            | VisReadError::Birli(_)
            | VisReadError::MS(_)
            | VisReadError::Uvfits(_) => Self::VisRead(s),
            VisReadError::MwafFlagsMissingForTimestep { .. } => Self::Mwaf(s),
            VisReadError::BadArraySize { .. } | VisReadError::SelectionError(_) => Self::Generic(s),
        }
    }
}

impl From<DiCalArgsError> for HyperdriveError {
    fn from(e: DiCalArgsError) -> Self {
        let s = e.to_string();
        match e {
            DiCalArgsError::NoInputData
            | DiCalArgsError::NoOutput
            | DiCalArgsError::NoTiles
            | DiCalArgsError::NoChannels
            | DiCalArgsError::NoTimesteps
            | DiCalArgsError::UnavailableTimestep { .. }
            | DiCalArgsError::DuplicateTimesteps
            | DiCalArgsError::TileFlag(_)
            | DiCalArgsError::NoSources
            | DiCalArgsError::BadArrayPosition { .. }
            | DiCalArgsError::ParseUvwMin(_)
            | DiCalArgsError::ParseUvwMax(_) => Self::DiCalibrate(s),
            DiCalArgsError::NoSourceList
            | DiCalArgsError::NoSourcesAfterVeto
            | DiCalArgsError::Veto(_)
            | DiCalArgsError::SourceList(_) => Self::Srclist(s),
            DiCalArgsError::NoDelays | DiCalArgsError::BadDelays => Self::Delays(s),
            DiCalArgsError::CalibrationOutputFile { .. } => Self::Solutions(s),
            DiCalArgsError::ParsePfbFlavour(_) => Self::RawDataCorrections(s),
            DiCalArgsError::Beam(_) => Self::Beam(s),
            DiCalArgsError::ParseCalTimeAverageFactor(_)
            | DiCalArgsError::ParseCalFreqAverageFactor(_)
            | DiCalArgsError::CalTimeFactorNotInteger
            | DiCalArgsError::CalFreqFactorNotInteger
            | DiCalArgsError::CalTimeResNotMultiple { .. }
            | DiCalArgsError::CalFreqResNotMultiple { .. }
            | DiCalArgsError::CalTimeFactorZero
            | DiCalArgsError::CalFreqFactorZero
            | DiCalArgsError::ParseOutputVisTimeAverageFactor(_)
            | DiCalArgsError::ParseOutputVisFreqAverageFactor(_)
            | DiCalArgsError::OutputVisTimeFactorNotInteger
            | DiCalArgsError::OutputVisFreqFactorNotInteger
            | DiCalArgsError::OutputVisTimeAverageFactorZero
            | DiCalArgsError::OutputVisFreqAverageFactorZero
            | DiCalArgsError::OutputVisTimeResNotMultiple { .. }
            | DiCalArgsError::OutputVisFreqResNotMultiple { .. } => Self::Averaging(s),
            DiCalArgsError::InvalidDataInput(_)
            | DiCalArgsError::MultipleMetafits(_)
            | DiCalArgsError::MultipleMeasurementSets(_)
            | DiCalArgsError::MultipleUvfits(_) => Self::VisRead(s),
            DiCalArgsError::VisRead(e) => Self::from(e),
            DiCalArgsError::VisFileType { .. } | DiCalArgsError::FileWrite(_) => Self::VisWrite(s),
            DiCalArgsError::UnrecognisedArgFileExt(_)
            | DiCalArgsError::TomlDecode { .. }
            | DiCalArgsError::JsonDecode { .. }
            | DiCalArgsError::Glob(_)
            | DiCalArgsError::IO(_) => Self::Generic(s),
            #[cfg(feature = "cuda")]
            DiCalArgsError::CudaError(_) => Self::Generic(s),
        }
    }
}
