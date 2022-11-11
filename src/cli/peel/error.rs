// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[derive(thiserror::Error, Debug)]
pub(crate) enum PeelArgsError {
    #[error("No calibration output was specified. There must be at least one calibration solution file.")]
    NoOutput,

    #[error("The data either contains no frequency channels or all channels are flagged")]
    NoChannels,

    #[error("The number of iono sub passes cannot be 0")]
    ZeroPasses,

    #[error("Error when parsing iono time average factor: {0}")]
    ParseIonoTimeAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing iono freq. average factor: {0}")]
    ParseIonoFreqAverageFactor(crate::unit_parsing::UnitParseError),

    #[error("Iono time average factor isn't an integer")]
    IonoTimeFactorNotInteger,

    #[error("Iono freq. average factor isn't an integer")]
    IonoFreqFactorNotInteger,

    #[error(
        "Iono time resolution isn't a multiple of input data's: {out} seconds vs {inp} seconds"
    )]
    IonoTimeResNotMultiple { out: f64, inp: f64 },

    #[error("Iono freq. resolution isn't a multiple of input data's: {out} Hz vs {inp} Hz")]
    IonoFreqResNotMultiple { out: f64, inp: f64 },

    #[error("Iono time average factor cannot be 0")]
    IonoTimeFactorZero,

    #[error("Iono freq. average factor cannot be 0")]
    IonoFreqFactorZero,

    #[error("Error when parsing minimum UVW cutoff: {0}")]
    ParseUvwMin(crate::unit_parsing::UnitParseError),

    #[error("Error when parsing maximum UVW cutoff: {0}")]
    ParseUvwMax(crate::unit_parsing::UnitParseError),

    #[error(transparent)]
    Glob(#[from] crate::io::GlobError),

    #[error(transparent)]
    VisRead(#[from] crate::io::read::VisReadError),

    #[error(transparent)]
    FileWrite(#[from] crate::io::write::FileWriteError),

    #[error("Error when trying to read source list: {0}")]
    SourceList(#[from] crate::srclist::ReadSourceListError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    Model(#[from] crate::model::ModelError),

    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[cfg(any(feature = "cuda", feature = "hip"))]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
}
