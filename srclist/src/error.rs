// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

/// Errors associated with reading in any kind of source list.
#[derive(Error, Debug)]
pub enum ReadSourceListError {
    #[error("Attempted to use RA {0}, but this is out of range (0 < RA < 360)")]
    InvalidRa(f64),

    #[error("Attempted to use HA {0}, but this is out of range (0 < HA < 24)")]
    InvalidHa(f64),

    #[error("Attempted to use Dec {0}, but this is out of range (-90 < Dec < 90)")]
    InvalidDec(f64),

    #[error("Source {source_name}: The sum of all Stokes {stokes_comp} flux densities was negative ({sum})")]
    InvalidFluxDensitySum {
        sum: f64,
        stokes_comp: String,
        source_name: String,
    },

    #[error("{0}")]
    Common(#[from] ReadSourceListCommonError),

    #[error("{0}")]
    Rts(#[from] ReadSourceListRtsError),

    #[error("{0}")]
    Woden(#[from] ReadSourceListWodenError),

    #[error("{0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("{0}")]
    Json(#[from] serde_json::Error),
}

/// Errors associated with reading in an RTS or WODEN source list.
#[derive(Error, Debug, PartialEq)]
pub enum ReadSourceListCommonError {
    #[error("Source list line {line_num}: Unrecognised keyword {keyword}")]
    UnrecognisedKeyword { line_num: u32, keyword: String },

    #[error(
        "Source list line {0}: Found a SOURCE keyword while not finished reading previous SOURCE"
    )]
    NestedSources(u32),

    #[error(
        "Source list line {0}: Found a COMPONENT keyword while not finished reading previous COMPONENT"
    )]
    NestedComponents(u32),

    #[error("Source list line {0}: Found {keyword} outside of a SOURCE context")]
    OutsideSource { line_num: u32, keyword: String },

    #[error("Source list line {0}: Found {keyword} outside of a COMPONENT context")]
    OutsideComponent { line_num: u32, keyword: String },

    #[error("Source list line {0}: Found ENDSOURCE but we're not reading a source")]
    EarlyEndSource(u32),

    #[error("Source list line {0}: At the end of the file, but there's no ENDSOURCE")]
    MissingEndSource(u32),

    #[error("Source list line {0}: Found ENDSOURCE, but did not find ENDCOMPONENT")]
    MissingEndComponent(u32),

    #[error("Source list line {0}: Found ENDCOMPONENT but we're not reading a component")]
    EarlyEndComponent(u32),

    #[error("Source list line {0}: Incomplete SOURCE")]
    IncompleteSourceLine(u32),

    #[error("Source list line {0}: Incomplete FREQ")]
    IncompleteFluxLine(u32),

    #[error("Source list line {0}: Found an additional component type when one already exists")]
    MultipleComponentTypes(u32),

    #[error("Source list line {0}: No sources found")]
    NoSources(u32),

    #[error("Source list line {0}: ENDSOURCE reached, but no components found")]
    NoComponents(u32),

    #[error("Source list line {0}: A component did not have any flux density information")]
    NoFluxDensities(u32),

    #[error("Found {0} as a shapelet basis function, which is not an int")]
    ShapeletBasisNotInt(f64),

    /// Error when converting a string to a float.
    #[error("Source list line {line_num}: Error converting string {string} to a float")]
    ParseFloatError { line_num: u32, string: String },

    /// Error when converting a float into an int.
    #[error("Source list line {line_num}: Couldn't convert float {float} into an 8-bit int")]
    FloatToIntError { line_num: u32, float: f64 },
}

/// Errors associated with reading in an RTS source list.
#[derive(Error, Debug, PartialEq)]
pub enum ReadSourceListRtsError {
    #[error("{0}")]
    Common(#[from] ReadSourceListCommonError),

    #[error("Source list line {0}: Incomplete GAUSSIAN")]
    IncompleteGaussianLine(u32),

    #[error("Source list line {0}: Incomplete SHAPELET2")]
    IncompleteShapelet2Line(u32),

    #[error("Source list line {0}: Incomplete COEFF")]
    IncompleteCoeffLine(u32),

    #[error(
        "Source list line {0}: Found COEFF, but there was no SHAPELET or SHAPELET2 line above it"
    )]
    MissingShapeletLine(u32),

    #[error(
        "Source list line {0}: Tried to parse a shapelet component but there were no COEFF lines"
    )]
    NoShapeletCoeffs(u32),
}

/// Errors associated with reading in a WODEN source list.
#[derive(Error, Debug, PartialEq)]
pub enum ReadSourceListWodenError {
    #[error("Source list line {0}: Found SCOEFF, but there was no SPARAMS line above it")]
    MissingSParamsLine(u32),

    #[error("Source list line {0}: Incomplete GPARAMS")]
    IncompleteGParamsLine(u32),

    #[error("Source list line {0}: Incomplete SPARAMS")]
    IncompleteSParamsLine(u32),

    #[error("Source list line {0}: Incomplete SCOEFF")]
    IncompleteSCoeffLine(u32),

    /// Handle those details on WODEN source lines (e.g. "P 1 G 0 S 0 0")
    #[error(
        "Source list line {line_num}: Expected {expected} {comp_type} components, but found {got}"
    )]
    CompCountMismatch {
        line_num: u32,
        expected: u32,
        got: u32,
        comp_type: String,
    },

    #[error(
        "Source list line {line_num}: Expected {expected} shapelet components, but read {got}"
    )]
    ShapeletCoeffCountMismatch {
        line_num: u32,
        expected: u32,
        got: u32,
    },

    #[error(
        "Source list line {0}: Tried to parse a shapelet component but there were no SCOEFF lines"
    )]
    NoShapeletSCoeffs(u32),
}

/// Errors associated with writing an RTS or WODEN source list.
#[derive(Error, Debug)]
pub enum WriteSourceListError {
    #[error("Source {0} has no components")]
    NoComponents(String),

    #[error("Source list type {source_list_type} does not support {fd_type} flux densities")]
    UnsupportedFluxDensityType {
        source_list_type: String,
        fd_type: String,
    },

    #[error("{0}")]
    Estimate(#[from] mwa_hyperdrive_core::EstimateError),

    /// An IO error.
    #[error("{0}")]
    IO(#[from] std::io::Error),

    #[error("{0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("{0}")]
    Json(#[from] serde_json::Error),
}
