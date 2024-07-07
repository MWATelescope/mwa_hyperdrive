// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

use crate::{
    beam::BeamError, io::GlobError, srclist::HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED,
};

/// Errors associated with reading in any kind of source list.
#[derive(Error, Debug)]
pub(crate) enum ReadSourceListError {
    #[error(
        "Source list error: Attempted to use RA {0}°, but this is out of range (0° <= RA < 360°)"
    )]
    InvalidRa(f64),

    #[error("Source list error: Attempted to use HA {0}, but this is out of range (0 < HA < 24)")]
    InvalidHa(f64),

    #[error(
        "Source list error: Attempted to use Dec {0}°, but this is out of range (-90° <= Dec <= 90°)"
    )]
    InvalidDec(f64),

    #[error("Source {source_name}: The sum of all Stokes {stokes_comp} flux densities was negative ({sum})")]
    InvalidFluxDensitySum {
        sum: f64,
        stokes_comp: &'static str,
        source_name: String,
    },

    #[error("Source {source_name}: A component contains NaNs for its flux densities. This is not allowed.")]
    NaNsInComponent { source_name: String },

    #[error("Could not interpret the contents of the source list. Specify which style source list it is, and a more specific error can be shown.")]
    FailedToReadAsAnyType,

    #[error("Could not deserialise the contents as yaml or json.\n\nyaml error: {yaml_err}\n\njson error: {json_err}")]
    FailedToDeserialise { yaml_err: String, json_err: String },

    #[error("No sky-model source list file supplied")]
    NoSourceList,

    #[error(transparent)]
    Glob(#[from] GlobError),

    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("Tried to use {requested} sources, but only {available} sources were available after vetoing")]
    VetoTooFewSources { requested: usize, available: usize },

    #[error("Beam error when trying to veto the source list: {0}")]
    Beam(#[from] BeamError),

    #[error(transparent)]
    Common(#[from] ReadSourceListCommonError),

    #[error(transparent)]
    Rts(#[from] ReadSourceListRtsError),

    #[error(transparent)]
    Woden(#[from] ReadSourceListWodenError),

    #[error(transparent)]
    AO(#[from] ReadSourceListAOError),

    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Sexagesimal(#[from] marlu::sexagesimal::SexagesimalError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}

/// Errors associated with reading in an RTS or WODEN source list.
#[derive(Error, Debug, PartialEq)]
pub(crate) enum ReadSourceListCommonError {
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

    #[error("Source list line {line_num}: Found {keyword} outside of a SOURCE context")]
    OutsideSource {
        line_num: u32,
        keyword: &'static str,
    },

    #[error("Source list line {line_num}: Found {keyword} outside of a COMPONENT context")]
    OutsideComponent {
        line_num: u32,
        keyword: &'static str,
    },

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

    #[error("Source list line {0}: Incomplete LINEAR")]
    IncompleteLinearLine(u32),

    #[error("Source list line {0}: Found an additional component type when one already exists")]
    MultipleComponentTypes(u32),

    #[error("Source list line {0}: No sources found")]
    NoSources(u32),

    #[error("Source list line {0}: ENDSOURCE reached, but no components found")]
    NoComponents(u32),

    #[error("Source list line {0}: Source has no non SHAPELET components, which can't be used (SHAPELET2 is OK)")]
    NoNonShapeletComponents(u32),

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
pub(crate) enum ReadSourceListRtsError {
    #[error(transparent)]
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

    #[error("Source list line {line_num}: {comp_type} component at RA {ra}, Dec {dec} did not contain any flux densities")]
    MissingFluxes {
        line_num: u32,
        comp_type: &'static str,
        ra: f64,
        dec: f64,
    },
}

/// Errors associated with reading in a WODEN source list.
#[derive(Error, Debug, PartialEq, Eq)]
pub(crate) enum ReadSourceListWodenError {
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
        comp_type: &'static str,
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

/// Errors associated with reading in a AO source list.
#[derive(Error, Debug, PartialEq)]
pub(crate) enum ReadSourceListAOError {
    #[error("Source list line {0}: Source name not in quotes")]
    NameNotQuoted(u32),

    #[error("Source list line {line_num}: Unrecognised component type: {comp_type}")]
    UnrecognisedComponentType { line_num: u32, comp_type: String },

    #[error("Source list line {0}: Missing component type")]
    MissingComponentType(u32),

    #[error("Source list line {0}: Source has no components")]
    MissingComponents(u32),

    #[error("Source list line {line_num}: {comp_type} component at RA {ra}, Dec {dec} did not contain any flux densities")]
    MissingFluxes {
        line_num: u32,
        comp_type: &'static str,
        ra: f64,
        dec: f64,
    },

    #[error("Source list line {0}: Incomplete position")]
    IncompletePositionLine(u32),

    #[error("Source list line {0}: Incomplete shape")]
    IncompleteShapeLine(u32),

    #[error("Source list line {0}: Incomplete frequency")]
    IncompleteFrequencyLine(u32),

    #[error("Source list line {line_num}: Unhandled frequency units: {units}")]
    UnhandledFrequencyUnits { line_num: u32, units: String },

    #[error("Source list line {0}: Incomplete fluxdensity")]
    IncompleteFluxDensityLine(u32),

    #[error("Source list line {line_num}: Unhandled flux density units: {units}")]
    UnhandledFluxDensityUnits { line_num: u32, units: String },

    #[error("Source list line {0}: Incomplete spectral-index")]
    IncompleteSpectralIndexLine(u32),

    #[error("Source list line {0}: Missing opening curly brace")]
    MissingOpeningCurly(u32),

    #[error("Source list line {0}: Missing closing curly brace")]
    MissingClosingCurly(u32),

    #[error("Source list line {0}: Found sed, but did not find its end")]
    MissingEndSed(u32),
}

/// Errors associated with writing out a source list.
#[derive(Error, Debug)]
pub(crate) enum WriteSourceListError {
    #[error("Source list type {source_list_type} does not support {comp_type} components")]
    UnsupportedComponentType {
        source_list_type: &'static str,
        comp_type: &'static str,
    },

    #[error("Source list type {source_list_type} does not support {fd_type} flux densities")]
    UnsupportedFluxDensityType {
        source_list_type: &'static str,
        fd_type: &'static str,
    },

    #[error("'{0}' is an invalid file type for a hyperdrive-style source list; must have one of the following extensions: {}", *HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED)]
    InvalidHyperdriveFormat(String),

    #[error(transparent)]
    Sexagesimal(#[from] marlu::sexagesimal::SexagesimalError),

    #[error(transparent)]
    Fitsio(#[from] fitsio::errors::Error),

    #[error(transparent)]
    Fits(#[from] crate::io::read::fits::FitsError),

    /// An IO error.
    #[error(transparent)]
    IO(#[from] std::io::Error),

    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Error, Debug)]
pub(crate) enum SrclistError {
    #[error("Source list error: Need a metafits file to perform work, but none was supplied")]
    MissingMetafits,

    #[error(transparent)]
    ReadSourceList(#[from] ReadSourceListError),

    #[error(transparent)]
    WriteSourceList(#[from] WriteSourceListError),

    #[error(transparent)]
    Beam(#[from] crate::beam::BeamError),

    #[error(transparent)]
    Mwalib(#[from] mwalib::MwalibError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
