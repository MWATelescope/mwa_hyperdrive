use mwa_hyperdrive_common::thiserror::Error;

#[derive(Error, Debug)]
pub enum ObsContextError {
    #[error("Array position partially specified. lat={lat:?}, lng={lng:?}")]
    PartialPosition { lat: Option<f64>, lng: Option<f64> },
}
