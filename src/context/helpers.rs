use mwa_hyperdrive_common::{
    hifitime::{Duration, Epoch, Unit},
    itertools::Itertools,
    lazy_static,
};

lazy_static::lazy_static! {
    static ref DURATION_MAX: Duration = Duration::from_f64(f64::MAX, Unit::Second);
}

pub fn guess_time_res(timestamps: Vec<Epoch>) -> Option<Duration> {
    timestamps
        .iter()
        .tuple_windows()
        .fold(None, |result, (&past, &future)| {
            Some(result.unwrap_or(*DURATION_MAX).min(future - past))
        })
}

pub fn guess_freq_res(frequencies: Vec<f64>) -> Option<f64> {
    frequencies
        .iter()
        .tuple_windows()
        .fold(None, |result, (&low, &high)| {
            Some(result.unwrap_or(f64::MAX).min(high - low))
        })
}
