// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Code to handle reading from and writing to various data container formats.
 */

mod error;
pub(crate) mod ms;
// pub(crate) mod raw;

pub(crate) use error::ReadInputDataError;
pub(crate) use ms::MS;
// pub(crate) use raw::RawData;

use std::collections::{HashMap, HashSet};

use ndarray::prelude::*;
use num::{Complex, Float};

use crate::context::{FreqContext, ObsContext};
use mwa_hyperdrive_core::{c32, c64, InstrumentalStokes, UVW};

pub(crate) trait InputData: Sync + Send {
    fn get_obs_context(&self) -> &ObsContext;

    fn get_freq_context(&self) -> &FreqContext;

    /// Read all frequencies and baselines for a single timestep into the
    /// `data_array` (similarly for the weights), and return the [UVW]
    /// coordinates associated with the baselines.
    fn read(
        &self,
        data_array: ArrayViewMut2<Vis<f32>>,
        timestep: usize,
        baseline_map: &HashMap<(usize, usize), usize>,
        flagged_fine_chans: &HashSet<usize>,
    ) -> Result<(Vec<UVW>, Array2<f32>), ReadInputDataError>;
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Vis<F: Float> {
    /// XX
    pub(crate) xx: Complex<F>,
    /// XY
    pub(crate) xy: Complex<F>,
    /// YX
    pub(crate) yx: Complex<F>,
    /// YY
    pub(crate) yy: Complex<F>,
}

impl<F: Float> Vis<F> {
    #[inline]
    pub(crate) fn from_fd_and_phase(fd: InstrumentalStokes, phase: c64) -> Self {
        let product = fd * phase;
        Self {
            xx: Complex::new(
                F::from(product.xx.re).unwrap(),
                F::from(product.xx.im).unwrap(),
            ),
            xy: Complex::new(
                F::from(product.xy.re).unwrap(),
                F::from(product.xy.im).unwrap(),
            ),
            yx: Complex::new(
                F::from(product.yx.re).unwrap(),
                F::from(product.yx.im).unwrap(),
            ),
            yy: Complex::new(
                F::from(product.yy.re).unwrap(),
                F::from(product.yy.im).unwrap(),
            ),
        }
    }
}

impl From<[c32; 4]> for Vis<f32> {
    fn from(arr: [c32; 4]) -> Self {
        Self {
            xx: arr[0],
            xy: arr[1],
            yx: arr[2],
            yy: arr[3],
        }
    }
}

impl<F: Float> std::ops::AddAssign<Vis<F>> for Vis<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.xx = self.xx + rhs.xx;
        self.xy = self.xy + rhs.xy;
        self.yx = self.yx + rhs.yx;
        self.yy = self.yy + rhs.yy;
    }
}

impl<F: Float> std::ops::Mul<F> for Vis<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        let mut out = self.clone();
        out.xx = self.xx * rhs;
        out.xy = self.xy * rhs;
        out.yx = self.yx * rhs;
        out.yy = self.yy * rhs;
        out
    }
}
