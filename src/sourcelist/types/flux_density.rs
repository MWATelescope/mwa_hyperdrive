// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#[derive(Clone, Copy, Debug)]
pub struct FluxDensity {
    /// The frequency that these flux densities apply to [MHz]
    pub freq: f64,
    /// The flux density of Stokes I [Jy]
    pub i: f64,
    /// The flux density of Stokes Q [Jy]
    pub q: f64,
    /// The flux density of Stokes U [Jy]
    pub u: f64,
    /// The flux density of Stokes V [Jy]
    pub v: f64,
}

impl std::ops::Mul<f64> for FluxDensity {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        FluxDensity {
            freq: self.freq,
            i: self.i * rhs,
            q: self.q * rhs,
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}
