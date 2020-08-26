// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle the pointing centre coordinates.
*/

use super::hadec::HADec;

/// Pointing centre coordinates. Pretty much just a `HADec` struct with a LST.
#[derive(Clone, Copy, Debug)]
pub struct PointingCentre {
    /// Local sidereal time [radians]
    pub lst: f64,
    /// Hour angle and Declination coordinates.
    pub hadec: HADec,
}

impl PointingCentre {
    /// Generate a `PointingCentre` using an hour angle `ha` and declination
    /// `dec`. All arguments have units of radians.
    ///
    /// As the pointing centre struct saves sine and cosine values, this `new`
    /// function exists to ease reduce programmer effort.
    pub fn new_from_ha(lst: f64, ha: f64, dec: f64) -> Self {
        Self {
            lst,
            hadec: HADec::new(ha, dec),
        }
    }

    /// Generate a `PointingCentre` using an hour angle `ha` and declination `dec`. All
    /// arguments have units of radians.
    ///
    /// As the pointing centre struct saves sine and cosine values, this `new`
    /// function exists to ease reduce programmer effort.
    pub fn new_from_hadec(lst: f64, hadec: HADec) -> Self {
        Self { lst, hadec }
    }

    /// Similar to `PointingCentre::new_from_ha`, but takes a right ascension
    /// `ra` instead of an hour angle. All arguments have units of radians.
    pub fn new_from_ra(lst: f64, ra: f64, dec: f64) -> Self {
        let ha = lst - ra;
        Self {
            lst,
            hadec: HADec::new(ha, dec),
        }
    }

    /// Given a new LST, update self.
    pub fn update(&mut self, lst: f64) {
        self.hadec.ha += lst - self.lst;
        self.lst = lst;
    }
}
