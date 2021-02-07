// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Handle pointing centre coordinates.
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
    pub fn new_from_ha(lst_rad: f64, ha_rad: f64, dec_rad: f64) -> Self {
        Self {
            lst: lst_rad,
            hadec: HADec::new(ha_rad, dec_rad),
        }
    }

    /// Generate a `PointingCentre` using an hour angle `ha` and declination `dec`. All
    /// arguments have units of radians.
    ///
    /// As the pointing centre struct saves sine and cosine values, this `new`
    /// function exists to ease reduce programmer effort.
    pub fn new_from_hadec(lst_rad: f64, hadec: HADec) -> Self {
        Self {
            lst: lst_rad,
            hadec,
        }
    }

    /// Similar to `PointingCentre::new_from_ha`, but takes a right ascension
    /// `ra` instead of an hour angle. All arguments have units of radians.
    pub fn new_from_ra(lst_rad: f64, ra_rad: f64, dec_rad: f64) -> Self {
        let ha = lst_rad - ra_rad;
        Self {
            lst: lst_rad,
            hadec: HADec::new(ha, dec_rad),
        }
    }

    /// Given a new LST, update self.
    pub fn update(&mut self, lst_rad: f64) {
        self.hadec.ha += lst_rad - self.lst;
        self.lst = lst_rad;
    }
}
