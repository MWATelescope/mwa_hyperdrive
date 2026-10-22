// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Tile metadata for grouping plots by receiver and cable flavour.
//!
//! Crosstalk on the MWA is mostly between tiles that share a receiver, and a
//! tile's signal-path gain depends on its cable flavour, so per-tile plots are
//! ordered by receiver, then flavour, then antenna number, and coloured by
//! flavour.

use std::collections::HashMap;
use std::str::FromStr;

use rizzma::artist::Rgba;

/// Colours assigned to cable flavours in sorted-name order, so that a flavour
/// keeps its colour between observations and plots. These match mwa_qa's
/// `plot_prepvis_groups.py`.
const FLAVOUR_COLOURS: [&str; 8] = [
    "#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
];

/// One tile's metadata, from the metafits.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TileInfo {
    pub(crate) name: String,
    /// The metafits antenna number (TILEDATA Antenna).
    pub(crate) ant: u32,
    /// The receiver number (TILEDATA Rx).
    pub(crate) rx: u32,
    /// The cable flavour (TILEDATA Flavors), e.g. "RG6_90".
    pub(crate) flavour: String,
}

/// How tiles are ordered and banded on per-tile plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GroupBy {
    /// Bands by receiver; within a receiver, tiles are sorted by flavour.
    #[default]
    ReceiverFlavour,
    /// Bands by receiver; within a receiver, tiles are sorted by antenna.
    Receiver,
    /// Bands by cable flavour.
    Flavour,
}

impl FromStr for GroupBy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "receiver+flavour" | "receiver+flavor" | "rx+flavour" | "rx+flavor" => {
                Ok(Self::ReceiverFlavour)
            }
            "receiver" | "rx" => Ok(Self::Receiver),
            "flavour" | "flavor" => Ok(Self::Flavour),
            _ => Err(format!(
                "unknown grouping '{s}'; expected one of receiver+flavour, receiver, flavour"
            )),
        }
    }
}

/// A run of consecutive tiles (in plotting order) that share a group.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroupSpan {
    pub(crate) label: String,
    /// Index of the first tile of the group, in plotting order.
    pub(crate) start: usize,
    /// One past the last tile of the group.
    pub(crate) end: usize,
}

/// All tiles of an array, with lookups by name and a plotting order.
#[derive(Debug, Clone)]
pub(crate) struct Tiles {
    pub(crate) tiles: Vec<TileInfo>,
    by_name: HashMap<String, usize>,
    flavour_colours: HashMap<String, Rgba>,
}

impl Tiles {
    pub(crate) fn new(tiles: Vec<TileInfo>) -> Tiles {
        let by_name = tiles
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let mut flavours: Vec<&str> = tiles.iter().map(|t| t.flavour.as_str()).collect();
        flavours.sort_unstable();
        flavours.dedup();
        let flavour_colours = flavours
            .into_iter()
            .enumerate()
            .map(|(i, f)| {
                let hex = FLAVOUR_COLOURS[i % FLAVOUR_COLOURS.len()];
                (
                    f.to_string(),
                    Rgba::from_hex(hex).expect("the flavour colours are valid hex"),
                )
            })
            .collect();
        Tiles {
            tiles,
            by_name,
            flavour_colours,
        }
    }

    /// Tile metadata from a metafits. The X-pol RF input supplies the receiver
    /// and flavour; mwalib already matches TILEDATA rows to antennas by
    /// antenna number, not by row position.
    pub(crate) fn from_metafits(context: &mwalib::MetafitsContext) -> Tiles {
        Tiles::new(
            context
                .antennas
                .iter()
                .map(|a| TileInfo {
                    name: a.tile_name.trim().to_string(),
                    ant: a.ant,
                    rx: a.rfinput_x.rec_number,
                    flavour: a.rfinput_x.flavour.trim().to_string(),
                })
                .collect(),
        )
    }

    pub(crate) fn len(&self) -> usize {
        self.tiles.len()
    }

    pub(crate) fn index_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }

    /// The colour of a cable flavour. Unknown flavours are grey.
    pub(crate) fn flavour_colour(&self, flavour: &str) -> Rgba {
        self.flavour_colours
            .get(flavour)
            .copied()
            .unwrap_or(Rgba::new(0.5, 0.5, 0.5, 1.0))
    }

    /// The colour of a tile's cable flavour.
    pub(crate) fn tile_colour(&self, tile: usize) -> Rgba {
        self.flavour_colour(&self.tiles[tile].flavour)
    }

    /// The flavours present, sorted by name, with their colours.
    pub(crate) fn flavour_key(&self) -> Vec<(Rgba, String)> {
        let mut key: Vec<(Rgba, String)> = self
            .flavour_colours
            .iter()
            .map(|(f, c)| (*c, f.clone()))
            .collect();
        key.sort_by(|a, b| a.1.cmp(&b.1));
        key
    }

    /// The indices of the tiles that `include` accepts, in plotting order, and
    /// the group spans along that order. Leaving out tiles with nothing to
    /// show stops them taking up rows.
    pub(crate) fn order(
        &self,
        group_by: GroupBy,
        include: impl Fn(usize) -> bool,
    ) -> (Vec<usize>, Vec<GroupSpan>) {
        let mut order: Vec<usize> = (0..self.tiles.len()).filter(|&i| include(i)).collect();
        order.sort_by(|&a, &b| {
            let (a, b) = (&self.tiles[a], &self.tiles[b]);
            match group_by {
                GroupBy::ReceiverFlavour => {
                    (a.rx, &a.flavour, a.ant).cmp(&(b.rx, &b.flavour, b.ant))
                }
                GroupBy::Receiver => (a.rx, a.ant).cmp(&(b.rx, b.ant)),
                GroupBy::Flavour => (&a.flavour, a.ant).cmp(&(&b.flavour, b.ant)),
            }
        });
        let label = |i: usize| match group_by {
            GroupBy::ReceiverFlavour | GroupBy::Receiver => format!("Rx{:02}", self.tiles[i].rx),
            GroupBy::Flavour => self.tiles[i].flavour.clone(),
        };
        let mut spans: Vec<GroupSpan> = vec![];
        for (pos, &i) in order.iter().enumerate() {
            let l = label(i);
            match spans.last_mut() {
                Some(span) if span.label == l => span.end = pos + 1,
                _ => spans.push(GroupSpan {
                    label: l,
                    start: pos,
                    end: pos + 1,
                }),
            }
        }
        (order, spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(name: &str, ant: u32, rx: u32, flavour: &str) -> TileInfo {
        TileInfo {
            name: name.to_string(),
            ant,
            rx,
            flavour: flavour.to_string(),
        }
    }

    fn tiles() -> Tiles {
        Tiles::new(vec![
            tile("Tile011", 0, 2, "RG6_90"),
            tile("Tile012", 1, 1, "RG6_150"),
            tile("Tile013", 2, 1, "RG6_90"),
            tile("Tile014", 3, 2, "RG6_150"),
            tile("Tile015", 4, 1, "RG6_150"),
        ])
    }

    #[test]
    fn receiver_flavour_order_bands_by_receiver() {
        let (order, spans) = tiles().order(GroupBy::ReceiverFlavour, |_| true);
        // Rx1: RG6_150 (ants 1, 4) then RG6_90 (ant 2); Rx2: RG6_150 (3) then RG6_90 (0).
        assert_eq!(order, vec![1, 4, 2, 3, 0]);
        assert_eq!(
            spans,
            vec![
                GroupSpan {
                    label: "Rx01".to_string(),
                    start: 0,
                    end: 3
                },
                GroupSpan {
                    label: "Rx02".to_string(),
                    start: 3,
                    end: 5
                },
            ]
        );
    }

    #[test]
    fn receiver_order_ignores_flavour() {
        let (order, _) = tiles().order(GroupBy::Receiver, |_| true);
        assert_eq!(order, vec![1, 2, 4, 0, 3]);
    }

    #[test]
    fn flavour_order_bands_by_flavour() {
        let (order, spans) = tiles().order(GroupBy::Flavour, |_| true);
        assert_eq!(order, vec![1, 3, 4, 0, 2]);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, "RG6_150");
        assert_eq!((spans[1].start, spans[1].end), (3, 5));
    }

    #[test]
    fn order_skips_excluded_tiles() {
        let (order, spans) = tiles().order(GroupBy::ReceiverFlavour, |i| i != 4);
        assert_eq!(order, vec![1, 2, 3, 0]);
        assert_eq!((spans[0].start, spans[0].end), (0, 2));
        assert_eq!((spans[1].start, spans[1].end), (2, 4));
    }

    #[test]
    fn flavour_colours_follow_sorted_names() {
        let t = tiles();
        // RG6_150 sorts first, so takes the first colour.
        assert_eq!(
            t.flavour_colour("RG6_150"),
            Rgba::from_hex(FLAVOUR_COLOURS[0]).unwrap()
        );
        assert_eq!(
            t.flavour_colour("RG6_90"),
            Rgba::from_hex(FLAVOUR_COLOURS[1]).unwrap()
        );
        assert_eq!(t.tile_colour(0), t.flavour_colour("RG6_90"));
        let key = t.flavour_key();
        assert_eq!(key.len(), 2);
        assert_eq!(key[0].1, "RG6_150");
    }

    #[test]
    fn group_by_parses() {
        assert_eq!("receiver+flavour".parse(), Ok(GroupBy::ReceiverFlavour));
        assert_eq!("RX".parse(), Ok(GroupBy::Receiver));
        assert_eq!("flavor".parse(), Ok(GroupBy::Flavour));
        assert!("cable".parse::<GroupBy>().is_err());
    }
}
