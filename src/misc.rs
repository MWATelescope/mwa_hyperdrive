// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Miscellaneous things.

use is_terminal::IsTerminal;

pub(crate) fn is_a_tty() -> bool {
    std::io::stdout().is_terminal() || std::io::stderr().is_terminal()
}
