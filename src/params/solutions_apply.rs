// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use super::{InputVisParams, OutputVisParams, VisConvertError, VisConvertParams};

pub(crate) struct SolutionsApplyParams {
    pub(crate) input_vis_params: InputVisParams,
    pub(crate) output_vis_params: OutputVisParams,
}

impl SolutionsApplyParams {
    pub(crate) fn run(&self) -> Result<(), VisConvertError> {
        let Self {
            input_vis_params,
            output_vis_params,
        } = self;

        assert!(
            input_vis_params.solutions.is_some(),
            "No calibration solutions are in the input vis params; this shouldn't be possible"
        );

        VisConvertParams::run_inner(input_vis_params, output_vis_params)
    }
}
