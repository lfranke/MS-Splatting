from typing import List, Optional, Tuple, Union, cast, Literal
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import os
import json

from nerfstudio.scripts.exporter import Exporter, ExportGaussianSplat
from nerfstudio.models.splatfacto import RGB2SH, SH2RGB
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox

from mmsplat.mmsplat_model import MMSplatModel
from mmsplat.mmsplat_datamanager import MMSplatDataManager
from mmsplat.util.eval_utils import eval_setup










@dataclass
class ExportMMSplatCameras:
    """Export Camera positions from MMSplatModel to .json files"""

    load_config: Path
    """Path to the config YAML file."""
    output_file: Path
    """Path to the output json file."""
    load_step: Optional[int] = None
    """Step for which to load the checkpoint, or None if last checkpoint should be loaded"""


    @staticmethod
    def export_camera_transforms(pipeline, out_file: Path) -> None:
        """Generate output .json files"""

        # get all train cameras and transforms
        cam_list = pipeline.datamanager.all_train()
        cam_transforms = pipeline.model.get_optimized_cam_to_worlds(cam_list)

        # generate json structure
        res_arr = []
        for (cam, cam_dict), opt_transform in zip(cam_list, cam_transforms):
            res_arr.append({
                "name": cam_dict["name"],
                "mm_channel": cam_dict["mm_channel"],
                "cam_to_world": cam.camera_to_worlds.cpu().numpy().tolist(),
                "optimized_cam_to_world": opt_transform.cpu().numpy().tolist()
            })

        # export
        with out_file.open("w") as json_file:
            json.dump(res_arr, json_file, indent=4)





    def main(self) -> None:
        if self.output_file.suffix == "":
            out_file = self.output_file / "model_stats.json"
        else:
            out_file = self.output_file

        # ensure that output dir exists
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)

        # get pipeline and model
        _, pipeline, _, _ = eval_setup(self.load_config, self.load_step)

        assert isinstance(pipeline.model, MMSplatModel)
        assert isinstance(pipeline.datamanager, MMSplatDataManager)

        ExportMMSplatCameras.export_camera_transforms(pipeline, out_file)

