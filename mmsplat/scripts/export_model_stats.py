from typing import List, Optional, Tuple, Union, cast, Literal
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict
from typing_extensions import Annotated

import torch
import numpy as np
import os
import sys
import json
import tyro

from nerfstudio.scripts.exporter import Exporter, ExportGaussianSplat
from nerfstudio.models.splatfacto import RGB2SH, SH2RGB
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.data.scene_box import OrientedBox

from mmsplat.mmsplat_model import MMSplatModel
from mmsplat.mmsplat_datamanager import MMSplatDataManager
from mmsplat.util.eval_utils import eval_setup










@dataclass
class ExportMMSplatModelStats:
    """Export Camera positions from MMSplatModel to .json files"""

    load_config: Path
    """Path to the config YAML file."""
    output_file: Path
    """Path to the output json file."""
    load_step: Optional[int] = None
    """Step for which to load the checkpoint, or None if last checkpoint should be loaded"""


    @staticmethod
    def export_model_stats(pipeline, out_file: Path) -> None:
        """Generate output .json files"""

        # get stats
        stats_json = pipeline.model.get_model_stats()

        # export
        with out_file.open("w") as json_file:
            json.dump(stats_json, json_file, indent=4)





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

        ExportMMSplatModelStats.export_model_stats(pipeline, out_file)








# define commands
Commands = Union[
    Annotated[ExportMMSplatModelStats, tyro.conf.subcommand(name="mmsplat")],
]


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")

    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + str(e))

if __name__ == "__main__":
    entrypoint()