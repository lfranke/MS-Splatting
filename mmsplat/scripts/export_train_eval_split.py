import sys
from typing import Union
from typing_extensions import Annotated
import tyro
from nerfstudio.utils.rich_utils import CONSOLE, status

from mmsplat.mmsplat_dataparser import MMSplatDataParser



def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")

    try:
        tyro.cli(MMSplatDataParser._export_train_eval_split)
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + str(e))

if __name__ == "__main__":
    entrypoint()