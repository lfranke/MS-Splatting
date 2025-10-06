import sys
from typing import Union
from typing_extensions import Annotated
import tyro
from nerfstudio.utils.rich_utils import CONSOLE, status

from mmsplat.mmsplat_camera_export import ExportMMSplatCameras


# define commands
Commands = Union[
    Annotated[ExportMMSplatCameras, tyro.conf.subcommand(name="mmsplat")],
]


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")

    try:
        tyro.cli(Commands).main()
    except RuntimeError as e:
        CONSOLE.log("[bold red]" + str(e))

if __name__ == "__main__":
    entrypoint()