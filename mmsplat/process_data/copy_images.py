from typing import List, Literal, Optional, OrderedDict, Tuple, Union
import shutil

from pathlib import Path

from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command
from nerfstudio.process_data import process_data_utils

from mmsplat.util.parallel_for import parallel_for







def copy_images_list(
    image_paths: List[Path],
    image_dir: Path,
    num_downscales: int,
    colmap_image_dir: Optional[Path] = None,
    subfolder: Optional[str] = None,
    image_prefix: str = "frame_",
    verbose: bool = False,
    keep_image_dir: bool = False,
    max_threads: int = 16
) -> List[Path]:
    """Implementation of nerfstudio.process_data.process_data_utils, with reduced feature set

    Fixes a bug in the original implementation, where ffmpeg downscales were called on the original image to process it with the same scale.
    For some reason, ffmpeg doesnt complain most of the time, but once in a while it refuses, so i just re-wrote the implementation myself.
    Also, possibly improves performance by launching parallel threads
    """

    # downscale directory names
    if subfolder is None:
        downscale_dirs = [Path(image_dir)] + [Path(f"{image_dir}_{2**(i+1)}") for i in range(num_downscales)]
    else:
        downscale_dirs = [Path(image_dir / subfolder)] + [Path(f"{image_dir}_{2**(i+1)}/{subfolder}") for i in range(num_downscales)]

    colmap_img_dir = colmap_image_dir if ((colmap_image_dir is None) or (subfolder is None)) else (colmap_image_dir / subfolder)

    # clear/create downscale dirs
    for out_dir in downscale_dirs:
        if not keep_image_dir: # clear output folder if requested
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(exist_ok=True, parents=True)

    if colmap_img_dir is not None:
        if not keep_image_dir: # clear output folder if requested
            shutil.rmtree(colmap_img_dir, ignore_errors=True)
        colmap_img_dir.mkdir(exist_ok=True, parents=True)

    copied_image_paths = [None for _ in image_paths]

    # define function that copies downscales a single image
    def copy_image(index):
        image_source = Path(image_paths[index])
        frame_name = f"{image_prefix}{index + 1:05d}{image_source.suffix}"

        # set main output path
        copied_image_paths[index] = Path(downscale_dirs[0] / frame_name)

        # define ffmpeg chains
        downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}[out{i}]" for i in range(num_downscales + 1)]

        downscale_chain = (
            f"split={num_downscales+1}"
            + "".join([f"[t{i}]" for i in range(0, num_downscales + 1)])
            + ";"
            + ";".join(downscale_chains)
        )

        ffmpeg_cmd = f'ffmpeg -y -noautorotate -i "{image_source}" -filter_complex "[0:v]{downscale_chain}"' + "".join(
            [
                f' -map "[out{i}]" -q:v 2 "{downscale_dirs[i] / frame_name}"'
                for i in range(num_downscales + 1)
            ]
        )

        if verbose:
            CONSOLE.log(f"... {ffmpeg_cmd}")
        run_command(ffmpeg_cmd, verbose=verbose)

        if colmap_img_dir is not None:
            colmap_out_file = colmap_img_dir / frame_name
            ffmpeg_colmap_cmd = f'ffmpeg -y -noautorotate -i "{image_source}" -vf format=rgb8 "{colmap_out_file}"'

            if verbose:
                CONSOLE.log(f"... {ffmpeg_colmap_cmd}")
            run_command(ffmpeg_colmap_cmd, verbose=verbose)


    # run downscale for every image
    parallel_for(
        copy_image,
        list(range(len(image_paths))),
        1 if verbose else max_threads
    )

    CONSOLE.log(f"[bold green]:tada: Done downscaling images with prefix '{image_prefix}'.")

    return copied_image_paths








def old_copy_images_list(
    image_paths: List[Path],
    image_dir: Path,
    num_downscales: int,
    subfolder: Optional[str] = None,
    image_prefix: str = "frame_",
    verbose: bool = False,
    keep_image_dir: bool = False,
    max_threads: int = 16
) -> List[Path]:
    """Implementation of nerfstudio.process_data.process_data_utils, with reduced feature set

    Fixes a bug in the original implementation, where ffmpeg downscales were called on the original image to process it with the same scale.
    For some reason, ffmpeg doesnt complain most of the time, but once in a while it refuses, so i just re-wrote the implementation myself.
    Also, possibly improves performance by launching parallel threads
    """

    # downscale directory names
    if subfolder is None:
        downscale_dirs = [Path(f"{image_dir}_{2**(i+1)}") for i in range(num_downscales)]
    else:
        downscale_dirs = [Path(f"{image_dir}_{2**(i+1)}/{subfolder}") for i in range(num_downscales)]

    # clear/create downscale dirs
    for out_dir in downscale_dirs:
        if not keep_image_dir: # clear output folder if requested
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(exist_ok=True, parents=True)


    # first, copy original images via process_data_utils, as this does some other processing that we dont have to implement again
    copied_image_paths = process_data_utils.copy_images_list(
        image_paths=image_paths,
        image_dir=(image_dir if (subfolder is None) else (image_dir / subfolder)),
        num_downscales=0,
        image_prefix=image_prefix,
        verbose=verbose,
        keep_image_dir=keep_image_dir,
        same_dimensions=False
    )

    # exit early if no downscales are required
    if num_downscales <= 0:
        return copied_image_paths


    # define function that downscales a single image
    def downscale_image(index):
        copied_img_path = copied_image_paths[index]
        frame_name = copied_img_path.name

        downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}[out{i}]" for i in range(1, num_downscales + 1)]

        downscale_chain = (
            f"split={num_downscales}"
            + "".join([f"[t{i}]" for i in range(1, num_downscales + 1)])
            + ";"
            + ";".join(downscale_chains)
        )

        ffmpeg_cmd = f'ffmpeg -y -noautorotate -i "{copied_img_path}" -filter_complex "[0:v]{downscale_chain}"' + "".join(
            [
                f' -map "[out{i+1}]" -q:v 2 "{downscale_dirs[i] / frame_name}"'
                for i in range(num_downscales)
            ]
        )

        if verbose:
            CONSOLE.log(f"... {ffmpeg_cmd}")
        run_command(ffmpeg_cmd, verbose=verbose)

    # run downscale for every image
    parallel_for(
        downscale_image,
        list(range(len(copied_image_paths))),
        1 if verbose else max_threads
    )

    CONSOLE.log(f"[bold green]:tada: Done downscaling images with prefix '{image_prefix}'.")

    return copied_image_paths