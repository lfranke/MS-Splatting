from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Dict, Tuple, Optional

from nerfstudio.utils.scripts import run_command
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.process_data import colmap_utils

import json
import pycolmap
import sys
import shutil
import copy

import mmsplat.process_data.mmutils as mmutils




def feature_extractor(
    database_path: Path,
    image_dir: Path,
    single_camera_per_folder: bool = False,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> bool:
    """Run colmap feature_extractor on all images in image_dir."""

    max_num_features = 12000  # Original: 8192
    max_image_size = 4000  # Original: 3200

    cmd = (f'{colmap_cmd} feature_extractor --database_path "{database_path}" '
           f'--image_path "{image_dir}" '
           # f'--SiftExtraction.max_image_size {int(max_image_size)} '
           f'--SiftExtraction.max_num_features {int(max_num_features)} ')

    if single_camera_per_folder:
        cmd += f'--ImageReader.single_camera_per_folder {int(single_camera_per_folder)} '

    run_command(cmd, verbose)
    
    return True




def exhaustive_matcher(
    database_path: Path,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> bool:
    """Run colmap exhaustive_matcher on all images in database."""

    cmd = f'{colmap_cmd} exhaustive_matcher --database_path "{database_path}"'
    run_command(cmd, verbose)
    
    return True





def run_reconstruction(
    database_path: Path,
    image_dir: Path,
    output_dir: Path,
    imagelist_path: Optional[Path] = None,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> bool:
    """Run colmap mapper on single image list."""

    # ensure that output dir exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # compose command
    cmd = f'{colmap_cmd} mapper --database_path "{database_path}" --image_path "{image_dir}" --output_path "{output_dir}" '

    if imagelist_path is not None:
        cmd += f'--image_list_path "{imagelist_path}"'

    # run command
    run_command(cmd, verbose)

    return True




def get_reconstruction_image_count(
    reconstruction_path: Path
) -> int:
    reconstruction = pycolmap.Reconstruction(str(reconstruction_path))
    return reconstruction.num_images()



def find_best_reconstruction(
    reconstruction_path: Path
) -> Tuple[Path, int]:
    """Given a reconstruction output folder, find the subfolder (e.g. 0, 1, ...) with the most registered images"""

    # remember best reconstruction
    best_rec_path = None
    best_rec_count = 0

    # iterate all subfolders
    for rpath in (f for f in reconstruction_path.iterdir() if f.is_dir()):
        reconstruction = pycolmap.Reconstruction(str(rpath))

        if reconstruction.num_images() > best_rec_count:
            best_rec_path = rpath
            best_rec_count = reconstruction.num_images()

    # return best reconstruction
    return best_rec_path, best_rec_count




def merge_consecutive(
    output_dir: Path,
    intermediate_dir: Optional[Path],
    reconstruction_paths: List[Path],
    reconstruction_names: Optional[List[str]] = None,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> bool:
    """Given a list of reconstructions (reconstruction_paths), attempt to consecutively merge them together via colmap model_merger.
    
    Arguments:
        output_dir: path to the final model, if successful
        intermediate_dir: path to a folder where the intermediate models can be stored. Ignored if no intermediate models are generated.
        reconstruction_names: List of descriptive names for each reconstruction. If set, will be used to name the folders of intermediate models. If not, "0_1", "0_1_2", ... will be used.
    """

    # check that models are present to merge
    if len(reconstruction_paths) < 2:
        CONSOLE.log(f"[bold red] merge_consecutive: At least two input reconstructions necessary.")
        return False
    
    # if necessary, check that intermediate_dir is assigned
    if len(reconstruction_paths) > 2 and intermediate_dir is None:
        CONSOLE.log(f"[bold red] merge_consecutive: intermediate_dir is needed, but was None.")
        return False
    
    # check reconstruction_names
    if reconstruction_names is None or len(reconstruction_names) < len(reconstruction_paths):
        # no (valid) reconstruction names specified, so choose indices 0, 1, 2, ...
        reconstruction_names = [str(i) for i in range(len(reconstruction_paths))]


    # remember last reconstruction (start with first in list)
    last_rec_path = reconstruction_paths[0]
    last_rec_name = reconstruction_names[0]

    # iterate reconstructions and merge consecutively
    for r_id in range(1, len(reconstruction_paths)):
        # collect variables
        r_path = reconstruction_paths[r_id]
        r_name = reconstruction_names[r_id]
        merge_name = last_rec_name + "_" + r_name

        # determine merge output path
        if r_id == len(reconstruction_paths) - 1:
            merge_path = output_dir
        else:
            merge_path = intermediate_dir / merge_name

        # create merge output path
        merge_path.mkdir(exist_ok=True, parents=True)
        
        # run colmap merge
        cmd = f'{colmap_cmd} model_merger --input_path1 "{r_path}" --input_path2 "{last_rec_path}" --output_path "{merge_path}"'
        run_command(cmd)

        # check if model merge happened
        merge_rec = pycolmap.Reconstruction(str(merge_path))
        if merge_rec.num_images() == 0:
            # merge definitely failed, skip this reconstruction and go to next
            CONSOLE.log(f"[bold red] merge_consecutive: merging {last_rec_name} and {r_name} failed!")
            return False
        else:
            # merge went succesfully (probably)
            #   note: even though there is a reconstruction in the output folder, the merge could still have failed
            #   if "colmap model_merger" fails, it copies the reconstruction from "--input_path2" into the output folder, and exits with EXIT_SUCCESS (=0)
            #   this is useful to us, bc it will allow us to automatically skip the failing merge, but is an implementation detail that is not guaranteed to stay

            # remember this reconstruction
            last_rec_path = merge_path
            last_rec_name = merge_name

    return True



def bundle_adjuster(
    input_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> bool:
    """Run colmap bundle_adjuster on the reconstruction in input_dir."""

    # create output directory
    output_dir.mkdir(exist_ok=True, parents=True)

    # run bundle adjuster
    cmd = f'{colmap_cmd} bundle_adjuster --input_path "{input_dir}" --output_path "{output_dir}"'
    run_command(cmd)

    return True





def run_colmap_extract_and_match(
    database_path: Path,
    image_dir: Path,
    single_camera_per_folder: bool = False,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
):
    """Run colmap feature_extractor and feature_matcher."""

    # create colmap folder
    database_path.parent.mkdir(exist_ok=True, parents=True)

    # feature extraction
    with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
        feature_extractor(
            database_path=database_path,
            image_dir=image_dir,
            verbose=verbose,
            colmap_cmd=colmap_cmd,
            single_camera_per_folder=single_camera_per_folder
        )

    CONSOLE.log("[bold green] Done with COLMAP feature extractor.")

    # exhaustive matching
    with status(msg="[bold yellow]Running COLMAP exhaustive matcher...", spinner="moon", verbose=verbose):
        exhaustive_matcher(
            database_path=database_path,
            verbose=verbose,
            colmap_cmd=colmap_cmd
        )

    CONSOLE.log("[bold green] Done with COLMAP exhaustive matcher.")




def run_colmap_mm_split_reconstruction(
    database_path: Path,
    output_dir: Path,
    image_dir: Path,
    channels: List[str],
    channel_images: List[List[str]],
    keep_folder_content: bool = False,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> Path:
    """Run the split colmap reconstruction pipeline on the given images.
    
    Arguments:
        output_dir: path to the colmap output directory
        image_dir: path to the image folder
        channels: list of channel names
        channel_images: list of lists of image names, one list per channel

    Returns:
        Path to the final, adjusted reconstruction
    """

    # setup paths and variables
    imagelists_dir = output_dir / "imagelists"
    merge_dir = output_dir / "merge"
    final_merge_dir = merge_dir / "final"
    final_reconstruction_path = output_dir / "sparse"


    # clear the output directory, if needed
    if not keep_folder_content:
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True, parents=True)



    # write image lists to files
    mmutils.write_imagelists(channels, channel_images, imagelists_dir)

    # set reconstruction output paths
    reconstruction_paths = [output_dir / "channels" / channel for channel in channels]

    # run seperate reconstructions
    for channel, rpath in zip(channels, reconstruction_paths):
        with status(msg=f"[bold yellow]Running COLMAP mapper on channel {channel}...", spinner="moon", verbose=verbose):
            imagelist_file = imagelists_dir / f"{channel}.txt"

            # run reconstruction with selected image list
            run_reconstruction(database_path, image_dir, rpath, imagelist_file, verbose, colmap_cmd)

            CONSOLE.log(f"[bold green] Done with COLMAP mapper on channel {channel}.")

    CONSOLE.log(f"[bold green] Done with COLMAP mapper on all channels.")

    # select best reconstructions
    reconstruction_paths = [find_best_reconstruction(rpath)[0] for rpath in reconstruction_paths]

    # merge reconstructions
    if len(reconstruction_paths) >= 2:
        with status(msg="[bold yellow]Merging reconstructions...", spinner="moon", verbose=verbose):
            merge_consecutive(
                output_dir=final_merge_dir,
                intermediate_dir=merge_dir,
                reconstruction_paths=reconstruction_paths,
                reconstruction_names=channels,
                verbose=verbose,
                colmap_cmd=colmap_cmd
            )

        CONSOLE.log(f"[bold green] Done merging reconstruction.")
    elif len(reconstruction_paths) == 1: # only one channel present
        final_merge_dir = reconstruction_paths[0]
    else:
        assert False, "RECONSTRUCTION FAILED SOMEHOW" # <-- descriptive error message


    # bundle adjustment
    with status(msg="[bold yellow]Running COLMAP bundle adjuster...", spinner="moon", verbose=verbose):
        bundle_adjuster(final_merge_dir, final_reconstruction_path, verbose, colmap_cmd)

    CONSOLE.log(f"[bold green] DONE: colmap mm split reconstruction.")


    return final_reconstruction_path





def run_colmap_mm_full_reconstruction(
    database_path: Path,
    output_dir: Path,
    image_dir: Path,
    keep_folder_content: bool = False,
    verbose: bool = False,
    colmap_cmd: str = "colmap"
) -> Path:
    """Run the full colmap reconstruction pipeline on the given images.
    
    Arguments:
        output_dir: path to the colmap output directory
        image_dir: path to the image folder

    Returns:
        Path to the final full reconstruction
    """

    # setup paths and variables
    final_reconstruction_path = output_dir / "sparse"
    intermediate_path = output_dir / "full"

    # clear the output directory, if needed
    if not keep_folder_content:
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True, parents=True)


    # run full reconstructions
    with status(msg=f"[bold yellow]Running COLMAP mapper on all channels ...", spinner="moon", verbose=verbose):
        run_reconstruction(database_path, image_dir, intermediate_path, None, verbose, colmap_cmd)

    CONSOLE.log(f"[bold green] Done with COLMAP mapper on all channels.")


    # select the best reconstruction if multiple are present
    reconstruction_path, rec_count = find_best_reconstruction(intermediate_path)

    # copy best reconstruction
    mmutils.copy_dir(reconstruction_path, final_reconstruction_path)

    CONSOLE.log(f"[bold green] DONE: colmap mm full reconstruction.")

    return final_reconstruction_path
    








    









def mm_colmap_to_transforms_json(
    reconstruction_dir: Path,
    output_dir: Path,
    channels: List[str],
    channel_images: List[List[str]]
) -> List[List[str]]:
    """Given a colmap reconstruction at reconstruction_dir, generate the nerfstudio transforms.json file, including channel information
    
    Arguments:
        reconstruction_dir: Path to colmap reconstruction
        output_dir: Path to where the transforms.json should be stored
        channels: List of channel names
        channel_images: List of image names for each channel

    Returns: One List of image names per channel, listing which images from each channel were succesfully registered
    """

    # first, generate the basic transforms.json file via nerfstudios colmap_utils
    colmap_utils.colmap_to_json(
        recon_dir=reconstruction_dir,
        output_dir=output_dir
    )
    
    # load transforms.json file to add channel information
    transforms_file = output_dir / "transforms.json"
    with transforms_file.open() as tf:
        transforms = json.load(tf)

    # validate format
    if not "frames" in transforms:
        CONSOLE.log("[bold red] transforms.json missing frames")
        return 0

    # sort frames by filenames (why? for debug reasons, and possibly useful for future channel frame matching)
    transforms["frames"] = sorted(transforms["frames"], key=lambda f: f["file_path"])

    # add channel information to frames and collect reconstructed image names
    rec_imgs = [[] for c in channels]

    for frame in transforms["frames"]:
        img_path = frame["file_path"]
        img_name = str(Path(*Path(img_path).parts[1:]))

        # find corresponding channel
        present_channels = [i for i,imgs in enumerate(channel_images) if img_name in imgs]

        if len(present_channels) > 0:
            frame["mm_channel"] = channels[present_channels[0]]
            rec_imgs[present_channels[0]].append(img_name)
    

    # write transforms.json
    with transforms_file.open("w", encoding="utf-8") as tf:
        json.dump(transforms, tf, indent=4)

    CONSOLE.log(f"[bold green] transforms.json generated.")
    
    return rec_imgs









def mm_thermal_colmap_to_transforms_json(
        reconstruction_dir: Path,
        output_dir: Path,
        channels: List[str],
        channel_images: List[List[str]],
        rewrite_sorted_filename: bool = True,
        use_single_camera_mode: bool = True,
        name_mapping: dict = {},
        train_test_indices: tuple = None) -> List[List[str]]:
    """Given a colmap reconstruction at reconstruction_dir, generate the nerfstudio transforms.json file, including channel information

    Arguments:
        reconstruction_dir: Path to colmap reconstruction
        output_dir: Path to where the transforms.json should be stored
        channels: List of channel names
        channel_images: List of image names for each channel

    Returns: One List of image names per channel, listing which images from each channel were succesfully registered
    """

    # first, generate the basic transforms.json file via nerfstudios colmap_utils
    colmap_utils.colmap_to_json(
        recon_dir=reconstruction_dir,
        output_dir=output_dir,
        use_single_camera_mode=use_single_camera_mode
    )

    # load transforms.json file to add channel information
    transforms_file = output_dir / "transforms.json"
    with transforms_file.open() as tf:
        transforms = json.load(tf)

    # validate format
    if not "frames" in transforms:
        CONSOLE.log("[bold red] transforms.json missing frames")
        return 0

    # sort frames by filenames (why? for debug reasons, and possibly useful for future channel frame matching)
    transforms["frames"] = sorted(transforms["frames"], key=lambda f: f["file_path"])

    # add channel information to frames and collect reconstructed image names
    rec_imgs = [[] for c in channels]

    num_images = transforms["frames"].__len__() + 1
    frame_thermal_list = []

    train_indecies, test_indices = train_test_indices

    train_images = [list(name_mapping['D'].keys())[i] for i in train_indecies]
    test_images = [list(name_mapping['D'].keys())[i] for i in test_indices]

    for frame in transforms["frames"]:
        img_path = frame["file_path"]
        img_name = Path(img_path).name

        frame_thermal = copy.deepcopy(frame)

        # if rewrite_sorted_filename:
        #    frame["file_path"] = str(Path(img_path).parents[1] / img_name)
        frame["file_path"] = str(Path(img_path).parents[0] / name_mapping['D'][img_name])
        frame["mm_channel"] = "D"
        frame["train"] = img_name in train_images

        frame_thermal['colmap_im_id'] += num_images
        frame_thermal['mm_channel'] = "MS_T"
        frame_thermal['file_path'] = str(Path(img_path).parents[0] / name_mapping['MS_T'][img_name])
        frame_thermal["train"] = img_name in train_images

        frame_thermal_list.append(frame_thermal)

    transforms["frames"].extend(frame_thermal_list)
    # write transforms.json
    with transforms_file.open("w", encoding="utf-8") as tf:
        json.dump(transforms, tf, indent=4)

    CONSOLE.log(f"[bold green] transforms.json generated.")

    return rec_imgs





def mm_merge_transforms_json(
        output_dir: Path,
        channels: List[str],
        channel_images: List[List[str]],
        rgb_mode_str: str = "D",
        image_folder_name: str = "images",
        camera_model: str = "PINHOLE") -> List[List[str]]:
    """Given a colmap reconstruction at reconstruction_dir, generate the nerfstudio transforms.json file, including channel information

    Arguments:
        reconstruction_dir: Path to colmap reconstruction
        output_dir: Path to where the transforms.json should be stored
        channels: List of channel names
        channel_images: List of image names for each channel

    Returns: One List of image names per channel, listing which images from each channel were succesfully registered
    """

    # load transforms.json file to add channel information
    transforms_file = output_dir / "transforms.json"
    with transforms_file.open() as tf:
        transforms = json.load(tf)

    # validate format
    if not "frames" in transforms:
        CONSOLE.log("[bold red] transforms.json missing frames")
        return 0

    # sort frames by filenames (why? for debug reasons, and possibly useful for future channel frame matching)
    transforms["frames"] = sorted(transforms["frames"], key=lambda f: f["file_path"])

    ms_transforms = []

    # Add RGB / D to transform json and delete obsolete
    rgb_image_name = sorted(channel_images[channels.index(rgb_mode_str)])
    del (channels[channels.index(rgb_mode_str)])
    for i, frame in enumerate(transforms["frames"]):
        transforms["frames"][i].update({"mm_channel": "D"})
        transforms["frames"][i]['file_path'] = image_folder_name + "/" + rgb_image_name[i]
        transforms["frames"][i]['file_path'] = image_folder_name + "/" + rgb_image_name[i]

        transforms["frames"][i]['camera_model'] = camera_model

        # Delete entries for the sake of a good overview
        if "semantics" in transforms["frames"][i].keys():
            transforms["frames"][i].pop("semantics")
        if "semantic_classes" in transforms["frames"][i].keys():
            transforms["frames"][i].pop("semantic_classes")
        if "instance" in transforms["frames"][i].keys():
            transforms["frames"][i].pop("instance")
        if "MS_R" in transforms["frames"][i].keys():
            transforms["frames"][i].pop("MS_R")
        if "MS_G" in transforms["frames"][i].keys():
            transforms["frames"][i].pop("MS_G")

        for c in channels:
            channel_image_name = sorted(channel_images[channels.index(c)])
            ms_frame = copy.deepcopy(transforms["frames"][i])

            # Set new mm_channel
            ms_frame['mm_channel'] = c

            # Set image path
            ms_frame['file_path'] = image_folder_name + "/" + channel_image_name[i]

            ms_transforms.append(ms_frame)

    # Extend transforms.json with new multi spectral data
    transforms["frames"] = transforms["frames"] + ms_transforms

    # write transforms.json
    with transforms_file.open("w", encoding="utf-8") as tf:
        json.dump(transforms, tf, indent=4)

    CONSOLE.log(f"[bold green] transforms.json generated.")
