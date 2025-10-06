import sys
import copy
import random
import re
from pathlib import Path
from typing import List, Literal, Dict, Tuple, Optional
import shutil
import os

from nerfstudio.process_data import process_data_utils
from nerfstudio.utils.rich_utils import CONSOLE, status

from mmsplat.process_data.copy_images import copy_images_list


def get_channel_list(
    data_dir: Path,
    primary_channel: str = None
) -> List[str]:
    """Get a list of all channel folders in the data directory. If primary_channel is in the list, it will be moved to the front of the list."""

    channels = sorted(str(channel.name) for channel in data_dir.iterdir() if channel.is_dir())

    # move primary channel to front of list
    if primary_channel in channels:
        channels.remove(primary_channel)
        channels.insert(0, primary_channel)
    
    return channels



def copy_images(
    data_dir: Path,
    channels: List[str],
    output_dir: Path,
    flatten_folders: bool,
    colmap_image_dir: Optional[Path] = None,
    num_downscales: int = 0,
    verbose=False
) -> List[List[str]]:
    """
    Copy images from the data directory to the output directory.
    Returns a List of Lists of the image names, where the outer list is the channel and the inner list is the image names.
    """

    # result
    channel_images = []

    # progress indicator
    with status(msg="[bold yellow]Copying images...", spinner="bouncingBall", verbose=verbose):

        # iterate channels and copy images for each channel
        for c_index, channel in enumerate(channels):
            channel_dir = data_dir / channel
            prefix = f"{channel}_"
            
            # get image list, and ensure it is sorted
            image_paths = sorted(process_data_utils.list_images(channel_dir))
            if len(image_paths) == 0:
                CONSOLE.log(f"[bold red]:skull: No usable images in the data folder {channel}.")
                sys.exit(1)

            # copy images
            res_paths = copy_images_list(
                image_paths=image_paths,
                image_dir=output_dir,
                image_prefix=prefix,
                colmap_image_dir=colmap_image_dir,
                subfolder=(None if flatten_folders else channel),
                num_downscales=num_downscales,
                verbose=verbose,
                keep_image_dir=(c_index != 0)   # only delete current content on first channel
            )

            # add to result
            if flatten_folders:
                channel_images.append([res_path.name for res_path in res_paths])
            else:
                channel_images.append([(channel + "/" + res_path.name) for res_path in res_paths])

    CONSOLE.log("[bold green] Done copying images.")

    return channel_images







def generate_unique_indices(array_length: int, num_indices: int) -> List[int]:
    if num_indices > array_length:
        num_indices = array_length
    
    return random.sample(range(array_length), num_indices)




def generate_imagelists(
    channel_images: List[List[str]],
    channel_overlap: int = 20,
    overlap_strategy: Literal["first", "last", "random", "random-separate"] = "first"
) -> List[List[str]]:
    """Generate image lists given the channel images and the overlap strategy"""
    
    # minimum list length
    min_list_len = min(len(img_names) for img_names in channel_images)
    
    # clone channel_images
    image_lists = copy.deepcopy(channel_images)
    
    # check if overlap is needed
    if channel_overlap <= 0 or len(image_lists) <= 1:
        return image_lists
    
    
    # iterate secondary channels and generate overlap
    for it, img_names in enumerate(channel_images[1:]):
        
        # add overlap based on strategy
        
        if overlap_strategy == "first":
            # append first {channel_overlap} images
            image_lists[0].extend(img_names[:channel_overlap])
            
        elif overlap_strategy == "last":
            # append last {channel_overlap} images
            image_lists[0].extend(img_names[-channel_overlap:])
            
        elif overlap_strategy == "random":
            # generate random indices on first iteration
            if it == 0:
                rnd_indices = generate_unique_indices(min_list_len, channel_overlap)
            
            # append images selected by indices
            image_lists[0].extend(img_names[i] for i in rnd_indices)
            
        elif overlap_strategy == "random-separate":
            # generate random indices on every iteration
            rnd_indices = generate_unique_indices(len(img_names), channel_overlap)
            
            # append images selected by indices
            image_lists[0].extend(img_names[i] for i in rnd_indices)
            
        else:
            raise ValueError(f"Invalid overlap strategy: {overlap_strategy}")
        
    return image_lists



def write_imagelists(
    channels: List[str],
    channel_images: List[List[str]],
    output_dir: Path
):
    """Write image lists to a output directory."""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # write image lists
    for channel, imglist in zip(channels, channel_images):
        list_file = output_dir / f"{channel}.txt"
        
        with open(list_file, "w") as f:
            f.write("\n".join(imglist))
            
            
def read_imagelists(
    dir: Path
) -> Dict[str, List[str]]:
    """Read image lists from a directory."""
    
    # read image lists
    image_lists = {}
    for list_file in dir.iterdir():
        channel = list_file.stem
        with open(list_file, "r") as f:
            image_lists[channel] = [line.strip() for line in f]
    
    return image_lists





def regenerate_image_information(
    image_dir: Path,
    primary_channel: Optional[str] = None,
    channel_subfolders: bool = True
) -> Tuple[List[str], List[List[str]]]:
    """Given the images folder, regenerate the channel list and the channel images.
    Assumption: All images are named in the form: {channel}_{index}.{suffix}
    """
    
    # read image names from image_dir
    img_names = process_data_utils.list_images(image_dir, recursive=False)

    # extract channel from filenames
    img_regex = r"(.*)_[0-9]+\..*"
    img_lists = {}
    
    for iname in img_names:
        match = re.fullmatch(img_regex, iname.name)

        if match:
            # successfully matched to channel, add to channels list
            if match.group(1) in img_lists:
                img_lists[match.group(1)].append(iname.name)
            else:
                img_lists[match.group(1)] = [iname.name]
        else:
            # match failed, invalid image file name format
            CONSOLE.log(f"[bold red]:skull: Invalid image file name \"{iname.name}\".")
            continue # skip image, but continue


    if channel_subfolders:
        for cdir in image_dir.iterdir():
            if not cdir.is_dir():
                continue

            c_images = process_data_utils.list_images(cdir)

            # add images to channel list
            if cdir.name in img_lists:
                img_lists[cdir.name].extend([f'{cdir.name}/{ci.name}' for ci in c_images])
            else:
                img_lists[cdir.name] = [f'{cdir.name}/{ci.name}' for ci in c_images]

    

    # collect channel list and move primary_channel to the front, if present
    channels = sorted(img_lists.keys())
    if primary_channel in channels:
        channels.remove(primary_channel)
        channels.insert(0, primary_channel)

    # collect channel images
    channel_images = [img_lists[c] for c in channels]

    # return
    return channels, channel_images




def copy_dir(
    src_dir: Path,
    dst_dir: Path
):
    if not src_dir.is_dir():
        raise ValueError(f"Source {src_dir} is not a directory")
    
    dst_dir.mkdir(parents=True, exist_ok=True)  # Ensure the destination exists

    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)  # Copy subdirectories recursively
        else:
            shutil.copy2(item, target)  # Copy files with metadata









def write_filenames_to_txt(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in os.listdir(directory):
            f.write(filename + '\n')


def find_indices_with_substring(paths_list, substring):
    return [index for index, path in enumerate(paths_list) if substring in path.__str__()]

def copy_images_train_test(
        data_dir: Path,
        channels: List[str],
        output_dir: Path,
        num_downscales: int = 0,
        verbose=False
) -> List[List[str]]:
    """
    Copy images from the data directory to the output directory.
    Returns a List of Lists of the image names, where the outer list is the channel and the inner list is the image names.
    """

    # result
    channel_images = []

    mapping = {}

    # progress indicator
    with status(msg="[bold yellow]Copying images...", spinner="bouncingBall", verbose=verbose):

        # iterate channels and copy images for each channel
        for c_index, channel in enumerate(channels):
            channel_dir = data_dir / channel
            prefix = f"{channel}_"

            # get image list, and ensure it is sorted
            image_paths = sorted(process_data_utils.list_images(channel_dir))
            if len(image_paths) == 0:
                CONSOLE.log(f"[bold red]:skull: No usable images in the data folder {channel}.")
                sys.exit(1)

            # copy images
            res_paths = copy_images_list(
                image_paths=image_paths,
                image_dir=output_dir,
                image_prefix=prefix,
                num_downscales=num_downscales,
                verbose=verbose,
                keep_image_dir=(c_index != 0)  # only delete current content on first channel
            )

            mapping.update({channel: {}})
            for source_image_name, target_image_name in zip(image_paths, res_paths):
                mapping[channel].update({source_image_name.name: target_image_name.name})

            indices_test = find_indices_with_substring(image_paths, "/test/")
            indices_train = find_indices_with_substring(image_paths, "/train/")
            # add to result
            channel_images.append([res_path.name for res_path in res_paths])

            test_images = [res_paths[i].name for i in indices_test]
            train_images = [res_paths[i].name for i in indices_train]

            with open(output_dir.parent / '{}_test.txt'.format(channel), 'w') as f:
                for line in test_images:
                    f.write(f"{line}\n")

            with open(output_dir.parent / '{}_train.txt'.format(channel), 'w') as f:
                for line in train_images:
                    f.write(f"{line}\n")

    CONSOLE.log("[bold green] Done copying images.")

    return channel_images, mapping, (indices_train, indices_test)
