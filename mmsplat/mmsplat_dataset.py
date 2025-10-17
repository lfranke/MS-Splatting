import sys
import re
from pprint import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union
import shutil
import os

from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE, status

import mmsplat.process_data.mmutils as mmutils
import mmsplat.process_data.mmcolmap as mmcolmap



@dataclass
class MMSplatDataset(BaseConverterToNerfstudioDataset):
    """Class to preprocess multi-modal camera data and generate sparse reconstruction"""

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""
    
    primary_channel: Optional[str] = None
    """The name of the image folder of the primary channel for the split strategy. If None or invalid, the alphabetical first will be chosen."""

    channel_overlap: int = 20
    """Number of overlap images that will be used during initial reconstruction in the split strategy."""

    overlap_strategy: Literal["first", "last", "random", "random-separate"] = "first"
    """What set of images to use for overlap with split strategy. Options are:
            first: Use the first {channel_overlap} images.
            last: Use the last {channel_overlap} images.
            random: Use random images, but the same indices from every channel.
            random-separate: Use independently random image from each channel.
    """

    colmap_cmd: str = "colmap"
    """The command to run colmap."""

    skip_image_copy: bool = False
    """Whether to skip the image copy step. If set, --data will be ignored, and channel information will be extracted from output file names, assuming they follow the format: {channel}_{index}.{suffix}"""

    skip_colmap: bool = False
    """Whether to skip reconstructing the camera poses with colmap."""

    skip_colmap_extract_and_match: bool = False
    """Whether to skip colmap feature_extractor and feature_matcher."""

    method: Literal["split", "full", "conditional", "both"] = "conditional"
    """Method for multi-spectrol camera co-registration
        split: all channels will be registered separately, and merged after
        full: all channels will be registered in a single COLMAP model
        conditional: "full" will be run first, and if the fraction of registered cameras is less than conditional_fraction, the "split" method is also attempted. Of both, the model with the higher number of registered cameras is selected.
        both: "full" and "split" are both run. Of both, the model with the higher number of registered cameras is selected.
    """

    conditional_fraction: float = 0.9
    """See method: conditional"""

    flatten_folders: bool = False
    """Whether to flatten the folder structure (e.g. all images one level deep) or keep the channel subfolders when copying"""

    single_camera_per_folder: bool = True
    """Whether to use a single camera per folder."""



    @property
    def colmap_dir(self) -> Path:
        return self.output_dir / "colmap"

    @property
    def colmap_image_dir(self) -> Path:
        return self.output_dir / "colmap_images"


    def run_reconstruction(
        self,
        channels: List[str],
        channel_images: List[List[str]],
    ) -> Path:
        database_path = self.colmap_dir / "database.db"
        full_dir = self.colmap_dir / "full"
        split_dir = self.colmap_dir / "split"
        final_dir = self.colmap_dir / "sparse"

        total_image_count = sum(len(l) for l in channel_images)

        if not self.skip_colmap_extract_and_match:
            mmcolmap.run_colmap_extract_and_match(
                database_path=database_path,
                image_dir=self.colmap_image_dir,
                verbose=self.verbose,
                colmap_cmd=self.colmap_cmd,
                single_camera_per_folder=self.single_camera_per_folder
            )


        
        if self.method == "full" or self.method == "both" or self.method == "conditional":
            full_path = mmcolmap.run_colmap_mm_full_reconstruction(
                database_path=database_path,
                output_dir=full_dir,
                image_dir=self.colmap_image_dir,
                keep_folder_content=False,
                verbose=self.verbose,
                colmap_cmd=self.colmap_cmd
            )

            full_img_count = mmcolmap.get_reconstruction_image_count(full_path)
        else:
            full_path = None
            full_img_count = -1

        # condition for method=="conditional"
        condition_met = total_image_count * self.conditional_fraction <= full_img_count

        # debug outputs
        if self.method == "conditional":
            if condition_met:
                CONSOLE.log(f"[bold green] {full_img_count} out of {total_image_count} registered, condition met. Skipping split reconstruction.")
            else:
                CONSOLE.log(f"[bold green] {full_img_count} out of {total_image_count} registered, condition not met. Trying split reconstruction.")


        if self.method == "split" or self.method == "both" or (self.method == "conditional" and not condition_met):
            # generate image lists
            image_lists = mmutils.generate_imagelists(
                channel_images=channel_images,
                channel_overlap=self.channel_overlap,
                overlap_strategy=self.overlap_strategy
            )

            split_path = mmcolmap.run_colmap_mm_split_reconstruction(
                database_path=database_path,
                output_dir=split_dir,
                image_dir=self.colmap_image_dir,
                channels=channels,
                channel_images=image_lists,
                keep_folder_content=False,
                verbose=self.verbose,
                colmap_cmd=self.colmap_cmd
            )

            split_img_count = mmcolmap.get_reconstruction_image_count(split_path)
        else:
            split_path = None
            split_img_count = -1


        #debug outputs
        if (self.method == "conditional" and not condition_met) or self.method == "both":
            if full_img_count >= split_img_count:
                CONSOLE.log(f"[bold green] Reconstructed: full={full_img_count}, split={split_img_count}, total={total_image_count}. Selecting 'full'.")
            else:
                CONSOLE.log(f"[bold green] Reconstructed: full={full_img_count}, split={split_img_count}, total={total_image_count}. Selecting 'split'.")


        final_dir.mkdir(parents=True, exist_ok=True)

        if full_img_count >= split_img_count:
            mmutils.copy_dir(full_path, final_dir)
        else:
            mmutils.copy_dir(split_path, final_dir)

        return final_dir


    def main(self):
        """Main function to process the data."""

        # image copying step
        if not self.skip_image_copy:
            # list channels
            channel_list = mmutils.get_channel_list(self.data, self.primary_channel)

            # validate channel presence
            if len(channel_list) == 0:
                CONSOLE.log("[bold red]No channels found in data directory.")
                sys.exit(1)
                
            # copy images
            channel_images = mmutils.copy_images(
                data_dir=self.data,
                channels=channel_list,
                output_dir=self.image_dir,
                colmap_image_dir=self.colmap_image_dir,
                flatten_folders=self.flatten_folders,
                num_downscales=self.num_downscales,
                verbose=self.verbose
            )
            
        else:  # skip images copy, generate channel_list and channel_images from images directory
            channel_list, channel_images = mmutils.regenerate_image_information(
                image_dir=self.image_dir,
                primary_channel=self.primary_channel,
                channel_subfolders=True
            )


        # run colmap
        if not self.skip_colmap:
            reconstruction_path = self.run_reconstruction(
                channels=channel_list,
                channel_images=channel_images
            )
        else:
            reconstruction_path = self.colmap_dir / "sparse"

        # convert colmap reconstruction to nerfstudio transforms.json
        rec_images = mmcolmap.mm_colmap_to_transforms_json(
            reconstruction_dir=reconstruction_path,
            output_dir=self.output_dir,
            channels=channel_list,
            channel_images=channel_images
        )



        # based on reconstruction ratios, give feedback on quality of reconstruction
        for c, cis, ris in zip(channel_list, channel_images, rec_images):
            channel_img_count = len(cis)
            recon_img_count = len(ris)
            channel_ratio = recon_img_count / channel_img_count

            # arbitrary percentage cutoffs
            if channel_ratio > 0.88:
                CONSOLE.log(f"[bold green]Channel {c}: {recon_img_count} / {channel_img_count} registered.")
            elif channel_ratio > 0.53:
                CONSOLE.log(f"[bold yellow]Channel {c}: {recon_img_count} / {channel_img_count} registered.")
            else:
                CONSOLE.log(f"[bold red]Channel {c}: {recon_img_count} / {channel_img_count} registered.")

        # quality of all channels at once
        img_count = sum(len(ci) for ci in channel_images)
        recon_count = sum(len(ci) for ci in rec_images)
        recon_ratio = recon_count / img_count

        if recon_ratio > 0.88:
            CONSOLE.log(f"[bold green]Total: {recon_count} / {img_count} registered.")
        elif recon_ratio > 0.53:
            CONSOLE.log(f"[bold yellow]Total: {recon_count} / {img_count} registered.")
        else:
            CONSOLE.log(f"[bold red]Total: {recon_count} / {img_count} registered.")


        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")





@dataclass
class MMSplatThermalDataset(BaseConverterToNerfstudioDataset):
    """Class to preprocess multi-modal camera data and generate sparse reconstruction"""

    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""

    primary_channel: Optional[str] = None
    """The name of the image folder of the primary channel. If None or invalid, the alphabetical first will be chosen."""

    skip_image_copy: bool = False
    """Whether to skip the image copy step. If set, --data will be ignored, and channel information will be extracted from output file names, assuming they follow the format: {channel}_{index}.{suffix}"""

    single_camera_per_folder: bool = True
    """Whether to use a single camera per folder."""

    @property
    def colmap_dir(self) -> Path:
        return self.data / "colmap"

    @staticmethod
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

    def main(self):
        """Main function to process the data."""

        # image copying step
        if not self.skip_image_copy:
            # list channels
            channel_list = ['D', 'MS_T']

            # validate channel presence
            if len(channel_list) == 0:
                CONSOLE.log("[bold red]No channels found in data directory.")
                sys.exit(1)

            # copy images
            channel_images, mapping, indices = mmutils.copy_images_train_test(
                data_dir=self.data,
                channels=channel_list,
                output_dir=self.image_dir,
                num_downscales=self.num_downscales,
                verbose=self.verbose
            )

        else:  # skip images copy, generate channel_list and channel_images from images directory
            channel_list, channel_images = mmutils.regenerate_image_information(self.image_dir, self.primary_channel)

        reconstruction_path = self.colmap_dir / "sparse/0"

        # convert colmap reconstruction to nerfstudio transforms.json
        rec_images = mmcolmap.mm_thermal_colmap_to_transforms_json(
            reconstruction_dir=reconstruction_path,
            output_dir=self.output_dir,
            channels=channel_list,
            channel_images=channel_images,
            use_single_camera_mode=False,
            name_mapping=mapping,
            train_test_indices=indices
        )

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")




@dataclass
class MMSplatBlenderDataset(BaseConverterToNerfstudioDataset):
    """Class to preprocess multi-modal camera data and generate sparse reconstruction"""

    data: List[Path]
    """Path to two projects. Fuse both into on project and correct transform.json"""

    image_folder_name = "images"
    """image folder name. This folder is selected from the first data path."""

    modes: List[str] = "MS_G", "MS_R", "MS_NIR", "MS_RE"
    """Different modes to use for a multi spectral case"""

    num_downscales: int = 1
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3 will downscale the
       images by 2x, 4x, and 8x."""

    skip_image_copy: bool = False
    """Whether to skip the image copy step. If set, --data will be ignored, and channel information will be extracted from output file names, assuming they follow the format: {channel}_{index}.{suffix}"""

    camera_model: str = "PINHOLE"
    """Camera Model for blender project"""

    @property
    def colmap_dir(self) -> Path:
        return self.output_dir / "colmap"

    def find_channels(self) -> dict:
        """Search for MS image folders in data paths. Only the first stage is checked."""

        ms_channel_dict = {}
        for base_path in self.data:
            for ms_folder in self.modes:
                ms_folder_path = base_path / ms_folder

                if ms_folder_path.exists():
                    ms_channel_dict.update({ms_folder: ms_folder_path})

        if ms_channel_dict.keys().__len__() != self.modes.__len__():
            CONSOLE.log("[bold red] Not all modalities are found in data directory. Please Check!")

        return ms_channel_dict

    def main(self):
        """Main function to process the data."""

        # list channels
        channel_dict = self.find_channels()
        channel_dict.update({"D": self.data[0] / "images"})
        channel_list = list(channel_dict.keys())

        transform_json_src_path = self.data[0] / "transforms.json"
        transform_json_dest_path = self.output_dir / "transforms.json"

        # image copying step
        if not self.skip_image_copy:

            # copy images
            channel_images = mmutils.copy_images(
                data_dir=self.data,
                channels=channel_dict,
                output_dir=self.image_dir,
                num_downscales=self.num_downscales,
                verbose=self.verbose,
                flatten_folders=True
            )

        else:  # skip images copy, generate channel_list and channel_images from images directory
            channel_list, channel_images = mmutils.regenerate_image_information(self.image_dir, self.primary_channel)

        if os.path.exists(transform_json_dest_path):
            os.remove(transform_json_dest_path)

        shutil.copy(transform_json_src_path, transform_json_dest_path)

        # convert colmap reconstruction to nerfstudio transforms.json

        # TODO: MISSING FUNCTION ?!?!?!?
        mmcolmap.mm_merge_transforms_json(
            output_dir=self.output_dir,
            channels=channel_list,
            channel_images=channel_images,
            image_folder_name=self.image_folder_name,
            rgb_mode_str="D",
            camera_model=self.camera_model
        )

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")