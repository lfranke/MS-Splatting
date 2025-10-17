from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple, Type, Dict, Any, List
import os

import math
import re
import json
import numpy as np
import torch
from PIL import Image

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParser
from nerfstudio.cameras import camera_utils
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)

from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE




@dataclass
class MMSplatDataParserConfig(DataParserConfig):
    """mmsplat data parser config"""

    _target: Type = field(default_factory=lambda: MMSplatDataParser)
    """target class"""
    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <= auto_downscale_max_res pixels."""
    auto_downscale_max_res: int = 1600
    """If downscale_factor is None, what should the maximum dimension of a chosen image be."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    eval_mode: Literal["fraction", "interval", "all", "json", "json-list", "fraction-groups"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    fraction: split based on a percentage for train and the remaining for eval
    interval: uses every nth frame for eval.
    all: uses all the images for any split.
    json: checks "train" flag in each frame of the transforms.json
    json-list: checks json file (see json_list_path) for two arrays (train and eval) containing file names of respective sets
    fraction-groups: split based on percentage, but make sure that equal image indices are chosen across all channels
    """
    json_list_path: Optional[Path] = None
    """Path to the list containing train/eval split filenames for eval_mode="json-list". Defaults to $data$/train_split.json"""
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    load_3D_points: bool = True
    """Whether to load the 3D points from the colmap reconstruction."""




@dataclass
class MMSplatDataParser(DataParser):
    """mmsplat DataParser"""

    config: MMSplatDataParserConfig


    def _generate_dataparser_outputs(self, split="train") -> DataparserOutputs:
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."

        # load transforms.json and set data containing directory
        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        # select downscale factor and image file names
        downscale_factor, ds_filenames = MMSplatDataParser._select_downscale_factor(
            frames=meta["frames"],
            data_dir=data_dir,
            downscale_factor=self.config.downscale_factor,
            autoscale_max_res=self.config.auto_downscale_max_res
        )

        # sort frames by file names
        frames = sorted(zip(ds_filenames, meta["frames"]), key=lambda t: t[0])
        
        # generate camera parameters
        fx, fy, cx, cy, width, height, camera_type, distort, poses = MMSplatDataParser._generate_camera_parameters(meta, (f for _,f in frames))
        metadata = {}
        if "fisheye_crop_radius" in meta:
            metadata["fisheye_crop_radius"] = meta.get("fisheye_crop_radius", None)


        # orient, center and scale poses
        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method
        
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=orientation_method,
            center_method=self.config.center_method
        )

        scale_factor = 1.0 * self.config.scale_factor
        poses[:, :3, 3] *= scale_factor


        # select indices for train and eval split
        i_train, i_eval = MMSplatDataParser._get_train_eval_split(
            frames=[f[1] for f in frames],
            eval_mode=self.config.eval_mode,
            train_split_fraction=self.config.train_split_fraction,
            eval_interval=self.config.eval_interval,
            json_list_path=(self.config.json_list_path if (self.config.json_list_path is not None) else (data_dir / "train_split.json"))
        )

        if split == "train":
            split_indices = i_train
        elif split in ["val", "test"]:
            split_indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        
        split_idx_tensor = torch.tensor(split_indices, dtype=torch.long)


        # reduce frames, poses and parameters via split indices
        frames = [frames[i] for i in split_indices]
        frame_channels = [MMSplatDataParser._get_mm_channel(f) for _,f in frames]  # determine channel of every frame
        camera_type = [camera_type[i] for i in split_indices]
        poses = poses[split_idx_tensor]
        cameras = Cameras(
            fx=torch.tensor(fx, dtype=torch.float32)[split_idx_tensor],
            fy=torch.tensor(fy, dtype=torch.float32)[split_idx_tensor],
            cx=torch.tensor(cx, dtype=torch.float32)[split_idx_tensor],
            cy=torch.tensor(cy, dtype=torch.float32)[split_idx_tensor],
            width=torch.tensor(width, dtype=torch.int32)[split_idx_tensor],
            height=torch.tensor(height, dtype=torch.int32)[split_idx_tensor],
            distortion_params=torch.stack(distort, dim=0)[split_idx_tensor],
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata,
        )

        # scale cameras according to downscale_factor
        cameras.rescale_output_resolution(scaling_factor=1.0 / (2 ** downscale_factor))
        
        # load 3d points, if requested
        metadata = {}
        if self.config.load_3D_points:
            if not "ply_file_path" in meta:
                CONSOLE.log("[bold red] transforms.json doesnt specify ply_file_path, cant load 3D points")
            else:
                # read 3d points from ply file
                ply_file_path = data_dir / meta["ply_file_path"]

                sparse_points = self._load_3D_points(ply_file_path, transform_matrix, scale_factor)
                if sparse_points is not None:
                    metadata.update(sparse_points)

        # set scene box
        # NOTE: for now this is a placeholder, but could become relevant later on
        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = 1.0 #self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )


        # create DataparserOutputs
        return DataparserOutputs(
            image_filenames=[f for f,_ in frames],
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "mm_channel": frame_channels, # per-frame channel information
                **metadata
            },
        )



    @staticmethod
    def _export_train_eval_split(
        json_file_path: Path, # Path to output json file containing train and eval image file names
        data: Path, # Directory or explicit json file path specifying location of data/transforms.json.
        eval_mode: Literal["fraction", "interval", "all", "json", "json-list", "fraction-groups"] = "fraction", # method for selecting images
        train_split_fraction: float = 0.9, # The percentage of the dataset to use for training. Only used when eval_mode is fraction.
        eval_interval: int = 8, # The interval between frames to use for eval. Only used when eval_mode is eval-interval.
        json_list_path: Optional[Path] = None
    ) -> bool:
        assert data.exists(), f"Data directory {data} does not exist."

        # load transforms.json and set data containing directory
        if data.suffix == ".json":
            meta = load_from_json(data)
            data_dir = data.parent
        else:
            meta = load_from_json(data / "transforms.json")
            data_dir = data

        # sort frames by file names
        frames = sorted(meta["frames"], key=lambda f: f["file_path"])

        # select indices for train and eval split
        i_train, i_eval = MMSplatDataParser._get_train_eval_split(
            frames=frames,
            eval_mode=eval_mode,
            train_split_fraction=train_split_fraction,
            eval_interval=eval_interval,
            json_list_path=(json_list_path if (json_list_path is not None) else (data_dir / "train_split.json"))
        )

        # generate and write json structure
        json_out = {
            "train": [frames[i]["file_path"] for i in i_train],
            "eval": [frames[i]["file_path"] for i in i_eval]
        }

        with json_file_path.open("w") as json_file:
            json.dump(json_out, json_file, indent=4)

        return True






    @staticmethod
    def _load_3D_points(ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # if no points found don't read in an initial point cloud
        if len(pcd.points) == 0:
            return None

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out




    @staticmethod
    def get_train_eval_split_json(frames: List) -> Tuple[np.ndarray, np.ndarray]:
        i_train = []
        i_eval = []
        for idx, frame in enumerate(frames):
            if frame[1]['train']:
                i_train.append(idx)
            else:
                i_eval.append(idx)
        return np.asarray(i_train), np.asarray(i_eval)

    @staticmethod
    def get_train_eval_split_fraction_groups(frames, fraction) -> Tuple[np.ndarray, np.ndarray]:
        # collect indices present for each channel
        channel_indices = {}

        for fi, f in enumerate(frames):
            c_name = f["mm_channel"]
            if c_name not in channel_indices:
                channel_indices[c_name] = []

            filestem = Path(f["file_path"]).stem
            index_match = match = re.search(r'_(\d+)$', filestem)
            if index_match:
                channel_indices[c_name].append((int(index_match.group(1)), fi))

        # generate overlapping set
        cl_lists = list(channel_indices.values())
        intersection = set(i for i,_ in cl_lists[0])
        for ils in cl_lists[1:]:
            intersection &= set(i for i,_ in ils)
        int_indices = list(sorted(intersection))

        # number of desired sets
        eval_set_count = math.ceil((1 - fraction) * (len(frames) / len(channel_indices.keys())))

        if eval_set_count >= len(int_indices):
            selected_indices = int_indices
        else:
            eval_sis = np.linspace(0, len(int_indices) - 1, eval_set_count, dtype=int)
            selected_indices = [int_indices[i] for i in eval_sis]

        # generate final index lists
        i_train = []
        i_eval = []
        
        for c_name, c_list in channel_indices.items():
            for i, fi in c_list:
                if i in selected_indices:
                    i_eval.append(fi)
                else:
                    i_train.append(fi)

        return np.asarray(i_train), np.asarray(i_eval)


    @staticmethod
    def get_train_eval_split_json_list(frames, json_list_path) -> Tuple[np.ndarray, np.ndarray]:
        def has_matching_file(path_without_ext, paths_with_ext):
            """
            Check if `path_without_ext` (e.g. "/imgs/pic1") matches any in
            `paths_with_ext` once you strip off their extensions.
            """
            return any(
                os.path.splitext(p)[0] == path_without_ext
                for p in paths_with_ext
            )

        assert Path(json_list_path).is_file(), f"{json_list_path} is not json file."

        with json_list_path.open("r") as file:
            json_content = json.load(file)

        i_train = []
        i_eval = []

        for fi, f in enumerate(frames):
            base_filename_path = f["file_path"].strip(Path(f["file_path"]).suffix)
            if has_matching_file(base_filename_path, json_content["train"]):
                i_train.append(fi)
            if has_matching_file(base_filename_path, json_content["eval"]):
                i_eval.append(fi)
        
        return np.asarray(i_train), np.asarray(i_eval)




    @staticmethod
    def _get_train_eval_split(frames, eval_mode, train_split_fraction, eval_interval, json_list_path):
        if eval_mode == "fraction":
            return get_train_eval_split_fraction(frames, train_split_fraction)
        elif eval_mode == "interval":
            return get_train_eval_split_interval(frames, eval_interval)
        elif eval_mode == "json":
            return MMSplatDataParser.get_train_eval_split_json(frames)
        elif eval_mode == "json-list":
            return MMSplatDataParser.get_train_eval_split_json_list(frames, json_list_path)
        elif eval_mode == "fraction-groups":
            return MMSplatDataParser.get_train_eval_split_fraction_groups(frames, train_split_fraction)
        elif eval_mode == "all":
            CONSOLE.log(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            return get_train_eval_split_all(frames)
        else:
            raise ValueError(f"Unknown eval mode {eval_mode}")
            


    @staticmethod
    def _get_frame_param(meta, frame, param):
        if param in frame:
            return frame[param]
        elif param in meta:
            return meta[param]
        else:
            raise RuntimeError(f"transforms.json: frame {frame['file_path']} missing {param}")


    @staticmethod
    def _generate_camera_parameters(meta, frames) -> Tuple:
        """Given meta and frames, generate camera information, including poses.
        
        Returns: Tuple(
            fx: List[float]
            fy: List[float]
            cx: List[float]
            cy: List[float]
            w: List[int]
            h: List[int]
            camera_type: List[CameraType]
            distort: List[torch.Tensor]
            poses: List[np.array]
        )
        """

        # results
        fx = []
        fy = []
        cx = []
        cy = []
        w = []
        h = []
        camera_type = []
        distort = []
        poses = []


        # iterate frames and get data
        for frame in frames:
            fx.append(float(MMSplatDataParser._get_frame_param(meta, frame, "fl_x")))
            fy.append(float(MMSplatDataParser._get_frame_param(meta, frame, "fl_y")))

            cx.append(float(MMSplatDataParser._get_frame_param(meta, frame, "cx")))
            cy.append(float(MMSplatDataParser._get_frame_param(meta, frame, "cy")))

            w.append(int(MMSplatDataParser._get_frame_param(meta, frame, "w")))
            h.append(int(MMSplatDataParser._get_frame_param(meta, frame, "h")))

            camera_type.append(CAMERA_MODEL_TO_TYPE[MMSplatDataParser._get_frame_param(meta, frame, "camera_model")])

            poses.append(np.array(frame["transform_matrix"]))

            if "distortion_params" in frame:
                distort.append(torch.tensor(frame["distortion_params"], dtype=torch.float32))
            elif any(key in frame for key in ["k1", "k2", "k3", "k4", "p1", "p2"]):
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )
            elif "distortion_params" in meta:
                distort.append(torch.tensor(meta["distortion_params"], dtype=torch.float32))
            else:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

        # resulting dict
        return fx, fy, cx, cy, w, h, camera_type, distort, poses

            
    @staticmethod
    def _get_mm_channel(frame: Dict[str, Any]) -> str:
        """Given a frame, return the mm channel it belongs to."""

        if (not "mm_channel" in frame) or (not isinstance(frame["mm_channel"], str)):
            return "__default"
        return frame["mm_channel"]

    @staticmethod
    def _select_downscale_factor(
        frames: List[Dict[str, Any]],
        data_dir,
        downscale_factor: Optional[int] = None,
        autoscale_max_res: int = 1600,
        downsample_folder_prefix = "images_"
    ) -> Tuple[int, List[Path]]:
        """Given a list of frames, select a reasonable downscale factor, and return (factor, list_of_image_paths)"""

        # remove first element of path (e.g. images/D/D_01.jpg becomes D/D_01.jpg)
        def _cut_first_path_element(path) -> Path:
            return Path(*Path(path).parts[1:])
        
        checked_channels = set()
        ds_factor = downscale_factor

        if ds_factor is None:
            # downscale factor not set, so we find downscale factor by checking resolutions
            ds_factor = 0

            for frame in frames:
                f_channel = MMSplatDataParser._get_mm_channel(frame)
                filepath = Path(frame["file_path"])

                # skip channel if already checked
                # note: this assumes that all images in a channel are of same or at least comparable sizes.
                #       for performance reasons we dont want to load and check every image to ensure the maximum resolution
                if f_channel in checked_channels:
                    continue

                # load image
                img_data = Image.open(data_dir / filepath)
                max_res = max(img_data.size)

                while True:
                    # check if current ds_factor meets maximum resolution
                    if (max_res / 2 ** (ds_factor)) <= autoscale_max_res:
                        break

                    # check if next downscale level is available
                    if not (data_dir / f"{downsample_folder_prefix}{2**(ds_factor+1)}" / _cut_first_path_element(filepath)).exists():
                        break

                    # downscale
                    ds_factor += 1
        
        # ds_factor is set, so now we generate file paths
        if ds_factor > 0:
            img_paths = [
                data_dir / f"{downsample_folder_prefix}{(2**ds_factor)}" / _cut_first_path_element(frame["file_path"])
                for frame in frames
            ]
        else:
            img_paths = [data_dir / Path(frame["file_path"]) for frame in frames]

        return (ds_factor, img_paths)