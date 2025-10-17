from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Type, Union, Any
import random
import itertools
from concurrent.futures import ThreadPoolExecutor
import re
import more_itertools

import copy
import torch
from rich.progress import track
from pathlib import Path
import numpy as np
from typing_extensions import assert_never

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image

from mmsplat.mmsplat_dataparser import MMSplatDataParserConfig
from mmsplat.util.parallel_for import parallel_for
from mmsplat.util import image_utils









@dataclass
class MMSplatDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: MMSplatDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=MMSplatDataParserConfig)

    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    cache_images: Literal["cpu", "gpu"] = "cpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_image_type: Literal["uint8", "float32", "auto"] = "auto"
    """The image type returned from manager, caching images in uint8 saves memory. 'auto' sets the format per-image based on the image file."""
    max_thread_workers: int = 16
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    camera_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    delay_channels: List[str] = field(default_factory=lambda: [])
    """Specify if channels should be delayed by some amount of iterations in training
    
    Format: [channelA_1] ... [channelA_n] [n_A] [channelB_1] ... [channelB_n] [n_B] ...
            where n_A, n_B ... are non-negative integers defining the delay for the given list of channels preceding them.
    Example: --delay-channels RE NIR 5000 T 10000
            (where RE, NIR and T are channel names)
    """
    stop_channels: List[str] = field(default_factory=lambda: [])
    """Specify if channels should stop training early, and at which iteration
    
    Format: [channelA_1] ... [channelA_n] [n_A] [channelB_1] ... [channelB_n] [n_B] ...
            where n_A, n_B ... are non-negative integers defining the delay for the given list of channels preceding them.
    Example: --stop-channels RE NIR 35000 T 50000
            (where RE, NIR and T are channel names; RE and NIR would stop after 35000 iterations, T would stop after 50000)
    """
    limit_channels: Optional[List[str]] = None
    """List of channels that should be used in training. Ignores all other channels. If None, all channels are used."""

    channel_size: Optional[List[str]] = None
    """List of specific channel sizes. Allowed sizes are 3 for RGB and 1 for grayscale, or <=0 to determine automatically. Also auto-determined if not specified.
    Format: List of channel names followed by number giving the channel size.
    Example: --channel-size D 3 MS_R MS_G MS_RE MS_NIR 1
    """

    channel_fractions: List[str] = field(default_factory=lambda: [])
    """Specify if, for specific channels, only a fraction of the images should be available in training (sampled evenly spread)
    
    Format: [channelA_1] ... [channelA_n] [f_A] [channelB_1] ... [channelB_n] [f_B] ...
            where f_A, f_B ... are floats [0, 1] defining the fraction for the given channels.
    Example: --channel-fractions RE NIR 0.2 T 0.3
            (where RE, NIR and T are channel names; RE and NIR would stop after 35000 iterations, T would stop after 50000)
    """

    channel_oversampling: List[str] = field(default_factory=lambda: [])
    """Specify if some channels should be sampled more or less often. Specified as a float factor:
        0.5: sample these channels half as frequent
        2: sample these channels twice as frequent
        ...
    
    Format: [channelA_1] ... [channelA_n] [f_A] [channelB_1] ... [channelB_n] [f_B] ...
            where f_A, f_B ... are floats >0 defining the sampling factor for the given channels.
    Example: --channel-fractions RE NIR 0.2 T 0.3
            (where RE, NIR and T are channel names; RE and NIR would stop after 35000 iterations, T would stop after 50000)
    """

    equal_channel_sampling: bool = True
    """Whether to sample every channel with the same frequency, or to have channels with more images appear more often during training."""



class MMSplatDataManager(DataManager):
    config: MMSplatDataManagerConfig
    train_data: List[Dict[str, Any]]
    eval_data: List[Dict[str, Any]]


    def __init__(
        self,
        config: MMSplatDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        **kwargs
    ):
        # assign basic params
        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.random_generator = random.Random(self.config.camera_sampling_seed)

        # determine data path
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        
        # setup dataparser
        self.dataparser = self.config.dataparser.setup()
        
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images

        # generate split dataparser outputs
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.eval_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split=self.test_split)

        self.train_dataset = self._create_dataset(self.train_dataparser_outputs)
        self.eval_dataset = self._create_dataset(self.eval_dataparser_outputs)
    

        # overwrite caching method if image count is too high
        total_frame_count = len(self.train_dataset) + len(self.eval_dataset)
        if total_frame_count > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"


        # load images and channel information
        self.train_data = self._create_data_list(self.train_dataset)
        self.eval_data = self._create_data_list(self.eval_dataset)

        # collect all present channels, and limit according to self.config.limit_channels
        self.mm_channel_list = sorted(list(set(
            f["mm_channel"] for f in itertools.chain(self.train_data, self.eval_data)
        )))

        if self.config.limit_channels is not None:
            self.mm_channel_list = [c for c in self.mm_channel_list if c in self.config.limit_channels]


        # for all channels, determine channel size and convert images if necessary
        self.mm_channel_size = []

        for channel in self.mm_channel_list:
            # get size set in config.channel_size
            set_size = 0
            if (self.config.channel_size is not None) and (channel in self.config.channel_size):
                for arg in self.config.channel_size[self.config.channel_size.index(channel)+1:]:
                    try:
                        ls_val = int(arg, base=10)
                        set_size = (0 if (ls_val != 1 and ls_val != 3) else ls_val)
                        break
                    except ValueError:
                        continue
            
            # if size is auto-determine, get max from array
            if set_size == 0:
                set_size = max(f["image"].shape[2] for f in itertools.chain(self.train_data, self.eval_data) if f["mm_channel"] == channel)
                if set_size >= 3:
                    set_size = 3
                else:
                    set_size = 1

            self.mm_channel_size.append(set_size)

            # convert all non-fitting images to set_size
            for f in (f2 for f2 in itertools.chain(self.train_data, self.eval_data) if f2["mm_channel"] == channel):
                if f["image"].shape[2] == set_size:
                    continue

                orig_dtype = f["image"].dtype

                # convert to grayscale
                if set_size == 1:
                    if f["image"].shape[2] == 2: # remove alpha
                        f["image"] = f["image"][:, :, 0:1] 
                    elif f["image"].shape[2] == 3 or f["image"].shape[2] == 4: # convert to grayscale
                        f["image"] = (
                            0.2989 * f["image"][:, :, 0:1]
                            + 0.5870 * f["image"][:, :, 1:2]
                            + 0.1140 * f["image"][:, :, 2:3]
                        )
                    else: # average over elements
                        f["image"] = f["image"].mean(dim=-1, keepdim=True)
                elif set_size == 3:
                    if f["image"].shape[2] <= 2: # repeat from grayscale to RGB
                        f["image"] = f["image"][:, :, 0:1].repeat(1, 1, 3)
                    elif f["image"].shape[2] == 4: # slice out first 3 elements
                        f["image"] = f["image"][:, :, 0:3]
                    else: # average to grayscale and repeat
                        f["image"].mean(dim=-1, keepdim=True).repeat(1, 1, 3)
                
                f["image"] = f["image"].to(orig_dtype)


        # collect channel delay and stop info
        self._init_channel_delays()
        self._init_channel_stops()
        self._init_channel_fractions()
        self._init_channel_oversampling()


        # add channel metadata to datasets
        dataset_metadata = {
            "mm_channel_list": self.mm_channel_list,
            "mm_channel_size": self.mm_channel_size
        }

        if self.train_dataset.metadata is None:
            self.train_dataset.metadata = copy.deepcopy(dataset_metadata)
        else:
            self.train_dataset.metadata.update(dataset_metadata)

        if self.eval_dataset.metadata is None:
            self.eval_dataset.metadata = copy.deepcopy(dataset_metadata)
        else:
            self.eval_dataset.metadata.update(dataset_metadata)


        # generate initial sample indices
        self.next_train_cameras = self._sample_train_cameras(0)
        self.next_eval_cameras = self._sample_eval_cameras(0)
        assert len(self.next_train_cameras) > 0, "No data found in dataset"

        super().__init__()




    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        
        # get next train camera
        while True:
            # refill queue if empty
            if len(self.next_train_cameras) == 0:
                self.next_train_cameras = self._sample_train_cameras(step)

            # get image index
            image_idx = self.next_train_cameras.pop(0)

            # check if image channel is currently in use, if not go to next
            if self._use_channel(self.train_data[image_idx]["mm_channel"], step):
                break

        # get image data dict, and move image to device
        data = self.train_data[image_idx].copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"

        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        camera.metadata["mm_channel"] = data["mm_channel"]

        return camera, data
    

    def all_train(self) -> List[Tuple[Cameras, Dict]]:
        """Generate list for all training cameras."""
        image_indices = [i for i in range(len(self.train_dataset))]

        # copy data
        data = [d.copy() for d in self.train_data]
        
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)

            # copy camera and assign correct metadata
            cam = copy.deepcopy(self.train_dataset.cameras[i : i+1]).to(self.device)
            if cam.metadata is None:
                cam.metadata = {}
            cam.metadata["cam_idx"] = i
            cam.metadata["mm_channel"] = data[i]["mm_channel"]

            cameras.append(cam)
        
        return list(zip(cameras, data))

    

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
            
        # get next eval camera
        while True:
            # refill queue if empty
            if len(self.next_eval_cameras) == 0:
                self.next_eval_cameras = self._sample_eval_cameras(step)

            # get image index
            image_idx = self.next_eval_cameras.pop(0)

            # check if image channel is currently in use, if not go to next
            if self._use_channel(self.eval_data[image_idx]["mm_channel"], step):
                break

        # get image data dict, and move image to device
        data = self.eval_data[image_idx].copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"

        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        camera.metadata["mm_channel"] = data["mm_channel"]

        return camera, data
    

    # -------- in the following section, we define some miscellaneous DataManager abstract methods, only to comply with implementation ----

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """
        Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples
        """
        image_indices = [i for i in range(len(self.eval_dataset))]

        # copy data
        data = [d.copy() for d in self.eval_data]
        
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)

            # copy camera and assign correct metadata
            cam = copy.deepcopy(self.eval_dataset.cameras[i : i+1]).to(self.device)
            if cam.metadata is None:
                cam.metadata = {}
            cam.metadata["cam_idx"] = i
            cam.metadata["mm_channel"] = data[i]["mm_channel"]

            cameras.append(cam)
        
        return list(zip(cameras, data))
    

    def get_datapath(self) -> Path:
        """Returns the path to the data. This is used to determine where to save camera paths."""
        return self.config.dataparser.data
    

    def setup_train(self):
        """Sets up the data manager for training."""

    def setup_eval(self):
        """Sets up the data manager for evaluation"""


    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
    

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        # AFAIK this has no effect if its not exact
        # also, its not clear if the return value should vary based on the upcoming batch
        # for that reason we always return the same value, but try to base it on an actual camera
        
        # return size of first present image data, default to 800*800
        return next((x["image"].shape[0] * x["image"].shape[1] for x in self.train_data if "image" in x), 800*800)

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        # AFAIK this has no effect if its not exact
        # also, its not clear if the return value should vary based on the upcoming batch
        # for that reason we always return the same value, but try to base it on an actual camera
        
        # return size of first present image data, default to 800*800
        return next((x["image"].shape[0] * x["image"].shape[1] for x in self.eval_data if "image" in x), 800*800)





    # -------- implementation of local helper functions --------

    def _init_channel_delays(self):
        """Parse information about channel delays in self.config.delay_channels """
        self.mm_channel_delays = [0 for _ in self.mm_channel_list]

        cur_delay_channels = set()
        for delay_arg in self.config.delay_channels:
            if delay_arg in self.mm_channel_list: # first, check if a known channel name is given
                cur_delay_channels.add(delay_arg)
            elif bool(re.fullmatch(r"[0-9]+", delay_arg)): # second, check if its a non-negative int
                # get delay value, and set delay for current accumulated channels
                delay_val = int(delay_arg)
                for i in range(len(self.mm_channel_list)):
                    if self.mm_channel_list[i] in cur_delay_channels:
                        self.mm_channel_delays[i] = delay_val
                
                # empty channel accumulation
                cur_delay_channels.clear()
            else: # some other string was input, that isnt a known channel. Skip it...
                pass

            
    def _init_channel_stops(self):
        """Parse information about channel stops in self.config.stop_channels """
        self.mm_channel_stops = [-1 for _ in self.mm_channel_list]

        cur_stop_channels = set()
        for stop_arg in self.config.stop_channels:
            if stop_arg in self.mm_channel_list: # first, check if a known channel name is given
                cur_stop_channels.add(stop_arg)
            elif bool(re.fullmatch(r"[0-9]+", stop_arg)): # second, check if its a non-negative int
                # get delay value, and set delay for current accumulated channels
                stop_val = int(stop_arg)
                for i in range(len(self.mm_channel_list)):
                    if self.mm_channel_list[i] in cur_stop_channels:
                        self.mm_channel_stops[i] = stop_val
                
                # empty channel accumulation
                cur_stop_channels.clear()
            else: # some other string was input, that isnt a known channel. Skip it...
                pass

    def _init_channel_fractions(self):
        """Parse information about channel fractions in self.config.channel_fractions """
        self.mm_channel_fractions = [1.0 for _ in self.mm_channel_list]

        cur_channel_fractions = set()
        for fraction_arg in self.config.channel_fractions:
            if fraction_arg in self.mm_channel_list: # first, check if a known channel name is given
                cur_channel_fractions.add(fraction_arg)
            elif bool(re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", fraction_arg)): # second, check if its a non-negative float
                # get fraction value, and set fraction for current accumulated channels
                fraction_val = float(fraction_arg)
                for i in range(len(self.mm_channel_list)):
                    if self.mm_channel_list[i] in cur_channel_fractions:
                        self.mm_channel_fractions[i] = min(1.0, fraction_val)
                
                # empty channel accumulation
                cur_channel_fractions.clear()
            else: # some other string was input, that isnt a known channel. Skip it...
                pass

    def _init_channel_oversampling(self):
        """Parse information about channel oversampling in self.config.channel_oversampling """
        self.mm_channel_oversampling = [1.0 for _ in self.mm_channel_list]

        cur_channel_oversampling = set()
        for fraction_arg in self.config.channel_oversampling:
            if fraction_arg in self.mm_channel_list: # first, check if a known channel name is given
                cur_channel_oversampling.add(fraction_arg)
            elif bool(re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", fraction_arg)): # second, check if its a non-negative float
                # get fraction value, and set fraction for current accumulated channels
                fraction_val = float(fraction_arg)
                for i in range(len(self.mm_channel_list)):
                    if self.mm_channel_list[i] in cur_channel_oversampling:
                        self.mm_channel_oversampling[i] = fraction_val
                
                # empty channel accumulation
                cur_channel_oversampling.clear()
            else: # some other string was input, that isnt a known channel. Skip it...
                pass




    def _use_channel(self, channel_name, step):
        """Check whether a certain channel should be used or not at a certain iteration.
        See delay_channels and _init_channel_delays above."""
        
        for ch, ch_delay, ch_stop in zip(self.mm_channel_list, self.mm_channel_delays, self.mm_channel_stops):
            if ch == channel_name:
                # check if channel stop is set and reached
                if ch_stop >= 0 and ch_stop < step:
                    return False
                
                # check if channel delay is reached
                return bool(step >= ch_delay)

        # this channel name is not present in the configuration, so we say it not be used
        return False
    

    def _sample_train_cameras(self, step):
        return self._sample_cameras(
            data=self.train_data,
            channel_list=self.mm_channel_list,
            random_generator=self.random_generator,
            step=step,
            channel_fractions=(self.mm_channel_fractions if any(f < 1.0 for f in self.mm_channel_fractions) else None),
            equalize_channels=self.config.equal_channel_sampling,
            channel_oversampling=(self.mm_channel_oversampling if any(f != 1.0 for f in self.mm_channel_oversampling) else None)
        )
    def _sample_eval_cameras(self, step):
        return self._sample_cameras(
            data=self.eval_data,
            channel_list=self.mm_channel_list,
            step=step,
            channel_fractions=None,
            equalize_channels=False,
            channel_oversampling=None
        )

    def _sample_cameras(
        self,
        data,
        channel_list,
        random_generator=None,
        step=0,
        equalize_channels=False, # whether to stretch all channels to the same number of images
        channel_fractions=None, # array of channel fractions or None
        channel_oversampling=None, # array of channel oversampling fractions or None
    ) -> List[int]:
        """Generate index list into cameras"""

        # collect per-channel lists
        mm_indices = [
            [i for i,f in enumerate(data) if f["mm_channel"] == channel]
            for channel in channel_list
        ]

        # apply fractions if set
        if channel_fractions:
            for i in range(len(mm_indices)):
                if channel_fractions[i] <= 0.0:
                    mm_indices[i] = []
                elif channel_fractions[i] < 1.0 and len(mm_indices[i]) > 0:
                    new_len = max(1, int(len(mm_indices[i]) * channel_fractions[i]))
                    # Use np.linspace for deterministic, evenly spread sampling
                    selected_idx = np.linspace(0, len(mm_indices[i]) - 1, new_len, dtype=int)
                    mm_indices[i] = [mm_indices[i][j] for j in selected_idx]

        # shuffle per-channel lists
        if random_generator is None:
            for c_list in mm_indices:
                random.shuffle(c_list)
        else:
            for c_list in mm_indices:
                random_generator.shuffle(c_list)

        if equalize_channels:
            # extend all sub-lists to the same length by repeating
            max_len = max(len(c_list) for c_list in mm_indices)
            for c_list in mm_indices:
                if len(c_list) == 0:
                    continue

                repeat_count = -(-max_len // len(c_list))  # Ceiling division
                c_list[:] = (c_list * repeat_count)[:max_len]

        if channel_oversampling:
            # over-sample the specified channels
            for i in range(len(mm_indices)):
                if len(mm_indices[i]) == 0:
                    continue

                if channel_oversampling[i] > 1.0:
                    new_len = int(len(mm_indices[i]) * channel_oversampling[i])
                    repeat_count = -(-new_len // len(mm_indices[i]))  # Ceiling division
                    mm_indices[i][:] = (mm_indices[i] * repeat_count)[:new_len]
                elif channel_oversampling[i] < 1.0:
                    new_len = max(0, int(len(mm_indices[i]) * channel_oversampling[i]))
                    mm_indices[i] = mm_indices[i][:new_len]

        # interleave per-channel lists
        indices = list(more_itertools.interleave_evenly(mm_indices))

        return indices



    def _create_dataset(self, dataparser_outputs: DataparserOutputs) -> InputDataset:
        return InputDataset(dataparser_outputs, self.config.camera_res_scale_factor)


    def _create_data_list(self, dataset: InputDataset) -> List[Dict[str, Any]]:
        """Sets up a data list given the dataparser outputs."""

        # enter each frame into data list
        data = []

        for i in range(len(dataset)):
            mm_channel = dataset.metadata["mm_channel"][i] if "mm_channel" in dataset.metadata else "__default"
            data.append({
                "mm_channel": mm_channel
            })

        # load images
        self._load_images(dataset, data)

        return data




    def _load_images(
        self,
        dataset: InputDataset,
        data: List[Dict[str, Any]],
        split_name = ""
    ):
        """Load all images in dataset. Cache to device self.config.cache_images in format self.config.cache_images_type"""

        # function to load image of a single frame in dataset
        def load_and_undistort_frame(frame_idx):
            data_frame = data[frame_idx]
            filepath = dataset.image_filenames[frame_idx]

            # load data, scale and convert type
            img_data = image_utils.load_image_data(
                filepath,
                self.config.camera_res_scale_factor,
                self.config.cache_image_type
            )

            # TODO: remove/replace alpha channel if present
            # TODO: for every channel, ensure that all images have the same number of subchannels

            # the following is only to test with SplatfactoModel, since it doesnt support grayscale image data
            #if img_data.shape[2] == 1:
            #    img_data = img_data.repeat(3, axis=2)

            # validate result
            if img_data is None:
                print(f"Failed to load image from file {filepath}")
                return

            # change format if necessary, since torch only supports uint8 or float32 for our application
            if self.config.cache_image_type == "auto" and img_data.dtype != np.uint8:
                img_data = image_utils.convert_image_type(img_data, "float32")
                if img_data is None:
                    print(f"Failed to convert image from file {filepath}")
                    return
            
            # check correct size
            camera = dataset.cameras[frame_idx].reshape(())
            assert img_data.shape[1] == camera.width.item() and img_data.shape[0] == camera.height.item(), (
                f'The size of image ({img_data.shape[1]}, {img_data.shape[0]}) loaded '
                f'does not match the camera parameters ({camera.width.item(), camera.height.item()})'
            )

            # undistort if necessary
            if (camera.distortion_params is not None) and torch.any(camera.distortion_params != 0):
                K = camera.get_intrinsics_matrices().numpy()
                distortion_params = camera.distortion_params.numpy()

                prev_type = img_data.dtype

                K, img_data, _ = _undistort_image(camera, distortion_params, {}, img_data, K)

                # ensure that image type is still correct
                if prev_type != img_data.dtype:
                    print("_undistort_image changed image type, converting back...")
                    img_data = image_utils.convert_image_type(img_data, prev_type.name)
                
                # add last dimension again for grayscale
                if len(img_data.shape) == 2:
                    img_data = img_data[:, :, np.newaxis]

                # adjust camera parameters
                dataset.cameras.fx[frame_idx] = float(K[0, 0])
                dataset.cameras.fy[frame_idx] = float(K[1, 1])
                dataset.cameras.cx[frame_idx] = float(K[0, 2])
                dataset.cameras.cy[frame_idx] = float(K[1, 2])
                dataset.cameras.width[frame_idx] = img_data.shape[1]
                dataset.cameras.height[frame_idx] = img_data.shape[0]

            # create image tensor
            data_frame["image"] = torch.from_numpy(img_data)
            data_frame["filepath"] = Path(filepath)
            data_frame["name"] = Path(filepath).name

            if self.config.cache_images == "gpu":
                data_frame["image"] = data_frame["image"].to(self.device)
            elif self.config.cache_images == "cpu":
                data_frame["image"] = data_frame["image"].pin_memory()
            else:
                assert_never(self.config.cache_images)


        # run load_and_undistort_frame for all frames
        CONSOLE.log(f"[yellow]Caching / undistorting {len(dataset)} images ({split_name})")
        parallel_for(
            load_and_undistort_frame,
            list(range(len(data))),
            max(self.config.max_thread_workers, 1)
        )