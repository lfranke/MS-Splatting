from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from time import time

import torch
import torchvision.utils as vutils
from pathlib import Path
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler




@dataclass
class MMSplatPipelineConfig(VanillaPipelineConfig):
    """MMSplatPipelineConfig"""

    _target: Type = field(default_factory=lambda: MMSplatPipeline)
    """target class to instantiate"""



class MMSplatPipeline(VanillaPipeline):
    """Pipeline for mmsplat training

    All the same as in VanillaPipeline, only fixes a problem when building average metrics,
    since in mmsplat models, not all iterations create the same metrics, but channel-specific metrics are built
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        NOTE: Fully copied from VanillaPipeline (nerfstudio 1.1.4), but fixed the calculation of mean (at bottom)

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                if outputs is None:
                    continue
                
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        #vutils.save_image(
                        #    image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        #)

                        import os
                        import numpy as np
                        import PIL.Image as Image

                        if image.shape[2] == 1:
                            image = image.expand(-1, -1, 3)

                        img = Image.fromarray((image.cpu().numpy()  * 255).astype(np.uint8))

                        if not os.path.exists(os.path.join(output_path, key)):
                            os.makedirs(os.path.join(output_path, key), exist_ok=True)

                        img.save(os.path.join(output_path, key, f"{image_prefix}_{key}_{idx:04d}.png"))
                        #np.save(os.path.join(output_path, key, f"{image_prefix}_{key}_{idx:04d}.npy"), image_dict[key].cpu().numpy() )


                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        # generate list of all present metric names
        metrics_keys = sorted(list(set(key for md in metrics_dict_list for key in md.keys())))

        metrics_dict = {}
        for key in metrics_keys:
            # collect all values to the current metric
            metric_list = [md[key] for md in metrics_dict_list if key in md]

            if get_std:
                key_std, key_mean = torch.std_mean(torch.tensor(metric_list))
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(torch.mean(torch.tensor(metric_list)))

        self.train()
        return metrics_dict
