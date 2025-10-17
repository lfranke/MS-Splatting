

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.utils.eval_utils import eval_load_checkpoint



def eval_setup(
    config_path: Path,
    load_step: Optional[int] = None,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # set step to load checkpoint for, if specified
    if load_step is not None:
        config.load_step = load_step

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step