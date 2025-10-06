from typing import List, Optional, Tuple, Union, cast, Literal
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
import os
import copy

from nerfstudio.scripts.exporter import Exporter, ExportGaussianSplat
from nerfstudio.models.splatfacto import RGB2SH, SH2RGB
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.field_components.encodings import NeRFEncoding, FFEncoding

from mmsplat.mmsplat_model import MMSplatModel
from mmsplat.util.eval_utils import eval_setup










@dataclass
class ExportMMSplat(Exporter):
    """Export 3D Gaussian Splatting models from MMSplatModel to .ply files"""

    load_step: Optional[int] = None
    """Step for which to load the checkpointl, or None if last checkpoint should be loaded"""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb", "auto", "mlp"] = "mlp"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""
    channels: Optional[List[str]] = None
    """List of channel names for which to export .ply files. If None, all channels will be exported."""



    def export_channel_ply(
        self,
        channel_info: str,
        model: MMSplatModel,
        filename: str
    ):
        """Given the channel info object from the model, export a .ply file for the given channel to the given filename."""

        # determine export color mode
        color_mode = self.ply_color_mode
        if color_mode == "auto":
            if channel_info.use_sh:
                color_mode = "sh_coeffs"
            else:
                color_mode = "rgb"

        # collect output ply value lists
        outputs_list = OrderedDict()

        with torch.no_grad():
            # get tensors
            positions = model.gauss_params["means"].data.cpu().numpy()
            scales = model.gauss_params["scales"].data.cpu().numpy()
            quats = model.gauss_params["quats"].data.cpu().numpy()
            opacities = model.gauss_params["opacities"].data.cpu().numpy()
            feature_embedding = model.gauss_params["feature_embedding"]
            g_count = positions.shape[0]

            # slice the correct elements for colors/sh
            features_dc = model.gauss_params["features_dc"][:, channel_info.fdc_index : (channel_info.fdc_index + channel_info.size)]
            f_rest_size = channel_info.size if channel_info.use_sh else 0
            features_rest = model.gauss_params["features_rest"][:, :, channel_info.frest_index : (channel_info.frest_index + f_rest_size)]

            # add positions and normals to output list
            outputs_list["x"] = positions[:, 0]
            outputs_list["y"] = positions[:, 1]
            outputs_list["z"] = positions[:, 2]
            outputs_list["nx"] = np.zeros(g_count, dtype=np.float32)
            outputs_list["ny"] = np.zeros(g_count, dtype=np.float32)
            outputs_list["nz"] = np.zeros(g_count, dtype=np.float32)

            # add colors or sh coefficients
            if color_mode == "rgb":
                if channel_info.use_sh:
                    rgbs = torch.clamp(SH2RGB(features_dc), 0.0, 1.0).data.cpu().numpy()
                    if model.config.sh_degree > 0:
                        CONSOLE.log(f"[yellow]Warning: Channel {channel_info.name} has higher level of spherical harmonics, ignoring them and only export rgb.")
                else:
                    rgbs = torch.clamp(torch.sigmoid(features_dc), 0.0, 1.0).data.cpu().numpy()

                # expand to at least 3 channels
                if rgbs.shape[1] < 3:
                    rgbs = rgbs.repeat(3, axis=1)

                # potentially convert to uint8?
                rgbs = rgbs.astype(np.float32)

                # add rgb to output
                outputs_list["red"] = rgbs[:, 0]
                outputs_list["green"] = rgbs[:, 1]
                outputs_list["blue"] = rgbs[:, 2]

            elif color_mode == "sh_coeffs":
                # force 3 channels
                if features_dc.shape[1] < 3:
                    features_dc = features_dc.expand(-1, 3)
                    if features_rest.shape[2] > 0:
                        features_rest = features_rest.expand(-1, -1, 3)

                # add f_dc to output list
                if channel_info.use_sh:
                    shs_0 = features_dc.data.cpu().numpy()
                else:
                    shs_0 = RGB2SH(torch.sigmoid(features_dc)).data.cpu().numpy()

                for i in range(shs_0.shape[1]):
                    outputs_list[f"f_dc_{i}"] = shs_0[:, i]

                # add f_rest to output list; transpose needed to match ply formatting
                shs_rest = features_rest.transpose(1, 2).data.cpu().numpy()
                shs_rest = shs_rest.reshape((g_count, -1))

                for i in range(shs_rest.shape[-1]):
                    outputs_list[f"f_rest_{i}"] = shs_rest[:, i, None]

            elif color_mode == "mlp":

                input_vec = feature_embedding

                if model.config.direction_encoding_flag:

                    direction_encoding_in = torch.nn.functional.normalize(torch.rand(positions.shape))
                    direction_encoding_out = self.direction_encoding.forward(direction_encoding_in)

                    input_vec = torch.cat((input_vec, direction_encoding_out.cuda()), axis=1)

                if model.config.positional_encoding_flag:
                    positional_encoding_in = model.gauss_params["means"].data
                    positional_encoding_out = self.positional_encoding.forward(positional_encoding_in)

                    input_vec = torch.cat((input_vec, positional_encoding_out.cuda()), axis=1)

                # TODO: knn neighbouring color prediction MLP

                #input_vec = torch.zeros((feature_embedding.shape[0], model.feature_mlp.fc1.weight.shape[1])).cuda()
                #input_vec[:, :feature_embedding.shape[1]] = feature_embedding

                colors = model.feature_mlp.forward(input_vec)
                colors = colors.data.cpu().numpy()


                if channel_info.name == "D":
                    colors = (colors[:, [0, 1, 2]] * 255).astype(np.uint8)
                    outputs_list["red"] = colors[:, 0]
                    outputs_list["green"] = colors[:, 1]
                    outputs_list["blue"] = colors[:, 2]
                elif channel_info.name == "MS_R":
                    colors = (colors[:, [3]].repeat(3, 1)  * 255).astype(np.uint8)
                    #outputs_list["MS_R"] = colors[:, 0]
                    outputs_list["red"] =colors[:, 0]
                    outputs_list["green"] = colors[:, 0]
                    outputs_list["blue"] = colors[:, 0]
                elif channel_info.name == "MS_G":
                    colors = (colors[:, [4]].repeat(3, 1)  * 255).astype(np.uint8)
                    #outputs_list["MS_G"] = colors[:, 0]
                    outputs_list["red"] =colors[:, 0]
                    outputs_list["green"] = colors[:, 0]
                    outputs_list["blue"] = colors[:, 0]
                elif channel_info.name == "MS_RE":
                    colors = (colors[:, [5]].repeat(3, 1)  * 255).astype(np.uint8)
                    #outputs_list["MS_RE"] = colors[:, 0]
                    outputs_list["red"] =colors[:, 0]
                    outputs_list["green"] = colors[:, 0]
                    outputs_list["blue"] = colors[:, 0]
                elif channel_info.name == "MS_NIR":
                    colors = (colors[:, [6]].repeat(3, 1)  * 255).astype(np.uint8)
                    #outputs_list["MS_NIR"] = colors[:, 0]
                    outputs_list["red"] =colors[:, 0]
                    outputs_list["green"] = colors[:, 0]
                    outputs_list["blue"] = colors[:, 0]
                elif channel_info.name == 'features':
                    for i in range(feature_embedding.shape[1]):
                        outputs_list["feature_embedding_{}".format(i)] = np.zeros(g_count, dtype=np.float32)
                        outputs_list["feature_embedding_{}".format(i)] = feature_embedding.data.cpu().numpy()[:, i]

            # add opacities
            outputs_list["opacity"] = opacities

            # add scales
            for i in range(3):
                outputs_list[f"scale_{i}"] = scales[:, i, None]

            # add quats
            for i in range(4):
                outputs_list[f"rot_{i}"] = quats[:, i, None]

            # apply bounding box, if necessary
            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                mask = crop_obb.within(torch.from_numpy(positions)).numpy()
                for k, t in outputs_list.items():
                    outputs_list[k] = outputs_list[k][mask]

                g_count = outputs_list["x"].shape[0]

            
        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(g_count, dtype=bool)
        for k, t in outputs_list.items():
            n_before = np.sum(select)
            select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < g_count:
            CONSOLE.print(f"values have NaN/Inf in outputs_list, only export {np.sum(select)}/{g_count}")
            for k, t in outputs_list.items():
                outputs_list[k] = outputs_list[k][select]
            g_count = np.sum(select)
                
        ExportGaussianSplat.write_ply(str(filename), g_count, outputs_list)
                




    def main(self) -> None:
        # TODO: detect if model is MLP or SH automatically

        # ensure that output dir exists
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # get pipeline and model
        _, pipeline, _, _ = eval_setup(self.load_config, self.load_step)

        assert isinstance(pipeline.model, MMSplatModel)
        model: MMSplatModel = pipeline.model

        channel_list = [c for c in model.mm_channels if self.channels is None or c.name in self.channels]

        if self.ply_color_mode == 'mlp':
            features = copy.copy(channel_list[0])
            features.name = 'features'
            channel_list.append(features)

            self.positional_encoding = NeRFEncoding(in_dim=3,
                                                    num_frequencies=model.config.positional_encoding_num_frequencies,
                                                    min_freq_exp=0.0,
                                                    max_freq_exp=8.0,
                                                    include_input=True)

            self.direction_encoding = NeRFEncoding(in_dim=3,
                                                   num_frequencies=model.config.direction_encoding_num_frequencies,
                                                   min_freq_exp=0.0,
                                                   max_freq_exp=8.0,
                                                   include_input=True)


        for channel in channel_list:
            filename = os.path.join(self.output_dir, f"{channel.name}.ply")
            self.export_channel_ply(channel, model, filename)

