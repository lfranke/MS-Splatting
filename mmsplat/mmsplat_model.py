from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any
from types import SimpleNamespace
import traceback
import copy
import time

from pathlib import Path
import torch
import numpy as np
import open3d.ml.torch as ml3d

from gsplat.rendering import rasterization

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models import splatfacto
from nerfstudio.utils import colors
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.viewer_elements import ViewerDropdown
from nerfstudio.field_components.encodings import NeRFEncoding, FFEncoding
from nerfstudio.viewer.viewer_elements import *

from mmsplat.gsplat.strategy import DefaultStrategy, MMMaxAverageStrategy
from mmsplat.mmsplat_neural_surface import MultiSpectralFeatureDecoder
from mmsplat.util.memory import *
from mmsplat.util.band_indices import *
from mmsplat.mmsplat_camera_optimizer import CameraOptimizer, CameraOptimizerConfig
from mmsplat.util.loss import smoothness_loss, unit_norm_regularization_loss, cosine_neighbor_loss_batched
from mmsplat.util.utils import get_viewmat

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.mean()



@dataclass
class MMSplatModelConfig(ModelConfig):
    """MMSplat Model Config"""

    _target: Type = field(default_factory=lambda: MMSplatModel)

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 5000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 50000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_sh_channels: List[str] = field(default_factory=lambda: [])
    """list of channel names for which spherical harmonics (of degree sh_degree) should be used"""
    use_mlp_channels: List[str] = field(default_factory=lambda: [])
    """list of channel names for which mlp color should be used"""
    use_scale_regularization: bool = True
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer_rgb: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the rgb camera optimizer to use"""
    camera_optimizer_ms: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the ms camera optimizer to use"""
    rgb_channel_names: List[str] = field(default_factory=lambda: ["D"])
    """List of names of channels which categorize as RGB for the purposes of the camera optimizers. E.g. if a channel is in this list, its cameras will be optimized with camera_optimizer_rgb, else with camera_optimizer_ms"""
    densification_strategy: Literal["standard", "max_average"] = "standard"
    """What strategy to choose for densification"""
    densification_pause_iterations: List[int] = None
    """iteration durations, for which densification and opacity reset is paused.
    Given as pair-wise [start, stop) numbers, for example:
        --densification-pause-iterations 10000 20000 22000 23000
    pauses densification between iteration 10000 and 20000, and again between iteration 22000 and 23000
    start is exclusive, stop is inclusive
    """
    export_optimized_cameras: Optional[Path] = None
    """File to which to export the optimized camera transforms after training"""
    color_corrected_channels: List[str] = field(default_factory=lambda: [])
    """List of channels for which to apply color correction to the rendered images before computing the metrics."""
    pos_optim_delay_channels: List[str] = field(default_factory=lambda: [])
    """List of channels that only affect the means, scales and quats during training after a delay of iterations
    
    Format: [channelA_1] ... [channelA_n] [n_A] [channelB_1] ... [channelB_n] [n_B] ...
            where n_A, n_B ... are integers defining the delay for the given list of channels preceding them.
            If the delay is negative, the associated channels will never affect the gaussian positions
    Example: --pos_optim_delay_channels NIR RE 5000 T 10000
                (where NIR, RE and T are channel names)
    """
    opacity_optim_delay_channels: List[str] = field(default_factory=lambda: [])
    """List of channels that only affect the gaussian opacities during training after a delay of iterations
    
    Format: [channelA_1] ... [channelA_n] [n_A] [channelB_1] ... [channelB_n] [n_B] ...
            where n_A, n_B ... are integers defining the delay for the given list of channels preceding them.
            If the delay is negative, the associated channels will never affect the gaussian opacities
    Example: --opacity_optim_delay_channels NIR RE 5000 T 10000
                (where NIR, RE and T are channel names)
    """

    mlp_type: Literal["standard", "post"] = "standard"
    """How the MLP operates (if used).
        standard: The MLP predicts per-splat colors before rasterization
        post: The splat feature vectors are rasterized like they are colors; and the MLP predicts colors from the rendered feature vectors per-pixel.
                This method was proposed by FeatSplat
    """
    mlp_post_pixel_embedding: bool = False
    """Whether with mlp_type==post, the pixel position should be appended to the feature vector."""
    mlp_post_camera_position_embedding: bool = False
    """Whether with mlp_type==post, the camera position should be appended to the feature vector."""
    mlp_post_viewing_direction: bool = False
    """Whether with mlp_type==post, the camera viewing direction should be appended to the feature vector."""

    feature_input_dim: int = 16
    """Dimension of the input feature vector to store color representation and to ffed into the feature decoding mlp.
    Set to 0 to disable feature embeddings, and only encode gaussian color via positional and directional encoding."""
    mlp_hidden_depth: int = 16
    """Number of neurons per hidden layer in the feature MLP. Default and minimum is 32."""
    mlp_hidden_layers: int = 1
    """Number of hidden layers in the feature MLP. Default and minimum is 1."""
    mlp_hidden_activation_fn: str = "ELU"
    """Activation function for hidden layers in the feature MLP. Default is SiLU. Options: "ReLU","LeakyReLU","SiLU","GELU","Tanh","Sigmoid","ELU"."""
    direction_encoding_flag: bool = True
    """ Flag to enable encoding of view-directional information (also applies to mlp_type==post)"""
    direction_encoding_num_frequencies: int = 10
    """ Num frequencies for view-directional encoding"""
    positional_encoding_flag: bool = True
    """Flag to enable positional encoding of spatial coordinates"""
    positional_encoding_num_frequencies: int = 10
    """ Num frequencies for positional encoding"""
    opacity_correction_flag: bool = False
    """Flag to activate opacity correction for improved transparency handling"""
    opacity_correction_channels: List[str] = field(default_factory=lambda: ["MS_G", "MS_R", "MS_RE", "MS_NIR"])
    """Channels for which opacity correction is enabled"""
    use_feature_norm_regularization: bool = True
    """Flag to activate normalization regularization on feature vectors"""
    lambda_norm:float = 0.1
    """lambda for normalization regularization on feature vectors"""
    use_smoothness_loss:bool = False
    """Activation for smoothness loss for thermal images (MS_T)"""
    lambda_smoothness:float = 0.1
    """lambda for regularization on thermal smoothness loss"""
    use_neighbouring_features: bool = False
    """Use the features of the k-nearest neighbours, BUT optimize only center one"""
    use_cosine_features: bool = False
    """Pul nearby features together by minimizing cosine similarity for nearby neighbours"""
    lambda_cosine:float = 0.1
    """lambda for regularization on cosine sim on neighbours"""
    knn_size: int = 8
    """Number of neighbouring features used for both use_cosine_features and use_neighbouring_features"""

    vi_red: str = "MS_R"
    """Name of the RED color channel for calculation of vegetation indices """
    vi_green: str = "MS_G"
    """Name of the GREEN color channel for calculation of vegetation indices """
    vi_rededge: str = "MS_RE"
    """Name of the RED-EDGE color channel for calculation of vegetation indices """
    vi_nir: str = "MS_NIR"
    """Name of the NEAR-INFRARED color channel for calculation of vegetation indices """



class MMSplatModel(Model):
    """mmsplat Model"""

    config: MMSplatModelConfig
    


    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        self.camera_optimizer_ms = None
        self.camera_optimizer_rgb = None
        self.feature_mlp = None
        self.positional_encoding = None
        self.direction_encoding = None
        self.seed_points = seed_points # (Location, Color)
        
        # validate necessary data from DataManager in metadata
        assert "mm_channel_list" in metadata, "metadata must include list of channels, mm_channel_list"
        assert "mm_channel_size" in metadata, "metadata must include list of channel sizes, mm_channel_size"
        assert len(metadata["mm_channel_list"]) > 0, "metadata.mm_channel_list is empty"
        assert len(metadata["mm_channel_size"]) > 0, "metadata.mm_channel_size is empty"
        assert len(metadata["mm_channel_size"]) >= len(metadata["mm_channel_list"]), "metadata.mm_channel_size cannot have less elements than metadata.mm_channel_list"

        # generate channel info list
        self.mm_channels = [
            SimpleNamespace(
                index=i,
                name=channel,
                size=metadata["mm_channel_size"][i]
            )
            for i, channel in enumerate(metadata["mm_channel_list"])
        ]

        # set options for viewer dropdown
        channel_options = [c.name for c in self.mm_channels]
        self.band_indices = list(BAND_INDICES.keys())
        channel_options.extend(self.band_indices)

        # enable viewer dropdown to choose channel
        self.default_channel = ViewerDropdown(
            name="mm_channel",
            default_value=self.mm_channels[0].name,
            options=channel_options
        )

        self.feature_channel = ViewerCheckbox(
            name="Feature Visualization",
            default_value=False,
            cb_hook=self.cb_refresh_random_projection
        )

        self.feature_colors = ViewerCheckbox(
            name="Feature Colors",
            default_value=False,
        )


        super().__init__(*args, **kwargs)



    def cb_refresh_random_projection(self, handle) -> None:
        self.refresh_projection_matrix()

    def refresh_projection_matrix(self):
        self.proj_vec = torch.randn((self.config.feature_input_dim, 3)).cuda()


    def populate_modules(self):
        """Initialize tensors"""

        # ensure appropriate params
        assert self.config.feature_input_dim >= 0, "feature_input_dim cannot be negative"

        # viewer selections for feature visualization
        if self.config.feature_input_dim > 0:
            self.feature_color_channel_0 = ViewerDropdown(
                name="Color_channel_0",
                default_value="0",
                options=[str(i) for i in range(0, self.config.feature_input_dim)]
            )

            self.feature_color_channel_1 = ViewerDropdown(
                name="Color_channel_1",
                default_value=("1" if self.config.feature_input_dim > 1 else "0"),
                options=[str(i) for i in range(0, self.config.feature_input_dim)]
            )

            self.feature_color_channel_2 = ViewerDropdown(
                name="Color_channel_2",
                default_value=("2" if self.config.feature_input_dim > 2 else "0"),
                options=[str(i) for i in range(0, self.config.feature_input_dim)]
            )


        # init additional channel info
        self._init_config_params()

        # first, populate means
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)

        num_points = means.shape[0]

        # use average to three nearest neighbors as initial scale
        distances, _ = MMSplatModel.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))

        # generate random quaternions
        quats = torch.nn.Parameter(splatfacto.random_quat_tensor(num_points))

        # determine number of sh bases for channels that use spherical harmonics
        dim_sh = splatfacto.num_sh_bases(self.config.sh_degree)

        # counters for indices into feature (color and MLP) vectors
        next_fdc_index = 0
        next_frest_index = 0
        next_mlpout_index = 0

        # count up the number of channels that use opacity correction
        opacity_correction_count = 0

        # collect information for sh
        for channel in self.mm_channels:
            # check if this channel should use spherical harmonics
            channel.use_sh = (channel.name in self.config.use_sh_channels)
            # check if this channel should use feature embedding for color representation
            channel.use_mlp = (channel.name in self.config.use_mlp_channels)

            # temporary: block MLP implementation
            if channel.use_mlp:
                raise NotImplementedError(f"MLP implementation is not yet available.")

            # set indices
            channel.fdc_index = next_fdc_index
            channel.frest_index = next_frest_index
            channel.mlpout_index = next_mlpout_index

            # increment next_*
            next_fdc_index += channel.size
            if channel.use_sh:
                next_frest_index += channel.size
            if channel.use_mlp:
                next_mlpout_index += channel.size

            # set opacity correction indices
            if self.config.opacity_correction_flag and (channel.name in self.config.opacity_correction_channels):
                channel.opacity_correction = True
                channel.opacity_correction_index = opacity_correction_count
                opacity_correction_count += 1
            else:
                channel.opacity_correction = False
                channel.opacity_correction_index = -1


        # create features_dc and features_rest with appropriate size
        features_dc = torch.nn.Parameter(torch.rand(num_points, next_fdc_index))
        features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, next_frest_index)))

        # initialize opacities
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))


        # MLP (Note: self.config.feature_input_dim can be 0)
        feature_embedding = torch.nn.Parameter(torch.normal(mean=0.0, std=0.2 * torch.ones((num_points, self.config.feature_input_dim))))

        # Default output dim: Sum up channel size for all channels that use MLP
        output_dim_mlp = sum(c.size for c in self.mm_channels if c.use_mlp)

        # additional input values for e.g. opacity
        additional_input_dim = 0

        if output_dim_mlp > 0: # check if MLP is necessary
            # mlp-type dependant embeddings
            if self.config.mlp_type == "post":
                if self.config.mlp_post_viewing_direction:
                    additional_input_dim += 3

                if self.config.mlp_post_camera_position_embedding:
                    additional_input_dim += 3

                if self.config.mlp_post_pixel_embedding:
                    additional_input_dim += 2

            else: # self.config.mlp_type == "standard"
                if self.config.direction_encoding_flag:
                    self.direction_encoding = NeRFEncoding(in_dim=3,  # ToDo: try FFEncoding
                                                        num_frequencies=self.config.direction_encoding_num_frequencies,
                                                        min_freq_exp=0.0,
                                                        max_freq_exp=8.0,
                                                        include_input=True)
                    additional_input_dim += self.direction_encoding.in_dim * self.direction_encoding.num_frequencies * 2 + self.direction_encoding.in_dim
                    
                if self.config.positional_encoding_flag:
                    self.positional_encoding = NeRFEncoding(in_dim=3,  # ToDo: try FFEncoding
                                                            num_frequencies=self.config.positional_encoding_num_frequencies,
                                                            min_freq_exp=0.0,
                                                            max_freq_exp=8.0,
                                                            include_input=True)
                    additional_input_dim += self.positional_encoding.in_dim * self.positional_encoding.num_frequencies * 2 + self.positional_encoding.in_dim


        feature_input_dim = self.config.feature_input_dim + additional_input_dim

        if self.config.mlp_type == "standard" and (self.config.use_neighbouring_features or self.config.use_cosine_features):
            self.k = self.config.knn_size
            self.knnsearch = ml3d.layers.KNNSearch(return_distances=True)
            self.knn_index_reshape = None

            if self.config.use_neighbouring_features:
                feature_input_dim = self.k * feature_input_dim
                self.knn_feature_input_dim = feature_input_dim

        # Create MLP
        self.feature_mlp = MultiSpectralFeatureDecoder(input_dim=feature_input_dim,
                                                       output_dim=output_dim_mlp,
                                                       hidden_depth=self.config.mlp_hidden_depth,
                                                       hidden_layers=self.config.mlp_hidden_layers,
                                                       hidden_activation_function=self.config.mlp_hidden_activation_fn,
                                                       opacity_correction_count=opacity_correction_count,
                                                       mlp_mode=(output_dim_mlp > 0))
        self.feature_mlp.cuda()


        # generate parameter dict
        self.gauss_params = torch.nn.ParameterDict({
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "feature_embedding": feature_embedding,
        })

        # crop area placeholder
        self.crop_box: Optional[OrientedBox] = None

        # setup camera optimizers
        self.camera_optimizer_rgb: CameraOptimizer = self.config.camera_optimizer_rgb.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.camera_optimizer_ms: CameraOptimizer = self.config.camera_optimizer_ms.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # init metrics
        self.psnr2 = torch.nn.ModuleList([PeakSignalNoiseRatio(data_range=1.0) for _ in self.mm_channels])
        self.psnr = [psnr for _ in self.mm_channels]
        self.ssim = torch.nn.ModuleList([SSIM(data_range=1.0, size_average=True, channel=c.size) for c in self.mm_channels])
        self.lpips = torch.nn.ModuleList([LearnedPerceptualImagePatchSimilarity(normalize=True) for _ in self.mm_channels])

        self.norm_reg = unit_norm_regularization_loss

        # init step
        self.step = 0

        # background color
        if self.config.background_color == "random":
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157])  # not really random, but whatever
        else:
            self.background_color = colors.get_color(self.config.background_color)

        # Conditionally create Strategy for GS densification
        strategyClass = (MMMaxAverageStrategy if self.config.densification_strategy == "max_average" else DefaultStrategy)
        self.strategy = (strategyClass)(
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=self.num_train_data + self.config.refine_every,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=True,
            pause_iterations=self.config.densification_pause_iterations
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)


        # Compute Size

        # MLP
        params = ["means", "scales", "quats", "opacities"]
        memory = 0

        if len(self.config.use_mlp_channels) == 0:
            # Feature MLP
            memory += compute_feature_mlp_memory(self.feature_mlp)
            # Add feature embedding
            params.append("feature_embedding")
        else:
            # Add SH components
            params.append("features_dc")
            params.append("features_rest")

        memory += compute_3dgs_memory(self, param_names=params)

        print("Occupied memory: {} MB".format(memory))



    def get_model_stats(self):
        """Generate a dict of representative values of the model; ToDo: expand"""
        return {
            "n_gaussians": self.gauss_params["means"].shape[0]
        }


    @property
    def num_points(self):
        return self.gauss_params["means"].shape[0]

    def _get_channel(self, channel_name):
        channel = next((c for c in self.mm_channels if c.name == channel_name), None)
        if channel == None:
            raise ValueError(f"channel {channel_name} not present in model")
        return channel


    def step_post_backward(self, step):
        """GS Strategy step after train iteration"""
        assert step == self.step
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            packed=False,
        )

    def step_cb(self, optimizers: Optimizers, step):
        """Callback before train iteration"""
        self.step = step
        self.optimizers = optimizers.optimizers

    def export_optimized_cameras(self, pipeline):
        if self.config.export_optimized_cameras is None:
            return

        from mmsplat.mmsplat_camera_export import ExportMMSplatCameras
        ExportMMSplatCameras.export_camera_transforms(pipeline, self.config.export_optimized_cameras)


    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Declare callbacks to run at specific steps in training"""
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )

        if self.config.export_optimized_cameras is not None:
            cbs.append(
                TrainingCallback(
                    [TrainingCallbackLocation.AFTER_TRAIN],
                    self.export_optimized_cameras,
                    args=[training_callback_attributes.pipeline]
                )
            )

        return cbs


    def get_gaussian_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "feature_embedding"]
        }

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer_rgb.get_param_groups(param_groups=gps)
        self.camera_optimizer_ms.get_param_groups(param_groups=gps)
        self.feature_mlp.get_param_groups(param_groups=gps)
        return gps


    def _get_downscale_factor(self):
        """Determine current downscale factor based on step and config"""
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1
        
    def _downscale_if_required(self, image):
        """Downscale image if needed for current step and config"""
        d = self._get_downscale_factor()
        if d > 1:
            return splatfacto.resize_image(image, d)
        return image


    def _get_background_color(self, num_channels=3):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(num_channels, device=self.device)
            elif num_channels == 3:
                background = self.background_color.to(self.device)
            else:
                gray_bg = 0.299 * self.background_color[0] + 0.587 * self.background_color[1] + 0.114 * self.background_color[2]
                return torch.tensor([gray_bg for i in range(num_channels)]).to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(num_channels, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(num_channels, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
    
        return background


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, Any]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # optimize camera if in training
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"

            # Check if RGB or MS optimizer should be used; see MMSplatModelConfig.rgb_channel_names
            if camera.metadata["mm_channel"] in self.config.rgb_channel_names:
                optimized_camera_to_world = self.camera_optimizer_rgb.apply_to_camera(camera)
            else:
                optimized_camera_to_world = self.camera_optimizer_ms.apply_to_camera(camera)

        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # get camera parameters based on current scale
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # determine which channel is to be rendered
        if "mm_channel" in camera.metadata:
            mm_channel_name = camera.metadata["mm_channel"]
        else:
            mm_channel_name = self.default_channel.value  # default to ViewerDropdown value

        render_channel = next((c for c in self.mm_channels if c.name == mm_channel_name), None)

        if render_channel is None:
            CONSOLE.log(f"[yellow]Channel {mm_channel_name} does not exist in model")
            raise ValueError(f"[yellow]Channel {mm_channel_name} does not exist in model")
        

        # based on channel, determine colors to pass to rasterization()
        if render_channel.use_mlp and (self.config.mlp_type == "standard"):
            # Feature vector from each splat
            feature_emb_colors = self.gauss_params["feature_embedding"]

            # Encode the viewing direction with positional encoding
            if self.config.direction_encoding_flag:
                # ToDo: Check if this is correct! Dont we just need two dim for viewing direction?
                camtoworlds = torch.inverse(viewmat)  # [C, 4, 4]

                # Direction between point and the current camera origin
                dirs = self.gauss_params["means"].detach().clone() - camtoworlds[:, None, :3, 3][0, 0]
                # Normalize it
                direction_encoding_in = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-12)
                # Encode the viewing direction
                direction_encoding_out = self.direction_encoding.forward(direction_encoding_in)
                # Concat it with the feature vector
                feature_emb_colors = torch.cat((feature_emb_colors, direction_encoding_out), axis=1)

            # Positional Encoding on splat position
            if self.config.positional_encoding_flag:
                positional_encoding_in = self.gauss_params["means"].detach().clone()
                positional_encoding_out = self.positional_encoding.forward(positional_encoding_in)
                # Concat it with the feature vector
                feature_emb_colors = torch.cat((feature_emb_colors, positional_encoding_out), axis=1)

            # Either knn for mlp with neighbours or the cosine sim for pulling nearby feature space together
            if self.config.use_neighbouring_features or self.config.use_cosine_features:
                positions = self.gauss_params["means"].detach().clone()

                # Conditionally compute KNN
                if (
                    self.step == 0      # on first iteration
                    or (self.step - 1) % self.config.refine_every == 0     # one step after densification
                    or (not self.training and isinstance(self.knn_index_reshape, type(None)))   # in viewer
                    or (self.knn_index_reshape.shape[0] != positions.shape[0])  # when shapes inconsistent (bugfix)
                ):
                    ret = self.knnsearch(positions.cpu(), positions.cpu(), k=self.k)
                    # Indices with self node and neighbouring in shapoe N x k x D (Num x neighbours x feature dim)
                    self.knn_index_reshape = ret.neighbors_index.reshape(ret.neighbors_index.shape[0] // self.k , self.k).cuda()


                # For rearranging the mlp input and concat the k nearest neighbours
                if self.config.use_neighbouring_features:
                    # Detach beacause we do no want to optimize all features - only the center one!
                    feature_emb_neighbours = feature_emb_colors[self.knn_index_reshape].detach()
                    # Put the center features back in place (they will be optimized!)
                    feature_emb_neighbours[:, 0, :] = feature_emb_colors
                    #
                    feature_emb_colors = feature_emb_neighbours.reshape(positions.shape[0], self.knn_feature_input_dim)

            # MLP forward
            predicted_values = self.feature_mlp.forward(feature_emb_colors)

            # Mapping from render_channel to color and opacity indices
            channel_info = {
                "color": list(range(render_channel.mlpout_index, render_channel.mlpout_index + render_channel.size)),
                "opacity": render_channel.opacity_correction_index
            }

            color_indices = channel_info["color"]
            colors = predicted_values[:, color_indices]

            # TODO: is this really necessary?
            if len(color_indices) == 1:
                colors = colors.repeat(1, 3)  # Convert single channel to RGB

            # Opacity correction
            opacity_correction = None
            if self.config.opacity_correction_flag and channel_info["opacity"] >= 0:
                opacity_correction = predicted_values[:, channel_info["opacity"]][:, None]

            if self.config.opacity_correction_flag and channel_info["opacity"] < 0:
                opacity_correction = torch.tensor(1.0).to(self.device)

            # For visualization of features! Only in visualizer and if flag ist set
            if self.training == False and self.feature_channel.value == True and self.config.feature_input_dim > 0:

                if not self.feature_colors.value:
                    # Only the feature embeddings
                    projected_rgb = feature_emb_colors[:, :self.config.feature_input_dim] @ self.proj_vec.cuda()
                else:
                    color_channels = [
                        int(self.feature_color_channel_0.value),
                        int(self.feature_color_channel_1.value),
                        int(self.feature_color_channel_2.value)
                    ]
                    projected_rgb = feature_emb_colors[:, :self.config.feature_input_dim][:, color_channels]

                # Normalize to 0 and 1
                colors = (projected_rgb - projected_rgb.min()) / (projected_rgb.max() - projected_rgb.min())
                colors = torch.clamp(colors, 0, 1)

            sh_degree_to_use = None
            #del feature_emb_colors

        elif render_channel.use_mlp and (self.config.mlp_type == "post"):
            # post-mlp; before rasterization we only have to set the feature vectors as colors and do N-D rasterization
            colors = self.gauss_params["feature_embedding"]
            sh_degree_to_use = None
        
        elif render_channel.use_sh:
            # generate tensor including all spherical harmonics coefficients
            sh_dc = self.gauss_params["features_dc"][:, None, render_channel.fdc_index : (render_channel.fdc_index + render_channel.size)]
            sh_rest = self.gauss_params["features_rest"][:, :, render_channel.frest_index : (render_channel.frest_index + render_channel.size)]
            colors = torch.cat((sh_dc, sh_rest), dim=1)

            # force 3 channels with sh
            if colors.shape[2] == 1:
                colors = colors.expand(-1, -1, 3)

            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)

        else:
            # get tensor of gaussian colors for the channel, and 
            colors = self.gauss_params["features_dc"][:, render_channel.fdc_index : (render_channel.fdc_index + render_channel.size)]
            # sigmoid activation (see gsplat.rasterize.rasterization arguments for reference)
            colors = torch.sigmoid(colors)

            sh_degree_to_use = None
        

        # set render mode (camera.times because only a viewer cameras is not None)
        if not self.training or (camera.times and (camera.metadata["cam_idx"] == 0)):
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"


        # determine whether to allow gradient for means, quats and scales
        if (render_channel.pos_optim_delay < 0) or (self.step < render_channel.pos_optim_delay):
            r_means = self.gauss_params["means"].detach().clone().requires_grad_()
            r_quats = self.gauss_params["quats"].detach().clone().requires_grad_()
            r_scales = self.gauss_params["scales"].detach().clone().requires_grad_()

            # ensure that gradients are cleared
            self.gauss_params["means"].grad = None
            self.gauss_params["quats"].grad = None
            self.gauss_params["scales"].grad = None
        else:
            r_means = self.gauss_params["means"]
            r_quats = self.gauss_params["quats"]
            r_scales = self.gauss_params["scales"]

        # determine whether to allow gradient for opacities
        if (render_channel.opacity_optim_delay < 0) or (self.step < render_channel.opacity_optim_delay):
            r_opacities = self.gauss_params["opacities"].detach().clone().requires_grad_()
            # ensure that gradients are cleared
            self.gauss_params["opacities"].grad = None
        else:
            r_opacities = self.gauss_params["opacities"]

        # Only apply opacity correction if flag is set
        if self.config.opacity_correction_flag:
            r_opacities = r_opacities * opacity_correction


        # let the magic happen: rasterize
        render, alpha, self.info = rasterization(
            means=r_means,
            quats=r_quats,
            scales=torch.exp(r_scales),
            opacities=torch.sigmoid(r_opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )


        # if mlp_type==post, we now can run the rasterized features through the MLP
        if render_channel.use_mlp and (self.config.mlp_type == "post"):

            # inputs to MLP start as render result
            mlp_post_inputs = render[..., :self.config.feature_input_dim] # slice out depth if present
            render_removed = render[..., self.config.feature_input_dim:]
            *render_batch_dims, H2, W2, D2 = render.shape
            
            # concat pixel embedding if necessary
            if self.config.mlp_post_pixel_embedding:

                # Create normalized pixel grid [H2, W2, 2]
                ys = torch.linspace(-1, 1, H2, device=render.device)
                xs = torch.linspace(-1, 1, W2, device=render.device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
                pixel_pos = torch.stack([grid_x, grid_y], dim=-1)  # [H2, W2, 2]

                # expand to render
                expand_shape = [1] * len(render_batch_dims) + [H2, W2, 2]
                pixel_pos = pixel_pos.view(*expand_shape)
                pixel_pos = pixel_pos.expand(*render_batch_dims, H2, W2, 2)

                # append to mlp_post_inputs
                mlp_post_inputs = torch.cat([mlp_post_inputs, pixel_pos], dim=-1)
            
            # reshape mlp_post_inputs into 2D (N, D)
            mlp_post_inputs = mlp_post_inputs.view(-1, mlp_post_inputs.shape[-1])

            # ToDo: Check if this is correct! Dont we just need two dim for viewing direction?
            camtoworlds = torch.inverse(viewmat)  # [C, 4, 4]

            # Encode the viewing direction with positional encoding
            # TODO: not implemented yet

            # Add camera position if requested
            if self.config.mlp_post_camera_position_embedding:
                cam_pos = camtoworlds[:, :3, 3].expand(mlp_post_inputs.shape[0], 3)
                mlp_post_inputs = torch.cat((mlp_post_inputs, cam_pos), axis=1)

            # add viewing direction if requested
            if self.config.mlp_post_viewing_direction:
                # Extract camera viewing direction (negative Z-axis in camera space)
                cam_viewing_dir = -camtoworlds[:, :3, 2]  # [batch, 3]
                # Normalize the viewing direction
                cam_viewing_dir = cam_viewing_dir / (torch.norm(cam_viewing_dir, dim=-1, keepdim=True) + 1e-12)
                # Expand to match mlp_post_inputs batch size
                cam_viewing_dir_expanded = cam_viewing_dir.expand(mlp_post_inputs.shape[0], 3)
                mlp_post_inputs = torch.cat((mlp_post_inputs, cam_viewing_dir_expanded), axis=1)


            # MLP forward
            predicted_values = self.feature_mlp.forward(mlp_post_inputs)

            # Mapping from render_channel to color and opacity indices
            channel_info = {
                "color": list(range(render_channel.mlpout_index, render_channel.mlpout_index + render_channel.size)),
                "opacity": render_channel.opacity_correction_index
            }

            color_indices = channel_info["color"]
            colors = predicted_values[:, color_indices]

            # reshape colors back to render shape
            render_shape = [*render_batch_dims, H2, W2, len(color_indices)]
            colors = colors.view(*render_shape)

            # re-attach render_removed
            render = torch.cat([colors, render_removed], dim=-1)


        # complete self.info for strategy
        self.info["n_cameras"] = 1
        self.info["mm_channel"] = mm_channel_name

        # strategy step if necessary
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        # determine depth image
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        # apply background color
        alpha = alpha[:, ...]
        background = self._get_background_color(render_channel.size)

        render = render[:, ..., 0:(render_channel.size)] + (1 - alpha) * background
        render = torch.clamp(render, 0.0, 1.0)

        if not self.training:
            background = background.expand(H, W, render_channel.size)

        # remove batch dimension, since batch size is one
        render = render.squeeze(0)

        # generate rgb if render is grayscale
        if render.shape[2] == 1:
            rgb = render.expand(-1, -1, 3)
        else:
            rgb = render

        # For vegetation index sometimes an error is thrown that outputs['depth'] is None
        if depth_im is None:
            depth_im = torch.ones_like(rgb)

        # result dict
        return {
            "rgb": rgb,
            "render": render,
            "depth": depth_im,
            "accumulation": alpha.squeeze(0),
            "background": background,
            "mm_channel": render_channel.name
        }


    def get_optimized_cam_to_worlds(self, cameras: List[Tuple[Cameras, Dict]]) -> List[torch.Tensor]:
        """For a list of TRAINING cameras, return the optimized camera to world transforms"""
        return [self.camera_optimizer.apply_to_camera(cam) for cam, cdict in cameras]


    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == (background.shape[0] + 1):
            num_channels = background.shape[0]
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, num_channels))
            return alpha * image[..., :num_channels] + (1 - alpha) * background
        else:
            return image


    @staticmethod
    def _add_mm_metric(metrics_dict, channel, metric_name, metric):
        """Helper funtion to duplicate a metric for a channel"""
        metrics_dict[metric_name] = metric
        metrics_dict[metric_name + "_" + channel.name] = metric



    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        # get ground-truth and rendered image
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_img = outputs["render"]

        # get channel information
        channel = self._get_channel(outputs["mm_channel"])

        # compute metrics
        metrics_dict = {}

        psnr_val = self.psnr[channel.index](predicted_img, gt_img)
        psnr_val2 = self.psnr2[channel.index](predicted_img, gt_img)
        MMSplatModel._add_mm_metric(metrics_dict, channel, "psnr", psnr_val)
        MMSplatModel._add_mm_metric(metrics_dict, channel, "psnr_torch", psnr_val2)

        if channel.name in self.config.color_corrected_channels:
            cc_img = splatfacto.color_correct(predicted_img, gt_img)
            cc_psnr = self.psnr[channel.index](cc_img, gt_img)
            MMSplatModel._add_mm_metric(metrics_dict, channel, "cc_psnr", cc_psnr)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer_rgb.get_metrics_dict(metrics_dict)
        self.camera_optimizer_ms.get_metrics_dict(metrics_dict)

        return metrics_dict




    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["render"]

        channel = self._get_channel(outputs["mm_channel"])

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim[channel.index](gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.gauss_params["scales"])
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)


        # Normalize feature length
        if self.config.use_feature_norm_regularization:
            feature_norm_reg = self.config.lambda_norm * self.norm_reg(self.gauss_params["feature_embedding"])
        else:
            feature_norm_reg = torch.tensor(0.0).to(self.device)

        # Add smoothness loss for thermal data if activated
        if self.config.use_smoothness_loss and outputs['mm_channel'] == "MS_T":
            thermal_smooth_loss = self.config.lambda_smoothness * smoothness_loss(pred_img)
        else:
            thermal_smooth_loss  = torch.tensor(0.0).to(self.device)

        # Use cosine similarity to pull neighbouring feature space together
        if self.config.use_cosine_features and self.config.feature_input_dim > 0:
            cosine_loss = self.config.lambda_cosine * cosine_neighbor_loss_batched(self.gauss_params["feature_embedding"][self.knn_index_reshape])
        else:
            cosine_loss  = torch.tensor(0.0).to(self.device)


        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + feature_norm_reg + thermal_smooth_loss + cosine_loss ,
            "scale_reg": scale_reg,
        }

        if self.training:
            # Add loss from camera optimizer
            if channel.name in self.config.rgb_channel_names:
                self.camera_optimizer_rgb.get_loss_dict(loss_dict)
            else:
                self.camera_optimizer_ms.get_loss_dict(loss_dict)

        return loss_dict


    def get_band_indices(self, camera: Cameras):
        """
        Computes the specified vegetation index (e.g., NDVI, NDRE) by setting the appropriate
        mm_channel metadata on the camera, calling get_outputs, and combining spectral bands.

        Args:
            camera (Cameras): Camera object with multispectral metadata.

        Returns:
            dict: Output dictionary with the computed band index in 'rgb' and 'render' fields.
        """
        if not camera.metadata:
            camera.metadata = {'mm_channel': None}

        # Mapping from index name to the required channels and the corresponding compute function
        band_config = {
            "NDVI": {
                "channels": (self.config.vi_red, self.config.vi_nir),
                "func": lambda r, nir: BAND_INDICES["NDVI"](red_channel=r, nir_channel=nir),
            },
            "NDRE": {
                "channels": (self.config.vi_rededge, self.config.vi_nir),
                "func": lambda re, nir: BAND_INDICES["NDRE"](red_edge_channel=re, nir_channel=nir),
            },
            "NDWI": {
                "channels": (self.config.vi_green, self.config.vi_nir),
                "func": lambda g, nir: BAND_INDICES["NDWI"](green_channel=g, nir_channel=nir),
            },
            "GNDVI": {
                "channels": (self.config.vi_green, self.config.vi_nir),
                "func": lambda g, nir: BAND_INDICES["GNDVI"](green_channel=g, nir_channel=nir),
            },
            "SAVI": {
                "channels": (self.config.vi_red, self.config.vi_nir),
                "func": lambda r, nir: BAND_INDICES["SAVI"](red_channel=r, nir_channel=nir, L=0.5),
            }
        }

        index_name = self.default_channel.value
        if index_name not in band_config:
            raise ValueError(f"{index_name} not in {self.band_indices}")

        channel_a, channel_b = band_config[index_name]["channels"]
        compute_fn = band_config[index_name]["func"]

        # Fetch data for each channel
        camera.metadata['mm_channel'] = channel_a
        out_a = self.get_outputs(camera.to(self.device))

        camera.metadata['mm_channel'] = channel_b
        out_b = self.get_outputs(camera.to(self.device))

        # Copy one of the outputs as base
        outs = copy.deepcopy(out_a)

        # Compute band index and update output
        band_index = compute_fn(out_a['rgb'], out_b['rgb'])
        outs['rgb'] = band_index
        outs['render'] = band_index[..., :1]

        return outs



    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        # TODO: implement crop box functionality in get_outputs
        # self.set_crop(obb_box)

        # if isinstance(camera.metadata, type(None)):
        #    camera.metadata = []  # Hacky bug fix for error  mmsplat/lib/python3.8/site-packages/nerfstudio/viewer/render_state_machine.py", line 249, in _send_output_to_viewer
        #     output_keys = set(outputs.keys())

        try:
            # if isinstance(camera.metadata, type(None)):
            #    camera.metadata = []  # Hacky bug fix for error  mmsplat/lib/python3.8/site-packages/nerfstudio/viewer/render_state_machine.py", line 249, in _send_output_to_viewer
            #     output_keys = set(outputs.keys())

            if (self.default_channel.value in self.band_indices and not 'mm_channel' in camera.metadata) or "render" in camera.metadata:  # (self.default_channel.value in self.band_indices) and (camera.metadata["cam_idx"] == 0) and (camera.times != None):
                outs = self.get_band_indices(camera=camera.to(self.device))
            else:
                outs = self.get_outputs(camera=camera.to(self.device))
        except Exception as e:
            # Add some error logging for viewer
            print(e)
            print(camera)
            print(camera.metadata)
            traceback.print_exc()
            try:
                outs = self.get_outputs(camera=camera.to(self.device))
            except Exception as e2:
                return None
            
            return outs

        return outs  # type: ignore



    @staticmethod
    def _convert_batch_to_rgb(images):
        """Helper function to force image to have 3 color channels
        
        images must be a tensor of shape (N, C, H, W), where:
            N is the batch size
            C is the channel count
            H is the height
            W is the width
        """
        if images.shape[1] < 3:
            return images[:, 0:1, :, :].expand(-1, 3, -1, -1)
        else:
            return images[:, 0:3, :, :]



    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        # get channel info
        channel = self._get_channel(outputs["mm_channel"])

        # get ground-truth image, rendered image, combined image, and color corrected image (if necessary)
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_img = outputs["render"]
        combined_img = torch.cat([gt_img, predicted_img], dim=1)
        cc_img = None

        if channel.name in self.config.color_corrected_channels:
            cc_img = splatfacto.color_correct(predicted_img, gt_img)
            cc_img = torch.moveaxis(cc_img, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_img = torch.moveaxis(gt_img, -1, 0)[None, ...]
        predicted_img = torch.moveaxis(predicted_img, -1, 0)[None, ...]

        # for lpips, we need to force 3 color channels
        lpips_gt_img = MMSplatModel._convert_batch_to_rgb(gt_img)
        lpips_pred_img = MMSplatModel._convert_batch_to_rgb(predicted_img)
        lpips_cc_img = MMSplatModel._convert_batch_to_rgb(cc_img) if cc_img is not None else None

        # calculate psnr, ssim and lpips
        psnr = self.psnr[channel.index](gt_img, predicted_img)
        psnr2 = self.psnr2[channel.index](gt_img, predicted_img)
        ssim = self.ssim[channel.index](gt_img, predicted_img)
        lpips = self.lpips[channel.index](lpips_gt_img, lpips_pred_img)

        # add metrics to dict
        metrics_dict = {}
        MMSplatModel._add_mm_metric(metrics_dict, channel, "psnr", float(psnr.item()))
        MMSplatModel._add_mm_metric(metrics_dict, channel, "psnr_torch", float(psnr2.item()))
        MMSplatModel._add_mm_metric(metrics_dict, channel, "ssim", float(ssim))
        MMSplatModel._add_mm_metric(metrics_dict, channel, "lpips", float(lpips))


        if channel.name in self.config.color_corrected_channels:
            assert cc_img is not None

            # calculate metrics for color-corrected images
            cc_psnr = self.psnr[channel.index](gt_img, cc_img)
            cc_ssim = self.ssim[channel.index](gt_img, cc_img)
            cc_lpips = self.lpips[channel.index](lpips_gt_img, lpips_cc_img)

            # add cc metrics to dict
            MMSplatModel._add_mm_metric(metrics_dict, channel, "cc_psnr", float(cc_psnr.item()))
            MMSplatModel._add_mm_metric(metrics_dict, channel, "cc_ssim", float(cc_ssim))
            MMSplatModel._add_mm_metric(metrics_dict, channel, "cc_lpips", float(cc_lpips))

        # add combined image to output image list, with channel as name
        images_dict = {channel.name: combined_img}

        return metrics_dict, images_dict



    def load_state_dict(self, dict, **kwargs):
        """Before we can load a checkpoint, we have to properly resize the param tensors"""

        self.step = 30000
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)



    def _load_channel_int_param_from_list(self, channel_list, param_list: List[str], default_val: int, param_name: str):
        """Given a parameter list (formatted like config.pos_optim_delay_channels for example), extract and store the integer values for every channel"""

        # first, set default value to all channels
        for ch in channel_list:
            setattr(ch, param_name, default_val)

        # parse channel param
        cur_channel_set = set()
        for param_arg in param_list:
            # first, check if argument is a known channel
            if param_arg in (c.name for c in channel_list):
                # add to channel accumulation for next value
                cur_channel_set.add(param_arg)
            else:
                # check if argument is a valid int
                try:
                    param_val = int(param_arg)

                    # assign param_name to accumulated channel list
                    for ch in channel_list:
                        if ch.name in cur_channel_set:
                            setattr(ch, param_name, param_val)

                    # clear accumulation
                    cur_channel_set.clear()

                except ValueError:
                    # the argument is neither a known channel, nor a number. Skip...
                    pass

    def _init_config_params(self):
        """Init channel info based on config param lists"""
        self._load_channel_int_param_from_list(self.mm_channels, self.config.pos_optim_delay_channels, 0, "pos_optim_delay")
        self._load_channel_int_param_from_list(self.mm_channels, self.config.opacity_optim_delay_channels, 0, "opacity_optim_delay")
        



    @staticmethod
    def k_nearest_sklearn(x: torch.Tensor, k: int):
        """Copied from nerfstudio.models.splatfacto.SplatfactoModel

        Find k-nearest neighbors using sklearn's NearestNeighbors.
            x: The data tensor of shape [num_samples, num_features]
            k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)