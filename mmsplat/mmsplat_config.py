from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.base_config import ViewerConfig, MachineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    ExponentialDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)


from mmsplat.mmsplat_dataparser import MMSplatDataParserConfig
from mmsplat.mmsplat_datamanager import MMSplatDataManagerConfig
from mmsplat.mmsplat_model import MMSplatModelConfig
from mmsplat.mmsplat_pipeline import MMSplatPipelineConfig
from mmsplat.mmsplat_camera_optimizer import CameraOptimizerConfig




mmsplat_method = MethodSpecification(
    config=TrainerConfig(
        method_name="mmsplat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=120000,
        pipeline=MMSplatPipelineConfig(
            datamanager=MMSplatDataManagerConfig(
                dataparser=MMSplatDataParserConfig(
                    load_3D_points=True
                )
            ),
            model=MMSplatModelConfig(
                camera_optimizer_rgb=CameraOptimizerConfig(mode="off", ms_type="D"), # SO3xR3
                camera_optimizer_ms=CameraOptimizerConfig(mode="off",ms_type="MS"),  # SO3xR3
            )

        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=120000,
                ),
            },
            "feature_embedding": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "feature_mlp": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt_D": {
                "optimizer": AdamOptimizerConfig(lr=1e-15, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-15, max_steps=10000, warmup_steps=1000, lr_pre_warmup=1e-15
                ),
            },
            "camera_opt_MS": {
                "optimizer": AdamOptimizerConfig(lr=1e-15, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=10000, warmup_steps=1000, lr_pre_warmup=1e-15
                ),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        machine=MachineConfig(seed=42)
    ),
    description="Multi-modal gaussian splatting"
)
