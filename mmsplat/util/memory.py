import torch
import torch.nn as nn

def compute_3dgs_memory(model,
                        param_names=["means", "scales", "quats", "features_dc", "features_rest", "opacities"]) -> float:
    """
    Computes memory (in MB) used by the 3DGS parameters only.

    Parameters:
        model (MMSplatModel): The initialized model instance.
        param_names (list): list of all parameter names to be considered.

    Returns:
        float: Memory usage in megabytes.
    """

    def tensor_bytes(t: torch.Tensor) -> int:
        return t.numel() * 4  # float32 = 4 bytes

    total_bytes = 0

    for name in param_names:
        if name in model.gauss_params:
            total_bytes += tensor_bytes(model.gauss_params[name])

    return total_bytes / (1024 ** 2)  # in MB

def __count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_feature_mlp_memory(model: nn.Module, bytes_per_param: int = 4) -> float:
    total_params = __count_parameters(model)
    total_bytes =  total_params * bytes_per_param
    return total_bytes / (1024 ** 2)  # in MB