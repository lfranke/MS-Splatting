import torch
from torch import nn
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Any

def get_activation_function(activation_name: Literal["ReLU","LeakyReLU","SiLU","GELU","Tanh","Sigmoid","ELU"]):
    """
    Returns an instance of the activation function from torch.nn based on its name.

    """
    activation_cls = getattr(nn, activation_name, None)

    if activation_cls is None:
        raise ValueError(f"'{activation_name}' is not found in torch.nn. Check the spelling and case.")

    if not isinstance(activation_cls, type) or not issubclass(activation_cls, nn.Module):
        raise ValueError(
            f"'{activation_name}' exists in torch.nn but is not a valid nn.Module subclass (not instantiable).")

    return activation_cls


class MultiSpectralFeatureDecoder(nn.Module):
    def __init__(
            self,
            input_dim: torch.tensor,
            output_dim: int = 3,
            hidden_depth: int = 32,
            hidden_layers: int = 1,
            hidden_activation_function: str = "SiLU",
            opacity_correction_count: int = 0,
            mlp_mode: bool = False,
            *args,
            **kwargs
    ):
        """
        Multi-Spectral Feature Decoder using a simple MLP (Multi-Layer Perceptron).

        Args:
            input_dim (torch.tensor): Dimension of input features.
            output_dim (int): Dimension of output features (default: 3 for RGB).
            hidden_depth (int): Number of hidden units in each hidden layer.
            opacity_correction_count (bool): Number of opacity correction output values
            mlp_mode (bool): Determines if this module should be included in param groups for optimization.
        """
        super().__init__(*args, **kwargs)

        if hidden_depth <= 0:
            raise ValueError("hidden_depth must be a positive integer. Value {}".format(hidden_depth))
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be a positive integer. Value {}".format(hidden_layers))


        # Store flags
        self.mlp_mode = mlp_mode
        self.num_hidden_layers = hidden_layers
        self.hidden_activation_function = get_activation_function(hidden_activation_function)
        self.opacity_correction_count = max(0, opacity_correction_count)

        # Input Layer
        self.fc_in = nn.Linear(input_dim, hidden_depth, bias=True)
        self.activation_in = self.hidden_activation_function()

        # hidden layers (at least 1)
        layer_modules = []
        for _ in range(self.num_hidden_layers):
            layer_modules.append(nn.Linear(hidden_depth, hidden_depth, bias=True))
            layer_modules.append(self.hidden_activation_function())
        self.hidden_layers = nn.Sequential(*layer_modules)

        # Output layer: predicts RGB (or other spectral channels) + optional opacity
        self.fc_out = nn.Linear(hidden_depth, output_dim + self.opacity_correction_count, bias=True)
        self.activation_out = nn.Sigmoid()  # Squashes spectral outputs to [0, 1]

        # Opacity uses ReLU activation to allow only positive values
        if self.opacity_correction_count > 0:
            self.activation_out_opacity = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Output tensor of shape [batch_size, output_dim (+ optional 4 opacity channels)].
        """
        x = self.fc_in(x)
        x = self.activation_in(x)

        # If self.hidden_layers is empty, the output is identical to the input
        x = self.hidden_layers(x)

        x = self.fc_out(x)

        if self.opacity_correction_count > 0:
            # Apply sigmoid to spectral channels and ReLU to opacity channels
            x_spectral = self.activation_out(x[:, :-self.opacity_correction_count])  # spectral output
            x_opacity = self.activation_out_opacity(x[:, -self.opacity_correction_count:])  # opacity output
            x = torch.concat([x_spectral, x_opacity], dim=1)
        else:
            x = self.activation_out(x)  # all output channels use sigmoid

        return x

    def get_param_groups(self, param_groups: dict) -> None:
        """
        Adds the MLP's parameters to the optimizer's parameter groups, if mlp_mode is enabled.

        Args:
            param_groups (dict): Dictionary of parameter groups for optimizer.
        """
        if self.mlp_mode:
            param_groups["feature_mlp"] = list(self.parameters())
