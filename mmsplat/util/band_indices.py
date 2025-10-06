import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Union


def compute_NDVI(red_channel: Union[np.ndarray, torch.tensor], nir_channel: Union[np.ndarray, torch.tensor], normalize: bool = True):
    """
    Computes NDVI based on red and nir channel

    Formula: NDVI = ((NIR - Red)/(NIR + Red))
    """
    ndvi = (nir_channel - red_channel) / (nir_channel + red_channel)
    if normalize:
        ndvi = torch.clamp((ndvi + 0.5 ) / 1.3, 0, 1)
    return ndvi


def compute_NDRE(red_edge_channel: Union[np.ndarray, torch.tensor], nir_channel: Union[np.ndarray, torch.tensor], normalize: bool = True):
    """
    Computes NDRE based on red edge and nir channel

    Formula: NDVI = ((NIR - Red_Edge)/(NIR + Red_Edge))
    """
    ndre = (nir_channel - red_edge_channel) / (nir_channel + red_edge_channel)
    if normalize:
        ndre = torch.clamp(ndre * 2 , 0, 1)
    return ndre


def compute_NDWI(green_channel: Union[np.ndarray, torch.tensor], nir_channel: Union[np.ndarray, torch.tensor], normalize: bool = True):
    """
    Computes NDWI based on green edge and nir channel

    Formula: NDVI = ((NIR - Green)/(NIR + Green))
    """
    ndwi = (green_channel - nir_channel) / (green_channel + nir_channel)
    if normalize:
        ndwi = torch.clamp((ndwi + 0.6 ) / 1.1, 0, 1)
    return ndwi


def compute_GNDVI(green_channel: Union[np.ndarray, torch.tensor], nir_channel: Union[np.ndarray, torch.tensor], normalize: bool = True):
    """
    Computes GNDVI based on green and nir channel

    Formula: GNDVI = ((NIR - Green)/(NIR + Green))
    """
    gndvi = (nir_channel - green_channel) / (nir_channel + green_channel)
    if normalize:
        gndvi = torch.clamp((gndvi + 0.5 ) / 1.1, 0, 1)
    return gndvi


def compute_SAVI(red_channel: Union[np.ndarray, torch.tensor],
                 nir_channel: Union[np.ndarray, torch.tensor],
                 L: float = 0.5,
                 normalize: bool = True):
    """
    Computes SAVI based on red and nir channel

    Formula: SAVI = ((NIR - Red) / (NIR + Red + L)) x (1 + L)
    """

    savi = (nir_channel - red_channel) / (nir_channel + red_channel + L) * (1 + L)
    if normalize:
        savi = torch.clamp((savi + 0.5 )/1.1, 0, 1 )

    return savi


BAND_INDICES = {
    "NDVI": compute_NDVI,
    "NDRE": compute_NDRE,
    "NDWI": compute_NDWI,
    "GNDVI": compute_GNDVI,
    "SAVI": compute_SAVI,
}
