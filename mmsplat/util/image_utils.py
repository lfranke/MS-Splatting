import imageio.v3 as iio
import cv2
import numpy as np
from pathlib import Path

from typing import Literal, Dict, Optional




def convert_image_type(
    image: np.ndarray,
    to_format: Literal["uint8", "uint16", "uint32", "float32"]
) -> Optional[np.ndarray]:
    """Given an image as np.ndarray of type uint8|uint16|uint32|float32, convert to the specified format.
    DOES NOT COPY if format is already fulfilled."""

    UINT8_MAX = 255
    UINT16_MAX = 65535
    UINT32_MAX = 4294967295

    if image.dtype.name == to_format:
        return image
    
    elif image.dtype.name == "uint8":  # uint8 source
        if to_format == "uint16":           # uint16 target
            return image.astype("uint16") * np.uint16(UINT16_MAX / UINT8_MAX)
        elif to_format == "uint32":         # uint32 target
            return image.astype("uint32") * np.uint32(UINT32_MAX / UINT8_MAX)
        elif to_format == "float32":        # float32 target
            return image.astype("float32") / np.float32(UINT8_MAX)
        else:
            print(f"image_utils.convert_image_type: Invalid target format {to_format}")
            return None
    elif image.dtype.name == "uint16":  # uint16 source
        if to_format == "uint8":           # uint8 target
            return (image / (UINT16_MAX / UINT8_MAX)).astype("uint8")
        elif to_format == "uint32":         # uint32 target
            return (image.astype("uint32") * np.uint32(UINT32_MAX / UINT16_MAX))
        elif to_format == "float32":        # float32 target
            return image.astype("float32") / np.float32(UINT16_MAX)
        else:
            print(f"image_utils.convert_image_type: Invalid target format {to_format}")
            return None
    elif image.dtype.name == "uint32":  # uint32 source
        if to_format == "uint8":           # uint8 target
            return (image / (UINT32_MAX / UINT8_MAX)).astype("uint8")
        elif to_format == "uint16":           # uint16 target
            return (image / (UINT32_MAX / UINT16_MAX)).astype("uint16")
        elif to_format == "float32":        # float32 target
            return image.astype("float32") / np.float32(UINT32_MAX)
        else:
            print(f"image_utils.convert_image_type: Invalid target format {to_format}")
            return None
    elif image.dtype.name == "float32":  # float32 source
        if to_format == "uint8":           # uint8 target
            return (image.clip(0, 1) * UINT8_MAX).astype("uint8")
        elif to_format == "uint16":           # uint16 target
            return (image.clip(0, 1) * UINT16_MAX).astype("uint16")
        elif to_format == "uint32":        # uint32 target
            return (image.clip(0, 1) * UINT32_MAX).astype("uint32")
        else:
            print(f"image_utils.convert_image_type: Invalid target format {to_format}")
            return None
    else:
        print(f"image_utils.convert_image_type: Invalid source format {image.dtype}")
        return None





def load_image_data(
    file: Path,
    scale_factor: float = 1.0,
    image_type: Literal["uint8", "uint16", "uint32", "float32", "auto"] = "auto"
) -> Optional[np.ndarray]:
    """Load the image data given the file path and convert to specified format."""

    # load data and validate image format
    imgdata = iio.imread(file)
    if imgdata.dtype.name not in ["uint8", "uint16", "uint32", "float32"]:
        print(f"image_utils.load_image_data: Invalid image format {imgdata.dtype}")
        return None
    
    # add last dimension for grayscale images
    if len(imgdata.shape) == 2:
        imgdata = imgdata[:, :, np.newaxis]
    
    # validate dimensions
    if (len(imgdata.shape) != 3) or any([d == 0 for d in imgdata.shape]) or (imgdata.shape[2] > 4):
        print(f"image_utils.load_image_data: Invalid image shape {imgdata.shape}")
        return None
    
    # scale image if necessary
    if scale_factor != 1:
        newsize = (int(imgdata.shape[1] * scale_factor), int(imgdata.shape[0] * scale_factor))
        prev_type = imgdata.dtype

        imgdata = cv2.resize(imgdata, dsize=newsize, interpolation=(cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR))

        assert imgdata.dtype == prev_type, "cv2.resize changed img.dtype"

    # convert to requested format
    if image_type != "auto":
        imgdata = convert_image_type(imgdata, image_type)

    return imgdata