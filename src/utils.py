from deepface.commons import distance as dst
from typing import Union
from viam.media.video import RawImage
from PIL import Image
from viam.logging import getLogger
from viam.media.video import CameraMimeType
import numpy as np
LOGGER = getLogger(__name__)
SUPPORTED_IMAGE_TYPE = [CameraMimeType.JPEG,
                        CameraMimeType.PNG,
                        CameraMimeType.VIAM_RGBA]

def euclidian_l2(source_embed,target_embed ):
    return dst.findEuclideanDistance(
                        dst.l2_normalize(source_embed),
                        dst.l2_normalize(target_embed),
                    )
    
    
def decode_image(image: Union[Image.Image, RawImage])-> np.ndarray:
    """decode image to BGR numpy array

    Args:
        raw_image (Union[Image.Image, RawImage])

    Returns:
        np.ndarray: BGR numpy array
    """
    if type(image) == RawImage:
        if image.mime_type not in SUPPORTED_IMAGE_TYPE:
            LOGGER.error(f"Unsupported image type: {image.mime_type}. Supported types are {SUPPORTED_IMAGE_TYPE}.")
            raise ValueError(f"Unsupported image type: {image.mime_type}.")
        im = Image.open(image.data).convert('RGB') #convert in RGB png openened in RGBA
        return np.array(im)[...,::-1]

    res = image.convert('RGB')
    rgb = np.array(res)
    bgr = rgb[...,::-1]
    return bgr

def dist_to_conf_sigmoid(dist, steep=10):
    return 1 / (1 + np.exp((steep*(dist-0.5))))
    