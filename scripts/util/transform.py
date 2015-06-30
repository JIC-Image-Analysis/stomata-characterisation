"""Custom transforms for identifying stomata."""

import numpy as np
import skimage.measure

from jicimagelib.transform import transformation
from jicimagelib.image import Image

@transformation
def find_connected_components(image, neighbors=8, background=None):

    connected_components = skimage.measure.label(image, 
                                                 neighbors=neighbors,
                                                 background=background,
                                                 return_num=False)

    return Image.from_array(connected_components.astype(np.uint8))

