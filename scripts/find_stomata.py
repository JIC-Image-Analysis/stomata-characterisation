"""Given a confocal image, find stomata within that image."""

import os
import argparse

import scipy.misc
import scipy.ndimage as nd

from util import safe_mkdir
from jicimagelib.io import FileBackend
from jicimagelib.image import DataManager

from protoimg.stack import Stack, normalise_stack
from protoimg.transform import (
    max_intensity_projection,
    find_edges,
    threshold_otsu
    )

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', 'data', 'unpack')

def unpack_data(confocal_file):
    """Unpack the file and return an image collection object."""
    safe_mkdir(UNPACK)

    backend = FileBackend(UNPACK)
    data_manager = DataManager(backend)

    data_manager.load(confocal_file)

    #print data_manager.get_image_proxy(s=1)
    image_collection = data_manager[0]

    return image_collection

def find_suitable_2D_image(z_stack):
    """From the z-stack, find a suitable 2D representation of the image."""

    normalised_stack = normalise_stack(z_stack)
    projection = max_intensity_projection(z_stack)
    normalised_projection = max_intensity_projection(normalised_stack, 'norm_projection')

    #return z_stack.plane(10)

    return projection
    
def find_stomata(confocal_file):
    """Given the confocal image file, find stomata in it."""

    image_collection = unpack_data(confocal_file)

    raw_z_stack = image_collection.zstack_array(s=8)[3:]

    z_stack = Stack(nd.gaussian_filter(raw_z_stack, (3, 3, 1)))
    z_stack.history = []

    representative_image = find_suitable_2D_image(z_stack)

    scipy.misc.imsave('rep_image.png', representative_image.image_array)

    threshold_otsu(representative_image)

    find_edges(representative_image)





def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    unpack_data(args.confocal_file)
    find_stomata(args.confocal_file)

if __name__ == "__main__":
    main()
