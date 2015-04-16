"""Given a confocal image, find stomata within that image."""

import os
import argparse

import numpy as np

import scipy.misc
import scipy.ndimage as nd

import skimage.measure
import skimage.filters

from util import safe_mkdir
from jicimagelib.io import FileBackend
from jicimagelib.image import DataManager, Image
from jicimagelib.transform import transformation

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

@transformation
def find_connected_components(image, neighbors=8, background=None):

    connected_components = skimage.measure.label(image, 
                                                 neighbors=neighbors,
                                                 background=background,
                                                 return_num=False)


    return Image.from_array(connected_components.astype(np.uint8))

@transformation
def max_intensity_projection(stack):
    """Return max intensity projection for stack."""

    iz_max = np.argmax(stack, 2)

    xmax, ymax, _ = stack.shape

    projection = np.zeros((xmax, ymax), dtype=stack.dtype)

    for x in range(xmax):
        for y in range(ymax):
            projection[x, y] = stack[x, y, iz_max[x, y]]

    return Image.from_array(projection)

def normalise_image(image):
    """Return image with values between 0 and 1."""

    im_max = float(image.max())
    im_min = float(image.min())

    return (image - im_min) / (im_max - im_min)

def test_normalise_image():

    array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    image = Image.from_array(array)

    normalised_image = normalise_image(image)

    assert(normalised_image.max() == 1)
    assert(normalised_image.min() == 0)
    assert(normalised_image[1,1] == 0.5)

def gaussian_filter(image, sigma=0.4):

    gauss = skimage.filters.gaussian_filter(image, sigma=sigma)

    gauss_norm = normalise_image(gauss)

    gauss_uint8 = (255 * gauss_norm).astype(np.uint8)

    return Image.from_array(gauss_uint8)

@transformation
def threshold_otsu(image, mult=1):

    otsu_value = skimage.filters.threshold_otsu(image)

    bool_image = image > mult * otsu_value

    scaled_image = 255 * bool_image

    return scaled_image.astype(np.uint8)

# NOTES:

# history

# type mangling

    
def find_stomata(confocal_file):
    """Given the confocal image file, find stomata in it."""

    image_collection = unpack_data(confocal_file)

    raw_z_stack = image_collection.zstack_array(s=8)[3:]
    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))

    projection = max_intensity_projection(smoothed_z_stack)

    im_thresholded = threshold_otsu(projection)

    # z_stack = Stack(nd.gaussian_filter(raw_z_stack, (3, 3, 1)))
    # z_stack.history = []

    # representative_image = find_suitable_2D_image(z_stack)

    # print type(representative_image)

    #im_thresholded = threshold_otsu(representative_image)

    connected_components = find_connected_components(im_thresholded)

    for ccID in np.unique(connected_components):
        coords = np.where(connected_components == ccID)
        print len(coords[0])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    unpack_data(args.confocal_file)
    find_stomata(args.confocal_file)

if __name__ == "__main__":
    main()
