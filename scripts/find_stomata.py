"""Given a confocal image, find stomata within that image."""

import os
import sys
import math
import timeit
import argparse
import unittest

import numpy as np



import cv2

import scipy.misc
import scipy.ndimage as nd

import skimage.measure
import skimage.filters
import skimage.morphology

from jicimagelib.region import Region

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

def find_candidate_regions(raw_z_stack):
    """Given the z stack, find regions in the image which are candidate stomata.
    Return a dictionary of regions, keyed by their ID."""


    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))

    projection = max_intensity_projection(smoothed_z_stack)

    im_thresholded = threshold_otsu(projection)

    connected_components = find_connected_components(im_thresholded)

    return {ccID : Region.select_from_array(connected_components, ccID)
            for ccID in np.unique(connected_components)}

def draw_square(array, coords, colour):

    x, y = coords
    for xo in range(-5, 5):
        for yo in range(-5, 5):
            array[x+xo, y+yo] = colour
    
def apply_mask(image, mask_region):
    """Given an original image and a mask consisting of boolean values, return an
    image obtained by applying the mask to the image."""

    masked_image = np.zeros(image.shape)

    for point in mask_region.points:
        masked_image[point] = image[point]

    return masked_image

def ellipse_box(region):
    """Return the box representing the ellipse (center, bounds, angle)."""

    border = region.border
    border_points = np.array(border.points)
    transposed_points = np.array([(a, b) for (b, a) in border_points])
    return cv2.fitEllipse(transposed_points)

def parameterise_single_stomata(stomata_region):
    """Given a region of interest representing a stomata, parameterise the
    stomata in that region."""

    scipy.misc.imsave('stomata_border.png', stomata_region.border.bitmap)
    box = ellipse_box(stomata_region)

    xdim, ydim = stomata_region.bitmap.shape
    annotated_array = np.zeros((xdim, ydim, 3), dtype=np.uint8)
    annotated_array[stomata_region.border.index_arrays] = 255, 255, 255
    cv2.ellipse(annotated_array, box, (0, 255, 0))

    scipy.misc.imsave('annotated_image.png', annotated_array)

def smoothed_max_intensity_projection(raw_z_stack):
    """Return the smoothed max intensity projection."""
    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))
    projection = max_intensity_projection(smoothed_z_stack)
    return projection

def save_masked_stomata(raw_z_stack, stomata_region):
    """Save an image with only the stomata of interest."""

    projection = smoothed_max_intensity_projection(raw_z_stack)

    dilated_stomata = stomata_region.dilate(iterations=10)
    stomata_image = apply_mask(projection, dilated_stomata)
    scipy.misc.imsave('stomata.png', stomata_image)


def find_stomata(raw_z_stack, region_id):
    """Return a region representing a stomata.
    
    Given an raw z stack, find stomata in it.
    """
    candidate_regions = find_candidate_regions(raw_z_stack)

    stomata_region = candidate_regions[region_id].convex_hull
    save_masked_stomata(raw_z_stack, stomata_region)

    parameterise_single_stomata(stomata_region)
    
    return stomata_region

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    raw_zstack = image_collection.zstack_array(s=30)

    # We know that region 8 is a stomata
    find_stomata(raw_zstack, region_id=8)

if __name__ == "__main__":
    main()
