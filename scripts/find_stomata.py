"""Given a confocal image, find stomata within that image."""

import os
import sys
import math
import timeit
import argparse
import unittest

import numpy as np

from nose.tools import raises

import cv2

import scipy.misc
import scipy.ndimage as nd

import skimage.measure
import skimage.filters
import skimage.morphology

from protoimg.region import Region

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

@raises(Exception)
def test_region_constructor():

    test_array = np.array([[0, 1, 2],
                           [0, 0, 1],
                           [0, 0, 0]])

    Region(test_array)

def find_candidate_regions(raw_z_stack):
    """Given the z stack, find regions in the image which are candidate stomata.
    Return a dictionary of regions, keyed by their ID."""


    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))

    projection = max_intensity_projection(smoothed_z_stack)

    im_thresholded = threshold_otsu(projection)

    connected_components = find_connected_components(im_thresholded)

    return {ccID : Region.from_id_array(connected_components, ccID)
            for ccID in np.unique(connected_components)}

def draw_square(array, coords, colour):

    x, y = coords
    for xo in range(-5, 5):
        for yo in range(-5, 5):
            array[x+xo, y+yo] = colour
    
def apply_mask(image, mask_region):

    masked_image = np.zeros(image.shape)

    for point in mask_region.coord_list:
        masked_image[point] = image[point]

    return masked_image
        
def find_stomata(confocal_file):
    """Given the confocal image file, find stomata in it."""

    image_collection = unpack_data(confocal_file)
    raw_z_stack = image_collection.zstack_array(s=30)
    candidate_regions = find_candidate_regions(raw_z_stack)

    stomata_region = candidate_regions[8].convex_hull

    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))
    projection = max_intensity_projection(smoothed_z_stack)

    dilated_stomata = stomata_region.dilate(iterations=20)
    annotated_image = apply_mask(projection, dilated_stomata)

    scipy.misc.imsave('stomata.png', annotated_image)

    # _, _, zdim = raw_z_stack.shape

    # for z in range(zdim):
    #     scipy.misc.imsave('plane{}.png'.format(z), apply_mask(raw_z_stack[:,:,z], stomata_region))

def more_stuff():
    border = stomata_region.border

    border_points = np.array(border.coord_list)
    border_points = np.array([(b, a) for a, b in border_points])


    for x in range(10):
        center, bounds, angle = cv2.fitEllipse(border_points)
        box = (center, bounds, angle)

    for x in range(10):
        em = skimage.measure.EllipseModel()
        em.estimate(border_points)

    annotated_array = 255 * border.bitmap_array.astype(np.uint8)

    cv2.ellipse(annotated_array, box, 255)
    #draw_square(annotated_array, center, 255)

    angle_r = 2 * math.pi * (angle / 360)
    x, y = center

    print box

    for xo in range(100):
        yo = xo * -math.sin(angle_r)
        annotated_array[x+xo,y+yo] = 255

    scipy.misc.imsave('ellipse.png', annotated_array)

    


def stuff():
    def shape_factor(region):
        S = region.area
        L = region.perimeter

        return (4 * math.pi * S) / (L * L)

    composite = np.zeros(connected_components.shape)

    def separate_regions():
        for ccID in np.unique(connected_components):
            r = Region.from_id_array(connected_components, ccID)
            dr = r.dilate(10)
            h = r.convex_hull
            pre_ratio = float(r.area) / r.perimeter
            after_ratio = float(dr.area) / dr.perimeter

            #print ccID, pre_ratio, after_ratio / pre_ratio, float(h.area) / h.perimeter
            print ccID, h.area, h.perimeter, shape_factor(h)
            scipy.misc.imsave('dr{}.png'.format(ccID), dr.border.bitmap_array)
            scipy.misc.imsave('h{}.png'.format(ccID), r.convex_hull.border.bitmap_array)

            composite[h.coord_list] = ccID

        scipy.misc.imsave('composite.png', composite)

    def fit_measure(ccID):
        r = Region.from_id_array(connected_components, ccID)
        h = r.convex_hull
        em = skimage.measure.EllipseModel()
        data = zip(*h.border.coord_list)
        em.estimate(np.array(data))
        
        residuals = em.residuals(np.array(data))

        return np.inner(residuals, residuals) / (len(residuals) ** 2)


    stomata_ids = []

    def print_all_residuals():
        for ccID in np.unique(connected_components):
            r = Region.from_id_array(connected_components, ccID)
            if 100 < r.convex_hull.perimeter < 400:
                fm = fit_measure(ccID)
                print ccID, r.convex_hull.perimeter, fm
                if fm < 0.01:
                    stomata_ids.append(ccID)

    print_all_residuals()

    xdim, ydim = connected_components.shape
    annotated = np.zeros((xdim, ydim, 3))
    for ccID in stomata_ids:
        r = Region.from_id_array(connected_components, ccID)
        h = r.convex_hull
        em = skimage.measure.EllipseModel()
        data = zip(*h.border.coord_list)
        em.estimate(np.array(data))
        t_range = np.linspace(0, 2 * math.pi, 500)
        predicted = em.predict_xy(t_range).astype(np.uint16)
        annotated[h.border.coord_list] = 255, 255, 255
        annotated[zip(*predicted)] = 255, 0, 0
    scipy.misc.imsave('annotated.png', annotated)

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')


    args = parser.parse_args()

    unpack_data(args.confocal_file)
    find_stomata(args.confocal_file)

if __name__ == "__main__":
    main()
