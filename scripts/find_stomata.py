"""Given a confocal image, find stomata within that image."""

import os
import sys
import math
import argparse
import unittest

import numpy as np

from nose.tools import raises

import scipy.misc
import scipy.ndimage as nd

import skimage.measure
import skimage.filters
import skimage.morphology

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

class Region(object):
    """Class representing a particular point of interest in an image, 
    represented as a bitmask with 1 indicating areas in the region."""

    def __init__(self, bitmap_array):
        bitmap_values = set(np.unique(bitmap_array))
        if len(bitmap_values - set([0, 1])):
            raise Exception('Region bitmap must have only 0 and 1 values')

        self.bitmap_array = bitmap_array.astype(bool)

    @classmethod
    def from_id_array(cls, id_array, id):
        """Initialise from an array where each unique value represents a
        region."""

        base_array = np.zeros(id_array.shape)
        array_coords = np.where(id_array == id)
        base_array[array_coords] = 1

        return cls(base_array)

    @property
    def inner(self):
        inner_array = nd.morphology.binary_erosion(self.bitmap_array)
        return Region(inner_array)

    @property
    def border(self):
        border_array = self.bitmap_array - self.inner.bitmap_array
        return Region(border_array)

    @property
    def convex_hull(self):
        hull_array = skimage.morphology.convex_hull_image(self.bitmap_array)
        return Region(hull_array)

    @property
    def area(self):
        return np.count_nonzero(self.bitmap_array)

    @property
    def coord_list(self):
        return np.where(self.bitmap_array == True)

    @property
    def perimeter(self):
        return self.border.area

    def dilate(self, iterations=1):
        dilated_array = nd.morphology.binary_dilation(self.bitmap_array, 
                                                      iterations=iterations)
        return Region(dilated_array)

    def __repr__(self):
        return self.bitmap_array.__repr__()

    def __str__(self):
        return self.bitmap_array.__str__()

class RegionTestCase(unittest.TestCase):

    def test_region(self):

        test_array = np.array([[0, 1, 1],
                               [0, 0, 1],
                               [0, 0, 0]])

        region = Region(test_array)

        bitmap_array = region.bitmap_array

        self.assertFalse(bitmap_array[0, 0])
        self.assertTrue(bitmap_array[0, 1])
        self.assertEqual(bitmap_array.shape, (3, 3))

def test_region_from_id_array():
    id_array = np.array([[0, 0, 0],
                         [1, 1, 1],
                         [2, 2, 2]])

    region_1 = Region.from_id_array(id_array, 1)

    assert(region_1.bitmap_array[0, 0] == False)
    assert(region_1.bitmap_array[1, 0] == True)
    assert(region_1.bitmap_array[2, 0] == False)

    assert(region_1.area == 3)

def test_region_area():

    test_array = np.array([[0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0]])

    region = Region(test_array)

    assert(region.area == 3)

def test_region_perimeter():

    test_array = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

    region = Region(test_array)

    assert(region.perimeter == 8)

def test_region_border():

    test_array = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

    region = Region(test_array)

    border_array = np.array([[0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0]])

    border_region = Region(border_array)

    assert(np.array_equal(region.border.bitmap_array, border_region.bitmap_array))

def test_region_inner():

    test_array = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

    region = Region(test_array)

    inner_array = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

    inner_region = Region(inner_array)

    assert(np.array_equal(region.inner.bitmap_array, inner_region.bitmap_array))

def devlet():
    test_array = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 1, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

    region = Region(test_array)

    inner = nd.morphology.binary_erosion(region.bitmap_array)
    
    sys.exit(0)

@raises(Exception)
def test_region_constructor():

    test_array = np.array([[0, 1, 2],
                           [0, 0, 1],
                           [0, 0, 0]])

    Region(test_array)

def find_stomata(confocal_file):
    """Given the confocal image file, find stomata in it."""

    image_collection = unpack_data(confocal_file)

    raw_z_stack = image_collection.zstack_array(s=30)
    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))

    projection = max_intensity_projection(smoothed_z_stack)

    im_thresholded = threshold_otsu(projection)

    connected_components = find_connected_components(im_thresholded)

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
