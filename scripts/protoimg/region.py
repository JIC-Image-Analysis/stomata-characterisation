
import unittest

from nose.tools import raises
import scipy.ndimage as nd
import numpy as np
import skimage

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
        """Region formed by taking non-border elements."""

        inner_array = nd.morphology.binary_erosion(self.bitmap_array)
        return Region(inner_array)

    @property
    def border(self):
        """Region formed by taking border elements."""

        border_array = self.bitmap_array - self.inner.bitmap_array
        return Region(border_array)

    @property
    def convex_hull(self):
        hull_array = skimage.morphology.convex_hull_image(self.bitmap_array)
        return Region(hull_array)

    @property
    def area(self):
        """Number of non-zero elements."""

        return np.count_nonzero(self.bitmap_array)

    @property
    def coord_elements(self):
        """All nonzero elements as a pair of arrays."""

        return np.where(self.bitmap_array == True)

    @property
    def coord_list(self):
        return zip(*self.coord_elements)

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

@raises(Exception)
def test_region_constructor():

    test_array = np.array([[0, 1, 2],
                           [0, 0, 1],
                           [0, 0, 0]])

    Region(test_array)
