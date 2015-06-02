import os
import errno

import math

import skimage.measure

from nose.tools import assert_almost_equal

from jicimagelib.image import DataManager
from jicimagelib.geometry import Point2D
from jicimagelib.io import FileBackend

HERE = os.path.dirname(__file__)
UNPACK = os.path.join(HERE, '..', 'data', 'unpack')


STOMATA = (
    dict(region=9, series=range(8, 15)),     # Stomata id 1
    dict(region=14, series=range(8, 15)),    # Stomata id 2
    dict(region=20, series=range(8, 15)),    # Stomata id 3
    dict(region=8, series=range(15, 24)),    # Stomata id 4
    dict(region=17, series=range(15, 24)),   # Stomata id 5
    dict(region=21, series=range(15, 24)),   # Stomata id 6
    None,                                    # Unable to identify stomata 7
    dict(region=8, series=range(24, 36)),    # Stomata id 8
)

def stomata_lookup(stomata_id):
    """Return stomata region and series information."""
    d = STOMATA[stomata_id]
    if d is None:
        raise(NotImplementedError("Have not found this stomata yet"))
    return d["region"], d["series"]


def safe_mkdir(dir_path):

    try:
        os.makedirs(dir_path)
    except OSError, e:
        if e.errno != errno.EEXIST:
            print "Error creating directory %s" % dir_path
            sys.exit(2)

def unpack_data(confocal_file):
    """Unpack the file and return an image collection object."""
    safe_mkdir(UNPACK)

    backend = FileBackend(UNPACK)
    data_manager = DataManager(backend)

    data_manager.load(confocal_file)

    #print data_manager.get_image_proxy(s=1)
    image_collection = data_manager[0]

    return image_collection

def angle2vector(angle):
    """Return the unit vector representation of the angle as x, y pair."""
    radians = (math.pi / 180.0) * angle
    return Point2D( math.cos(radians), math.sin(radians) )

def line(center, angle, length):
    """Return the two points representing the line."""
    center = Point2D(center)
    direction = angle2vector(angle)
    half_length = length/2
    p1 = center - (direction * half_length)
    p2 = center + (direction * half_length)
    return p1, p2

def minor_and_major_lines_from_box(box):
    """Return the lines that cut the box into four quadrants."""

    center, bounds, angle = box
    width, height = bounds

    p1, p2 = line(center, angle, width)
    p3, p4 = line(center, angle+90, height)

    return p1, p2, p3, p4

def line_profile(image, box, linewidth=1):
    """Return minor and major line profiles of an ellipse box."""
    p1, p2, p3, p4 = minor_and_major_lines_from_box(box)

    # Convert to cv2 points to scikit image points.
    ski_p1 = p1[1], p1[0]
    ski_p2 = p2[1], p2[0]
    ski_p3 = p3[1], p3[0]
    ski_p4 = p4[1], p4[0]
    
    minor_profile = skimage.measure.profile_line(image, ski_p1, ski_p2,
        linewidth=linewidth)
    major_profile = skimage.measure.profile_line(image, ski_p3, ski_p4,
        linewidth=linewidth)

    return minor_profile, major_profile

#############################################################################
# Tests
#############################################################################

def test_angle2vector_0_degrees():
    x, y = angle2vector(0)
    assert x == 1.0, "{} != 1.0".format(x)
    assert y == 0.0, "{} != 0.0".format(y)

def test_angle2vector_90_degrees():
    x, y = angle2vector(90)
    assert_almost_equal(x, 0.0)
    assert_almost_equal(y, 1.0)

def test_angle2vector_180_degrees():
    x, y = angle2vector(180)
    assert_almost_equal(x, -1.0)
    assert_almost_equal(y, 0.0)

def test_angle2vector_270_degrees():
    x, y = angle2vector(270)
    assert_almost_equal(x, 0.0)
    assert_almost_equal(y, -1.0)

def test_angle2vector_360_degrees():
    x, y = angle2vector(360)
    assert_almost_equal(x, 1.0)
    assert_almost_equal(y, 0.0)

