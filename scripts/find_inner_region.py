"""Script to calculate a line profiles across stomata."""

import argparse
import math

from nose.tools import assert_almost_equal

import numpy as np
import cv2
import scipy
import skimage.transform
import skimage.measure

from jicimagelib.geometry import Point2D

from find_stomata import (
    unpack_data,
    find_candidate_regions,
    ellipse_box,
    gaussian_filter,
    max_intensity_projection,
)

def angle2vector(angle):
    """Return the unit vector representation of the angle as x, y pair."""
    radians = (math.pi / 180.0) * angle
    return Point2D( math.cos(radians), math.sin(radians) )

def quadrant_lines_from_box(box):
    """Return the lines that cut the box into four quadrants."""

    def line(center, angle, length):
        """Return the two points representing the line."""
        center = Point2D(center)
        direction = angle2vector(angle)
        half_length = length/2
        p1 = center - (direction * half_length)
        p2 = center + (direction * half_length)
        return p1.astype("int"), p2.astype("int")

    center, bounds, angle = box
    width, height = bounds

    p1, p2 = line(center, angle, width)
    p3, p4 = line(center, angle+90, height)

    return p1, p2, p3, p4

def line_profile(confocal_file):
    """Return minor and major line profiles of a stomata."""

    image_collection = unpack_data(confocal_file)
    raw_z_stack = image_collection.zstack_array(s=30)
    candidate_regions = find_candidate_regions(raw_z_stack)

    # We know that region 8 is a stomata.
    stomata_region = candidate_regions[8].convex_hull

    # Annotated array to plot on top of.
    xdim, ydim = stomata_region.bitmap_array.shape
    annotated_array = np.zeros((xdim, ydim, 3), dtype=np.uint8)
    annotated_array[stomata_region.border.coord_elements] = 255, 255, 255

    box = ellipse_box(stomata_region)
    p1, p2, p3, p4 = quadrant_lines_from_box(box)

    cv2.line(annotated_array, p1.astuple(), p2.astuple(), (0, 255, 0), 1)
    cv2.line(annotated_array, p3.astuple(), p4.astuple(), (255, 0, 0), 1)
    scipy.misc.imsave('quadrant_lines_image.png', annotated_array)

    # Image to be analyzed.
    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))
    projection = max_intensity_projection(smoothed_z_stack)

    ski_p1 = p1[1], p1[0]
    ski_p2 = p2[1], p2[0]
    ski_p3 = p3[1], p3[0]
    ski_p4 = p4[1], p4[0]
    
    minor_profile = skimage.measure.profile_line(projection, ski_p1, ski_p2)
    major_profile = skimage.measure.profile_line(projection, ski_p3, ski_p4)

    return minor_profile, major_profile
    
    

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    unpack_data(args.confocal_file)
    minor_profile, major_profile = line_profile(args.confocal_file)

    for height in minor_profile:
        print "X"*height

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

if __name__ == "__main__":
    main()
