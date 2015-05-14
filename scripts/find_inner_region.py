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
    find_stomata,
    ellipse_box,
    smoothed_max_intensity_projection,
)

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

def quadrant_lines_from_box(box):
    """Return the lines that cut the box into four quadrants."""

    center, bounds, angle = box
    width, height = bounds

    p1, p2 = line(center, angle, width)
    p3, p4 = line(center, angle+90, height)

    return p1, p2, p3, p4

def annotated_region_image(region):
    """Return an annotated region image to plot on top of."""
    xdim, ydim = region.bitmap_array.shape
    annotated_array = np.zeros((xdim, ydim, 3), dtype=np.uint8)
    annotated_array[region.border.coord_elements] = 255, 255, 255
    return annotated_array

def save_major_and_minor_lines(annotated_array, box):
    """Save image with minor and major lines."""
    p1, p2, p3, p4 = quadrant_lines_from_box(box)

    cv2.line(annotated_array, p1.astype("int").astuple(), p2.astype("int").astuple(), (0, 255, 0), 1)
    cv2.line(annotated_array, p3.astype("int").astuple(), p4.astype("int").astuple(), (255, 0, 0), 1)
    scipy.misc.imsave('quadrant_lines_image.png', annotated_array)

def line_profile(image, box):
    """Return minor and major line profiles of a stomata."""
    p1, p2, p3, p4 = quadrant_lines_from_box(box)

    # Convert to cv2 points to scikit image points.
    ski_p1 = p1[1], p1[0]
    ski_p2 = p2[1], p2[0]
    ski_p3 = p3[1], p3[0]
    ski_p4 = p4[1], p4[0]
    
    minor_profile = skimage.measure.profile_line(image, ski_p1, ski_p2)
    major_profile = skimage.measure.profile_line(image, ski_p3, ski_p4)

    return minor_profile, major_profile
    
def find_relative_profile_length(profile):
    """Return relative cut points from a line profile."""

    # Find relative cut points in the profile.
    otsu_cutoff = skimage.filters.threshold_otsu(profile)
    thresholded_profile = profile > otsu_cutoff
    relative_cut_points = []
    current = True
    for i, high_intensity in enumerate(thresholded_profile):
        if current == high_intensity:
            continue
        relative_cut_points.append(i/float(len(profile)))
        current = not current
    assert len(relative_cut_points) == 2, \
         "Expected 2 relative cut points, not: {}".format(len(relative_cut_points))

    # Make the cuts symmetric by averaging.
    rel1 = 1.0 - relative_cut_points[0]
    rel2 = relative_cut_points[1]
    relative_length = (rel1 + rel2) / 2.0
    return relative_length


def find_inner_region(raw_zstack):
    """Given an image collection, identify the inner region of a stomata."""
    # We know that region 8 is a stomata.
    stomata_region = find_stomata(raw_zstack, region_id=8)
    projection = smoothed_max_intensity_projection(raw_zstack)
    annotated_array = annotated_region_image(stomata_region)
    box = ellipse_box(stomata_region)
    save_major_and_minor_lines(annotated_array, box)
    minor_profile, major_profile = line_profile(projection, box)

    minor_rel_length = find_relative_profile_length(minor_profile)
    
    print("Minor relative length: {}".format(minor_rel_length))



    

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    raw_zstack = image_collection.zstack_array(s=30)
    find_inner_region(raw_zstack)



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
