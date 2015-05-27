"""Calculate the stomata opening for a time series."""

import sys
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

from jicimagelib.geometry import Point2D

from find_stomata import unpack_data, find_stomata, ellipse_box
from find_inner_region import line_profile

STOMATA = (
    dict(region=9, series=range(8, 14)),   # Stomata id 1
    dict(region=14, series=range(8, 14)),  # Stomata id 2
    dict(region=20, series=range(8, 14))   # Stomata id 3
)

PLOT = True

def ellipse_of_interest(image_collection, series, region):
    """Return the stomata ellipse box."""
    raw_zstack = image_collection.zstack_array(s=series)
    stomata_region = find_stomata(raw_zstack, region)
    return ellipse_box(stomata_region)

def line_profile_x_values(image_collection, box):
    """Return list of x values for line profile plots."""
    im = image_collection.image()
    minor_profile, major_profile = line_profile(im, box)
    return range(len(minor_profile))

def line_profile_mid_point(image_collection, box):
    """Return line profile mid point."""
    xs = line_profile_x_values(image_collection, box)
    mid_pt =  len(xs) / 2.0
    if PLOT:
        plt.axvline(x=mid_pt, c='m', linestyle='--')
    return mid_pt

def projected_average_line_profile(image_collection, series, box):
    """Return the z-stack projected average line profile."""
    total = None
    num_z_slices = 0
    for proxy_im in image_collection.zstack_proxy_iterator(s=series, c=2):
        im = proxy_im.image
        minor_profile, major_profile = line_profile(im, box, 10)
        if total is None:
            total = np.zeros(minor_profile.shape, dtype=float)
        total = total + minor_profile
        num_z_slices = num_z_slices + 1
    average = total / num_z_slices

    if PLOT:
        xs = line_profile_x_values(image_collection, box)
        plt.plot(xs, average)
    return average
            

def local_maxima(profile):
    """Find the local maxima in a 1D profile line."""
    return ((profile >= np.roll(profile, 1))
        & (profile >= np.roll(profile, -1)))

def local_minima(profile):
    """Find the local minima in a 1D profile line."""
    return ((profile <= np.roll(profile, 1))
        & (profile <= np.roll(profile, -1)))

def test_local_maxima():
    ar = np.array([2, 1, 2, 3, 4, 2, 4, 1, 0,])
    maxima = local_maxima(ar)
    assert maxima.tolist() == [True, False, False, False, True, False, True, False, False]

def test_local_minima():
    ar = np.array([2, 1, 2, 3, 4, 2, 4, 1, 0,])
    minima = local_minima(ar)
    assert minima.tolist() == [False, True, False, False, False, True, False, False, True]

def xy_arrays(profile, func):
    """Return x and y arrays of coordinates."""
    xs, = np.where( func(profile) )
    ys = np.take(profile, xs)
    return xs, ys

def midpoint_minima(mid_pt, profile):
    """Return the two minima surrounding the mid point."""
    min_xs, min_ys = xy_arrays(profile, local_minima)
    left_min = None
    right_min = None
    for minima in min_xs:
        left_min = right_min
        right_min = minima
        if (mid_pt - minima) <= 0:
            break
    xs = [left_min, right_min]
    ys = np.take(profile, xs)
    if PLOT:
        plt.plot(xs, ys, "ro")
    return Point2D(xs[0], ys[0]), Point2D(xs[1], ys[1])

def midpoint_maximum(mid_pt, profile):
    """Return the mid point maximum."""
    max_xs, max_ys = xy_arrays(profile, local_maxima)
    prev_diff = sys.maxint
    maximum_x = None
    maximum_y = None
    for x, y in zip(max_xs, max_ys):
        diff = abs(mid_pt - x)
        if diff < prev_diff:
            prev_diff = diff
            maximum_x = x
            maximum_y = y
    if PLOT:
        plt.plot(maximum_x, maximum_y, "go")
    return Point2D(maximum_x, maximum_y)
    
def distance(pt1, pt2):
    """Return x-axis distance in microns."""
    diff = pt1.x - pt2.x
    d2 = diff * diff
    d = math.sqrt(d2)
    microns = d * 0.157
    if PLOT:
        y = min(pt1.y, pt2.y)
        plt.plot([pt1.x, pt2.x],[y, y])
    return microns

def calculate_opening(image_collection, series, box):
    """Calculate the opening of the stomata."""

    mid_pt = line_profile_mid_point(image_collection, box)
    average = projected_average_line_profile(image_collection, series, box)
    left_minima, right_minima = midpoint_minima(mid_pt, average)
    maximum = midpoint_maximum(mid_pt, average)
    opening = distance(left_minima, right_minima)
    print("Opening: {} um".format(opening))


def analyse_all(image_collection, stomata_id):
    """Analyse all stomata in a time series."""
    first = STOMATA[stomata_id]["series"][0]
    region = STOMATA[stomata_id]["region"]
    box = ellipse_of_interest(image_collection, first, region)
    for series in STOMATA[stomata_id]["series"]:
        calculate_opening(image_collection, series, box)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('stomata_id', type=int, help='Zero based stomata index')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    analyse_all(image_collection, args.stomata_id)
