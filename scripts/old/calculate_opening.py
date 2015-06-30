"""Calculate the stomata opening for a time series."""

import sys
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from jicimagelib.geometry import Point2D
from jicimagelib.io import AutoWrite
AutoWrite.on = False

from util import (
    unpack_data,
    ellipse_line_profiles,
    stomata_lookup,
)
from find_stomata import find_stomata, ellipse_box

COLOR = 'b'

def ellipse_of_interest(image_collection, series, region):
    """Return the stomata ellipse box."""
    raw_zstack = image_collection.zstack_array(s=series)
    stomata_region = find_stomata(raw_zstack, region)
    return ellipse_box(stomata_region)

def line_profile_x_values(image_collection, box):
    """Return list of x values for line profile plots."""
    im = image_collection.image()
    minor_profile, major_profile = ellipse_line_profiles(im, box)
    return range(len(minor_profile))

def line_profile_mid_point(image_collection, box, plot=False):
    """Return line profile mid point."""
    xs = line_profile_x_values(image_collection, box)
    mid_pt =  len(xs) / 2.0
    if plot:
        plt.axvline(x=mid_pt, c='k', linestyle='--')
    return mid_pt

def projected_average_line_profile(image_collection, series, box, plot=False):
    """Return the z-stack projected average line profile."""
    total = None
    num_z_slices = 0
    for proxy_im in image_collection.zstack_proxy_iterator(s=series, c=2):
        im = proxy_im.image
        minor_profile, major_profile = ellipse_line_profiles(im, box, 10)
        if total is None:
            total = np.zeros(minor_profile.shape, dtype=float)
        total = total + minor_profile
        num_z_slices = num_z_slices + 1
    average = total / num_z_slices

    if plot:
        xs = line_profile_x_values(image_collection, box)
        plt.plot(xs, average, color=COLOR)
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

def midpoint_minima(mid_pt, profile, plot=False):
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
    if plot:
        plt.plot(xs, ys, marker="v", linestyle="_", color=COLOR)
    return Point2D(xs[0], ys[0]), Point2D(xs[1], ys[1])

def midpoint_maximum(mid_pt, profile, plot=False):
    """Return the mid point maximum."""
    left_min, right_min = midpoint_minima(mid_pt, profile)
    profile_of_interest = profile[left_min.x:right_min.x+1]
    max_xs, max_ys = xy_arrays(profile_of_interest, local_maxima)
    assert len(max_xs) == 1, "More than one maxima in between two neighboring minima!"
    max_xs[0] += left_min.x
    if plot:
        plt.plot(max_xs, max_ys, marker="^", linestyle="_", color=COLOR)
    return Point2D(max_xs[0], max_ys[0])

def half_height(pt1, pt2):
    """Return the half height of the peak."""
    tmp_pt = (pt1 + pt2) / 2.0
    return tmp_pt.y

def test_helf_height():
    p1 = Point2D(1.0, 3.4)
    p2 = Point2D(5.0, 1.2)
    assert half_height(p1, p2) == 2.3

def y_values_between_points(average, left_pt, right_pt):
    """Return the list of values between left_pt and right_pt."""
    start = int(math.floor(left_pt.x))
    end = int(math.ceil(right_pt.x)) + 1
    return [average[i] for i in range(start, end)]

def test_y_values_between_points():
    a = [float(i) for i in range(7)]
    p1 = Point2D(1.0, 3.4)
    p2 = Point2D(5.0, 1.2)
    ys = y_values_between_points(a, p1, p2)
    assert ys == [1., 2., 3., 4., 5.]
    
def closest_observed_points(y_values, target_height):
    """Return the observed points neighboring the target height."""
    initial_sign = y_values[0] - target_height
    prev_diff = initial_sign * sys.maxint
    prev_point = None
    for i, y in enumerate(y_values):
        diff = y - target_height
        if prev_diff/abs(prev_diff) != diff/abs(diff):
            # Change of sign means that we have moved past the target value.
            return prev_point, Point2D(i, y)
        if abs(diff) < abs(prev_diff):
            # We are one step closer to the closest observed point.
            prev_point = Point2D(i, y)
            pref_diff = diff

def test_closest_observed_points():
    a = [float(i) for i in range(7)]
    t = 4.5
    p1, p2 = closest_observed_points(a, t)
    assert p1 == Point2D(4., 4.)
    assert p2 == Point2D(5., 5.)

    r = [i for i in reversed(a)]
    # r = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    p1, p2 = closest_observed_points(r, t)
    assert p1 == Point2D(1., 5.), "{} != Ponint2D(5., 5.)".format(p1)
    assert p2 == Point2D(2., 4.), "{} != Ponint2D(4., 4.)".format(p2)

def optimised_point(left_pt, right_pt, target_height):
    """Return a point with y value within 0.001 of the target height."""
    mean_pt = ( left_pt + right_pt ) / 2.0
    diff = mean_pt.y - target_height
    if abs(diff) > 0.001:
        if left_pt.y < right_pt.y:
            # Upward slope
            if target_height < mean_pt.y:
                # Left hand side.
                mean_pt = optimised_point(left_pt, mean_pt, target_height) 
            else:
                # Right hand side.
                mean_pt = optimised_point(right_pt, mean_pt, target_height) 
        else:
            # Downward slope.
            if target_height > mean_pt.y:
                # Left hand side.
                mean_pt = optimised_point(left_pt, mean_pt, target_height) 
            else:
                # Right hand side.
                mean_pt = optimised_point(right_pt, mean_pt, target_height) 
    return mean_pt

def test_optimised_point():

    # Slope up left hand side.
    p1 = Point2D(4.0, 4.0)
    p2 = Point2D(8.0, 8.0)
    p3 = optimised_point(p1, p2, 5.0)
    assert round(p3.x, 2) == 5.0, "{} != 5.0".format(p3.x)
    assert round(p3.y, 2) == 5.0, "{} != 5.0".format(p3.y)

    # Slope up right hand side.
    p3 = optimised_point(p1, p2, 7.0)
    assert round(p3.x, 2) == 7.0, "{} != 7.0".format(p3.x)
    assert round(p3.y, 2) == 7.0, "{} != 7.0".format(p3.y)

    # Slope down left hand side.
    p1 = Point2D(4.0, 8.0)
    p2 = Point2D(8.0, 4.0)
    p3 = optimised_point(p1, p2, 7.0)
    assert round(p3.x, 2) == 5.0
    assert round(p3.y, 2) == 7.0
    
    # Slope down right hand side.
    p1 = Point2D(4.0, 8.0)
    p2 = Point2D(8.0, 4.0)
    p3 = optimised_point(p1, p2, 5.0)
    assert round(p3.x, 2) == 7.0
    assert round(p3.y, 2) == 5.0
    
def peak_half_height(average, left_pt, right_pt, plot=False):
    """Return the half peak height point."""
    target_height = half_height(left_pt, right_pt)
    x_offset = min(left_pt.x, right_pt.x)
    ys = y_values_between_points(average, left_pt, right_pt)
    left_pt, right_pt = closest_observed_points(ys, target_height)
    half_height_pt =  optimised_point(left_pt, right_pt, target_height)
    half_height_pt.x = half_height_pt.x + x_offset
    if plot:
        plt.plot(half_height_pt.x, half_height_pt.y, marker="o", linestyle="_",
            color=COLOR)
    return half_height_pt
    
def opening_points(image_collection, series, box, plot=False):
    """Return the two points representing the opening."""
    mid_pt = line_profile_mid_point(image_collection, box, plot=plot)
    average = projected_average_line_profile(image_collection, series, box,
        plot=plot)
    left_minima, right_minima = midpoint_minima(mid_pt, average, plot=plot)
    maximum = midpoint_maximum(mid_pt, average, plot=plot)

    assert left_minima.x < maximum.x, "Left minima to the right of the maximum."

    left_half_height = peak_half_height(average, left_minima, maximum,
        plot=plot)
    right_half_height = peak_half_height(average, maximum, right_minima,
        plot=plot)
    return left_half_height, right_half_height

def distance(pt1, pt2, plot=False):
    """Return x-axis distance in microns."""
    diff = pt1.x - pt2.x
    d2 = diff * diff
    d = math.sqrt(d2)
    microns = d * 0.157
    if plot:
        y = min(pt1.y, pt2.y)
        plt.plot([pt1.x, pt2.x],[y, y], color=COLOR)
    return microns

def calculate_opening(image_collection, stomata_id, series_id, plot=False):
    """Calculate the opening of the stomata."""
    region_id, series_identifiers = stomata_lookup(stomata_id)
    first = series_identifiers[0]
    box = ellipse_of_interest(image_collection, first, region_id)
    pt1, pt2 = opening_points(image_collection, series_id, box, plot=plot)
    d = distance(pt1, pt2, plot=plot)
    return d

def analyse(image_collection, stomata_id, timepoint):
    """Analyse all stomata in a time series."""
    region_id, series_identifiers = stomata_lookup(stomata_id)
    colors = sns.cubehelix_palette(len(series_identifiers), start=0, light=0.8, reverse=True)
    for i, series_id in enumerate(series_identifiers):
        if (timepoint is not None) and (timepoint != i):
            continue
        global COLOR
        COLOR = colors[i]
        d = calculate_opening(image_collection, stomata_id, series_id, plot=True)
        print("Distance: {:.2f} um".format(d))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('stomata_id', type=int, help='Zero based stomata index')
    parser.add_argument('-t', '--timepoint', type=int, default=None,
        help='Zero based time point index')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    analyse(image_collection, args.stomata_id, args.timepoint)
    
