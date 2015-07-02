"""Array helper functions."""

import sys
import math
import numpy as np

from jicimagelib.geometry import Point2D

def local_maxima(profile):
    """Find the local maxima in a 1D profile line."""
    return ((profile >= np.roll(profile, 1))
        & (profile >= np.roll(profile, -1)))

def local_minima(profile):
    """Find the local minima in a 1D profile line."""
    return ((profile <= np.roll(profile, 1))
        & (profile <= np.roll(profile, -1)))

def xy_arrays(profile, func):
    """Return x and y arrays of coordinates."""
    xs, = np.where( func(profile) )
    ys = np.take(profile, xs)
    return xs, ys


def half_height(pt1, pt2):
    """Return the half height of the peak."""
    tmp_pt = (pt1 + pt2) / 2.0
    return tmp_pt.y

def y_values_between_points(average, left_pt, right_pt):
    """Return the list of values between left_pt and right_pt."""
    start = int(math.floor(left_pt.x))
    end = int(math.ceil(right_pt.x)) + 1
    return [average[i] for i in range(start, end)]

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

def rmsd(ar1, ar2):
    """Return the root mean square deviation between two arrays."""
    return np.sqrt( np.sum( (ar1 - ar2)**2 )  / ar1.size )

# Tests...

def test_local_maxima():
    ar = np.array([2, 1, 2, 3, 4, 2, 4, 1, 0,])
    maxima = local_maxima(ar)
    assert maxima.tolist() == [True, False, False, False, True, False, True, False, False]

def test_local_minima():
    ar = np.array([2, 1, 2, 3, 4, 2, 4, 1, 0,])
    minima = local_minima(ar)
    assert minima.tolist() == [False, True, False, False, False, True, False, False, True]

def test_helf_height():
    p1 = Point2D(1.0, 3.4)
    p2 = Point2D(5.0, 1.2)
    assert half_height(p1, p2) == 2.3

def test_y_values_between_points():
    a = [float(i) for i in range(7)]
    p1 = Point2D(1.0, 3.4)
    p2 = Point2D(5.0, 1.2)
    ys = y_values_between_points(a, p1, p2)
    assert ys == [1., 2., 3., 4., 5.]
    
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
    
def test_rmsd():
    ar0 = np.zeros((5,))
    assert rmsd(ar0, ar0) == 0.
    ar1 = np.ones((5,))
    assert rmsd(ar1, ar1) == 0.
    assert rmsd(ar0, ar1) == 1., rmsd(ar0, ar1)
    assert rmsd(ar1, ar0) == 1.
    ar2 = np.ones((5,)) * 2
    assert rmsd(ar2, ar0) == 2., rmsd(ar2, ar0)
    assert rmsd(ar0, ar2) == 2., rmsd(ar0, ar2)

