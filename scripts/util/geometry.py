"""Custom geometry operations for identifying stomata."""

import math

import numpy as np
import cv2

from nose.tools import assert_almost_equal

from jicimagelib.geometry import Point2D

def ellipse_box(region):
    """Return the box representing the ellipse (center, bounds, angle)."""

    border = region.border
    border_points = np.array(border.points)
    transposed_points = np.array([(a, b) for (b, a) in border_points])
    return cv2.fitEllipse(transposed_points)

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

# Tests...

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

