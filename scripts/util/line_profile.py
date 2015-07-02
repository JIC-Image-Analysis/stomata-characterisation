"""Line profile helper funcitons."""

import numpy as np
import scipy.ndimage
import skimage.measure

from jicimagelib.geometry import Point2D
from jicimagelib.util.array import normalise

from util.geometry import angle2vector

from util.array import (
    xy_arrays,
    local_minima,
    local_maxima,
)


class LineProfile(object):
    """Class for working with line profiles."""

    def __init__(self, ys, identifier=None):
        self.ys = ys
        self.identifier = identifier

    @property
    def xs(self):
        """Return x values."""
        return range(len(self.ys))

    @property
    def mid_point(self):
        """Return the mid point."""
        return len(self.ys) / 2.0

    @property
    def normalised(self):
        """Return normalised line profile."""
        return normalise(self.ys)

    @property
    def smoothed_gaussian(self):
        """Return normalised Gaussian smoothed line profile."""
        return scipy.ndimage.filters.gaussian_filter(self.normalised, 2.0)

    def __repr__(self):
        return "<LineProfile(identifier={})>".format(self.identifier)

class LineProfileCollection(list):
    
    def add_line_profile(self, line_profile):
        """Add a line profile to the collection."""
        self.append(line_profile)

    def average(self, data='ys'):
        """Return the average line profile."""
        total = np.zeros(self[0].ys.shape, dtype=float)
        for line_profile in self:
            total += getattr(line_profile, data)
        return LineProfile(total / len(self))

def midpoint_minima(profile):
    """Return the two minima surrounding the mid point."""
    min_xs, min_ys = xy_arrays(profile.ys, local_minima)
    left_min = None
    right_min = None
    for minima in min_xs:
        left_min = right_min
        right_min = minima
        if (profile.mid_point - minima) <= 0:
            break
    xs = [left_min, right_min]
    ys = np.take(profile.ys, xs)
    return Point2D(xs[0], ys[0]), Point2D(xs[1], ys[1])

def midpoint_maximum(profile):
    """Return the mid point maximum."""
    left_min, right_min = midpoint_minima(profile)
    values_of_interest = profile.ys[left_min.x:right_min.x+1]
    max_xs, max_ys = xy_arrays(values_of_interest, local_maxima)
    assert len(max_xs) == 1, "More than one maxima in between two neighboring minima!"
    max_xs[0] += left_min.x
    return Point2D(max_xs[0], max_ys[0])
