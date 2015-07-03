"""Line profile helper funcitons."""

import numpy as np
import scipy.ndimage
import skimage.measure

from jicimagelib.geometry import Point2D
from jicimagelib.util.array import normalise

from util.geometry import angle2vector

class LineProfile(object):
    """Class for working with line profiles."""

    def __init__(self, ys, identifier=None):
        if isinstance(ys, LineProfile):
            # Create a LineProfile from a LineProfile.
            self.ys = ys.ys
            self.include = ys.include
        else:
            # Create a line profile form a numpy array.
            self.ys = ys
            self.include = True
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
        identifer = "normalised({})".format(self.identifier)
        return LineProfile(normalise(self.ys), identifer)

    @property
    def smoothed_gaussian(self):
        """Return normalised Gaussian smoothed line profile."""
        ar = scipy.ndimage.filters.gaussian_filter(self.ys, 1.0)
        identifer = "smoothed_gaussian({})".format(self.identifier)
        return LineProfile(ar, identifer)

    def __repr__(self):
        return "<LineProfile(identifier={})>".format(self.identifier)

class LineProfileCollection(list):
    
    def add_line_profile(self, line_profile):
        """Add a line profile to the collection."""
        self.append(line_profile)

    def average(self, data='ys'):
        """Return the average line profile."""
        total = np.zeros(self[0].ys.shape, dtype=float)
        num = 0 
        for line_profile in self:
            if line_profile.include:
                total += getattr(line_profile, data).ys
                num += 1
        average = total / float(num)
        return LineProfile(average, "average")
