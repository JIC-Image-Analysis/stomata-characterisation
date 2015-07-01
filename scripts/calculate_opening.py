"""Calculate the opening of a stomate."""

import sys
import os
import os.path
import argparse
import math

import numpy as np

import scipy.ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import skimage.measure

from jicimagelib.io import AutoWrite, AutoName
from jicimagelib.transform import max_intensity_projection
from jicimagelib.util.array import normalise
from jicimagelib.geometry import Point2D

from util import (
    STOMATA,
    unpack_data,
    stomata_timeseries_lookup,
)

from util.line_profile import (
    LineProfile,
    LineProfileCollection,
    midpoint_minima,
    midpoint_maximum,
)

from util.geometry import line
from util.array import peak_half_height, rmsd

from stomata_finder import find_stomate_ellipse_box


class StomateOpening(object):
    """Class for calculating the stomate opening."""
    
    def __init__(self, image_collection, stomate_id, timepoint):
        self.image_collection = image_collection
        self.stomata_timeseries = stomata_timeseries_lookup(stomate_id)
        self.stomate = self.stomata_timeseries.stomate_at_timepoint(timepoint)
        
        self.removed_raw_line_profiles = []
        self.removed_line_profiles = []

        self.ellipse_box_init()
        self.minor_axis_pts_init()
        self.line_profiles_init()
        self.best_line_profile_init()
        self.minima_maximum_init()
        self.opening_pts_init()
        self.opening_distance_init()

    def ellipse_box_init(self):
        """Initialise the stomate ellipse box."""
        flourescent_zstack = self.image_collection.zstack_array(
            s=self.stomate.series, c=0)
        self.box = find_stomate_ellipse_box(
            flourescent_zstack, self.stomate.x, self.stomate.y)

    def minor_axis_pts_init(self):
        """Initialise the minor axis pts."""
        center, bounds, angle = self.box
        width, height = bounds
        self.minor_axis_p1, self.minor_axis_p2 = line(center, angle, width)
        
    def line_profiles_init(self):
        """Initialise the stomata minor axis line profiles."""
        self.line_profiles = LineProfileCollection()
        self.ignored_line_profiles = LineProfileCollection()
        self.raw_line_profiles = LineProfileCollection()

        zstack = [z for z in self.image_collection.zstack_proxy_iterator(
            self.stomate.series, c=2)]

        fraction_to_exclude = 0.2
        exclude_int = int(round(len(zstack) * fraction_to_exclude / 2.))
        to_include = range(exclude_int, len(zstack)-exclude_int)
        print "exclude_int", exclude_int
        print "total range", range(len(zstack))
        print "include range", to_include

        for i, zslice in enumerate(zstack):
            # Convert from cv2 points to scikit image points.
            ski_p1 = self.minor_axis_p1[1], self.minor_axis_p1[0]
            ski_p2 = self.minor_axis_p2[1], self.minor_axis_p2[0]

            profile = skimage.measure.profile_line( zslice.image, ski_p1,
                ski_p2, linewidth=10)
            profile = normalise(profile)
            line_profile = LineProfile(profile, i)

            if i in to_include:
                # Save the raw, albeit normalised, profile.
                self.raw_line_profiles.add_line_profile(line_profile)

                # Save the Gaussian filtered profile.
                profile = scipy.ndimage.filters.gaussian_filter(profile, 2.0)
                line_profile = LineProfile(profile, i)
                self.line_profiles.add_line_profile(line_profile)
            else:
                # Log the ignored line profile.
                self.ignored_line_profiles.add_line_profile(line_profile)


        self.average_line_profile_init()
        self.minima_maximum_init()

    def average_line_profile_init(self):
        """Calculate the average line profile."""
        self.average_line_profile = self.line_profiles.average()

    def best_line_profile_init(self):
        """Iteratively remove the worst fitting line profiles."""
        while len(self.line_profiles) > 10:
            worst_line_profile = None
            worst_rmsd = -1000
            worst_index = None
            start = self.left_minima.x
            end = self.right_minima.x
            for i, line_profile in enumerate(self.line_profiles):
                r = rmsd(line_profile.ys[start:end],
                    self.average_line_profile.ys[start:end])
                if r > worst_rmsd:
                    worst_rmsd = r
                    worst_line_profile = line_profile
                    worst_index = i
            self.removed_line_profiles.append(
                self.line_profiles.pop(worst_index))
            self.removed_raw_line_profiles.append(
                self.raw_line_profiles.pop(worst_index))
            self.average_line_profile_init()
            self.minima_maximum_init()

    def minima_maximum_init(self):
        """Initialise the minima surrounding the opening."""
        self.left_minima, \
        self.right_minima = midpoint_minima(self.average_line_profile)
        self.maximum = midpoint_maximum(self.average_line_profile)
        assert self.left_minima.x < self.maximum.x, "Left minima to the right of the maximum."

    def opening_pts_init(self):
        """Initialise the opening points."""
        self.left_opening = peak_half_height(
            self.average_line_profile.ys, self.left_minima, self.maximum)
        self.right_opening = peak_half_height(
            self.average_line_profile.ys, self.maximum, self.right_minima)

    def opening_distance_init(self):
        """Initialise the opening distance."""
        # These points are from the profile line so we only care about the
        # x-axis.
        diff = self.left_opening.x - self.right_opening.x
        d2 = diff * diff
        d = math.sqrt(d2)
        self.opening_distance = d * self.stomate.scale_factor

    def line_to_image_space(self, line_point):
        """Return point in image space."""
        line = self.minor_axis_p2 - self.minor_axis_p1
        m = line.magnitude
        return self.minor_axis_p1 + line * (line_point.x / line.magnitude)

    def plot(self):
        """Create plot to verify that everything is sane."""
        
        # Initialise the figure.
        fig = plt.figure(figsize=(24,6))

        # Start working on the first subplot.
        ax = plt.subplot(131)
        ax.grid(False)

        flourescent_zstack = self.image_collection.zstack_array(
            s=self.stomate.series, c=0)
        projection = max_intensity_projection(flourescent_zstack)
        projection = (normalise(projection) * 255).astype(np.uint8)

        # Add the projected image.
        plt.imshow(projection, cmap=plt.cm.gray)
        ax.autoscale(False)

        # Add the title for the first subplot.
        plt.title("Stomate {} timepoint {}".format(
            self.stomate.stomate_id, self.stomate.timepoint_id))

        # Plot the ellipse.
        ellipse = Ellipse(self.box[0], self.box[1][0], self.box[1][1],
            self.box[2], fill=False, lw=1.2, color="y")
        ax.add_artist(ellipse)

        # Start working on the second subplot.
        ax = plt.subplot(132)
        ax.grid(False)

        brightfiled_zstack = self.image_collection.zstack_array(
            s=self.stomate.series, c=2)
        projection = max_intensity_projection(brightfiled_zstack)
        projection = (normalise(projection) * 255).astype(np.uint8)

        # Add the projected image.
        plt.imshow(projection, cmap=plt.cm.gray)
        ax.autoscale(False)

        # Plot the ellipse.
        ellipse = Ellipse(self.box[0], self.box[1][0], self.box[1][1],
            self.box[2], fill=False, lw=1.2, color="y")
        ax.add_artist(ellipse)

        # Plot the minor axis of the ellipse.
        plt.plot(
            [self.minor_axis_p1.x, self.minor_axis_p2.x],
            [self.minor_axis_p1.y, self.minor_axis_p2.y])

        # Plot the mid point.
        line_pt = Point2D(self.average_line_profile.mid_point, 0)
        im_pt = self.line_to_image_space(line_pt)
        plt.plot(im_pt.x, im_pt.y, marker="+", linestyle="_",
                color="m")

        # Plot the minima.
        for line_point in [self.left_minima, self.right_minima]:
            im_point = self.line_to_image_space(line_point)
            plt.plot(im_point.x, im_point.y, marker="+", linestyle="_",
                color="g")

        # Plot the maximum.
        im_pt = self.line_to_image_space(self.maximum)
        plt.plot(im_pt.x, im_pt.y, marker="+", linestyle="_",
                color="r")

        # Plot the opening points.
        for line_point in [self.left_opening, self.right_opening]:
            im_point = self.line_to_image_space(line_point)
            plt.plot(im_point.x, im_point.y, marker="+", linestyle="_",
                color="c")


        # Start the third plot.
        plt.subplot(133)

        # Plot the individual line profiles and the average line profile line.
        for line_profile in self.ignored_line_profiles:
            plt.plot(line_profile.xs, line_profile.ys, color="0.7", lw=3)
        for line_profile in self.removed_line_profiles:
            plt.plot(line_profile.xs, line_profile.ys, color="0.5")
        for line_profile in self.raw_line_profiles:
            plt.plot(line_profile.xs, line_profile.ys, color="peru")
        for line_profile in self.line_profiles:
            plt.plot(line_profile.xs, line_profile.ys, color="orange")
        plt.plot(self.average_line_profile.xs, self.average_line_profile.ys,
            lw=3, linestyle="--")

        # Plot the midpoint line.
        plt.axvline(x=self.average_line_profile.mid_point, linestyle="--",
            color="m")

        # Plot the minima.
        plt.plot([self.left_minima.x, self.right_minima.x],
                 [self.left_minima.y, self.right_minima.y],
                 marker="v", linestyle="_", color="g")

        # Plot the maximum.
        plt.plot(self.maximum.x, self.maximum.y, marker="^", linestyle="_",
            color="r")

        # Plot opening points.
        plt.plot([self.left_opening.x, self.right_opening.x],
            [self.left_opening.y, self.right_opening.y],
            marker="o", linestyle="_", color="c")

        plt.xlim((0, len(self.average_line_profile.xs)-1))
        plt.title("Stomate opening: {:.3f} microns".format(self.opening_distance))

def calculate_opening(image_collection, stomate_id, timepoint):
    """Return the stomate opening in micro meters."""
    stomate_opening = StomateOpening(image_collection, stomate_id, timepoint)
    stomate_opening.plot()

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('stomate_id', type=int, help='Zero based stomata index')
    parser.add_argument('timepoint', type=int, help='Zero based time point index')

    args = parser.parse_args()
    AutoWrite.on = False

    image_collection = unpack_data(args.confocal_file)
    calculate_opening(image_collection, args.stomate_id, args.timepoint)
    plt.show()

if __name__ == "__main__":
    main()
