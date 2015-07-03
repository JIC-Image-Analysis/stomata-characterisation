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
)

from util.geometry import line
from util.array import (
    peak_half_height,
    rmsd,
    midpoint_minima,
    midpoint_maximum,
)

from stomata_finder import find_stomate_ellipse_box


class StomateOpening(object):
    """Class for calculating the stomate opening."""
    
    def __init__(self, image_collection, stomate_id, timepoint):
        self.image_collection = image_collection
        self.stomata_timeseries = stomata_timeseries_lookup(stomate_id)
        self.stomate = self.stomata_timeseries.stomate_at_timepoint(timepoint)
        
        self.ellipse_box_init()
        self.minor_axis_pts_init()
        self.line_profiles_init()
        self.exclude_top_and_bottom_zslices_init()
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

        zstack = [z for z in self.image_collection.zstack_proxy_iterator(
            self.stomate.series, c=2)]

        for i, zslice in enumerate(zstack):
            # Convert from cv2 points to scikit image points.
            ski_p1 = self.minor_axis_p1[1], self.minor_axis_p1[0]
            ski_p2 = self.minor_axis_p2[1], self.minor_axis_p2[0]

            profile = skimage.measure.profile_line( zslice.image, ski_p1,
                ski_p2, linewidth=10)
            line_profile = LineProfile(profile, i)

            self.line_profiles.add_line_profile(line_profile)


        self.average_line_profile_calc()
        self.minima_maximum_calc()

    def average_line_profile_calc(self):
        """Calculate the average line profile."""
        self.average_line_profile = self.line_profiles.average('normalised')

    def exclude_top_and_bottom_zslices_init(self):
        """Exclude top and bottom zslices."""
        fraction_to_exclude = 0.4
        exclude_int = int(round(len(self.line_profiles) * fraction_to_exclude / 2.))
        to_include = set(range(exclude_int, len(self.line_profiles)-exclude_int))
        to_exclude = set(range(len(self.line_profiles))) - to_include

        while len(to_exclude) > 0:
            for line_profile in self.line_profiles:
                if line_profile.identifier not in to_include:
                    line_profile.include = False
                    to_exclude.remove(line_profile.identifier)

        self.average_line_profile_calc()
        self.minima_maximum_calc()

    def minima_maximum_calc(self):
        """Calculate the minima surrounding the opening and the maxium within."""
        self.left_minima, \
        self.right_minima = midpoint_minima(
            self.average_line_profile.smoothed_gaussian.ys,
            self.average_line_profile.mid_point)
        self.maximum = midpoint_maximum(
            self.average_line_profile.smoothed_gaussian.ys,
            self.average_line_profile.mid_point)
        assert self.left_minima.x < self.maximum.x, "Left minima to the right of the maximum."

    def opening_pts_init(self):
        """Initialise the opening points."""
        self.left_opening = peak_half_height(
            self.average_line_profile.smoothed_gaussian.ys,
            self.left_minima, self.maximum)
        self.right_opening = peak_half_height(
            self.average_line_profile.smoothed_gaussian.ys,
            self.maximum, self.right_minima)

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

    def zoom_image(self, zslice, included):
        """Return a 100x100 image zoomed in on the stomate."""
        def draw_line(im, pt1, pt2, rgb_color):
            rr, cc = skimage.draw.line(
                int(round(pt1.y)),
                int(round(pt1.x)),
                int(round(pt2.y)),
                int(round(pt2.x))
            )
            im[rr, cc] = rgb_color
            return im
            
        # Create rgb image from zslice.
        zslice = self.image_collection.image(
            s=self.stomate.series,
            c=2,
            z=zslice)
        rgb_im = np.dstack([zslice, zslice, zslice])

        # Draw red opening line on rgb image.
        pt1 = self.line_to_image_space(self.left_opening)
        pt2 = self.line_to_image_space(self.right_opening)
        rgb_im = draw_line(rgb_im, pt1, pt2, (0, 255, 255))

        # Get selection of interest.
        x, y = self.box[0]
        offset = 50
        selection = rgb_im[
            y-offset:y+offset,
            x-offset:x+offset
            ]

        # Draw red outlines around excluded boxes.
        if not included:
            ydim, xdim, _ = selection.shape
            xend = xdim - 1
            yend = ydim - 1
            tlc = Point2D(0, 0)
            trc = Point2D(xend, 0)
            blc = Point2D(0, yend)
            brc = Point2D(xend, yend)
            red = (255, 0, 0)

            selection = draw_line(selection, tlc, trc, red)
            selection = draw_line(selection, tlc, blc, red)
            selection = draw_line(selection, blc, brc, red)
            selection = draw_line(selection, trc, brc, red)


        return selection

    def zoom_collage(self):
        """Return a composite image of all the selected zslices."""
        ncols = 13.
        rows = []
        i = 0
        for r in range(int(math.ceil(len(self.line_profiles) / ncols))):
            cols = []
            for c in range(int(ncols)):
                if i < len(self.line_profiles):
                    lp = self.line_profiles[i]
                    im = self.zoom_image(lp.identifier, lp.include)
                    cols.append(im)
                else:
                    # Add an empty image of the same shape as the first in the
                    # row.
                    cols.append( np.zeros(cols[0].shape, dtype=np.uint8) )
                i += 1
            rows.append( np.concatenate(cols, axis=1) )
        return np.concatenate(rows)

    def add_stomate_ellipse_to_plot(self, ax):
        """Add the stomate ellipse to the plot."""
        ellipse = Ellipse(self.box[0], self.box[1][0], self.box[1][1],
            self.box[2], fill=False, lw=1.2, color="y")
        ax.add_artist(ellipse)
        
    def flourescent_plot(self, ax):
        """Plot the fluorescent channel."""
        flourescent_zstack = self.image_collection.zstack_array(
            s=self.stomate.series, c=0)
        projection = max_intensity_projection(flourescent_zstack)
        projection = (normalise(projection) * 255).astype(np.uint8)

        plt.imshow(projection, interpolation="none", cmap=plt.cm.gray)

        self.add_stomate_ellipse_to_plot(ax)

        plt.title("Stomate {} timepoint {}".format(
            self.stomate.stomate_id, self.stomate.timepoint_id))

        ax.autoscale(False)
        ax.grid(False)

    def add_points_to_plot(self, points, color, marker="+", linestyle="_"):
        """Add points to plot."""
        for p in points:
            # Convert from line profile space to image space.
            im_point = self.line_to_image_space(p)
            plt.plot(im_point.x, im_point.y, marker=marker,
                linestyle=linestyle, color=color)


    def brightfield_plot(self, ax):
        """Plot the bright field channel."""
        brightfiled_zstack = self.image_collection.zstack_array(
            s=self.stomate.series, c=2)
        projection = max_intensity_projection(brightfiled_zstack)
        projection = (normalise(projection) * 255).astype(np.uint8)

        plt.imshow(projection, interpolation="none", cmap=plt.cm.gray)
         
        ax.autoscale(False)
        ax.grid(False)

        self.add_stomate_ellipse_to_plot(ax)

        # Plot the minor axis of the ellipse.
        plt.plot(
            [self.minor_axis_p1.x, self.minor_axis_p2.x],
            [self.minor_axis_p1.y, self.minor_axis_p2.y])

        # Plot the mid point.
        line_pt = Point2D(self.average_line_profile.mid_point, 0)
        self.add_points_to_plot([line_pt], "m")

        # Plot the minima, maximum and opening points.
        self.add_points_to_plot([self.left_minima, self.right_minima], "g")
        self.add_points_to_plot([self.maximum], "r")
        self.add_points_to_plot([self.left_opening, self.right_opening], "c")

    def profile_lines_plot(self, ax):
        """Plot the intensity profile lines."""
        # Plot the individual line profiles and the average line profile line.
        for line_profile in self.line_profiles:
            if not line_profile.include:
                plt.plot(line_profile.xs, line_profile.normalised.ys, color="r")
        for line_profile in self.line_profiles:
            if line_profile.include:
                plt.plot(line_profile.xs, line_profile.normalised.ys,
                    color="0.7")
        plt.plot(self.average_line_profile.xs, self.average_line_profile.ys,
            lw=3, color="0.3")
        plt.plot(self.average_line_profile.xs,
            self.average_line_profile.smoothed_gaussian.ys,
            lw=3, color="b")

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

    def stomate_zstack_closeup_plot(self, ax):
        """Plot the stomate zstack close up."""
        im = self.zoom_collage()
        ax.grid(False)
        plt.imshow(im, interpolation="none", cmap=plt.cm.gray)
        ax.autoscale(False)
        plt.title("Z-slices: {}".format(
            [lp.identifier for lp in self.line_profiles]))

    def plot(self):
        """Create plot to verify that everything is sane."""
        
        # Initialise the figure.
        fig = plt.figure(figsize=(20,12))

        # First subplot.
        ax = plt.subplot(231)
        self.flourescent_plot(ax)

        # Second subplot.
        ax = plt.subplot(232)
        self.brightfield_plot(ax)


        # Third splot.
        ax = plt.subplot(233)
        self.profile_lines_plot(ax)

        # Fourth subplot.
        ax = plt.subplot2grid((2,3), (1, 0), colspan=3)
        self.stomate_zstack_closeup_plot(ax)


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
