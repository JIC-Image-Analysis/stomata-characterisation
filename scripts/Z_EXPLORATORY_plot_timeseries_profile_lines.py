"""Script for plotting line profile histograms of bright filed channel."""

import argparse

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from jicimagelib.io import AutoWrite

from util import unpack_data, ellipse_line_profiles
from find_stomata import find_stomata, ellipse_box
from calculate_opening import local_maxima, local_minima

AutoWrite.on = False

#FIRST_REGION_ID, SERIES = 9, range(8, 14)   # Stomata id 1
FIRST_REGION_ID, SERIES = 14, range(8, 14)  # Stomata id 2
#FIRST_REGION_ID, SERIES = 20, range(8, 14)  # Stomata id 3
#FIRST_REGION_ID, SERIES = 8, range(24, 35)  # Stomata id?

LINE_STYLES = ['_', '-', '--']
COLORS = sns.cubehelix_palette(len(SERIES), start=0, light=0.8, reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)

    # Find the region of interest from the first image.
    raw_zstack = image_collection.zstack_array(s=SERIES[0])
    stomata_region = find_stomata(raw_zstack, FIRST_REGION_ID)
    box = ellipse_box(stomata_region)

    # Find the x ticks for the line profile
    im = image_collection.image()
    minor_profile, major_profile = ellipse_line_profiles(im, box)
    xs = range(len(minor_profile))
    pmx = len(xs) / 2.0  # Profile midpoint.

    for i, s in enumerate(SERIES):
        total = np.zeros(minor_profile.shape, dtype=float)
        tot_z = 0
        for proxy_im in image_collection.zstack_proxy_iterator(s=s, c=2):
            tot_z = tot_z + 1
            im = proxy_im.image
            minor_profile, major_profile = ellipse_line_profiles(im, box, 10)
    #       plt.plot(xs, minor_profile)
            total = total + minor_profile
        average = total / tot_z
        linestyle = LINE_STYLES[(s+1) % len(LINE_STYLES)]
        plt.plot(xs, average, linewidth=4,
            color=COLORS[i],  label="Series {}".format(s))

        # Plot the local maxima.
        local_maxima_xs, = np.where( local_maxima(average) )
        local_maxima_ys = np.take(average, local_maxima_xs)
        for maxima in local_maxima_xs:
            plt.plot(local_maxima_xs, local_maxima_ys, c=COLORS[i], marker="*",
                markersize=10, linestyle='_')

        # Plot the local minima.
        local_minima_xs, = np.where( local_minima(average) )
        local_minima_ys = np.take(average, local_minima_xs)
        for minima in local_minima_xs:
            plt.plot(local_minima_xs, local_minima_ys, c=COLORS[i], marker="o",
                markersize=10, linestyle='_')

    #   plt.title('Series {}'.format(s))

    plt.xlim((0, len(xs)-1))
    plt.axvline(x=pmx, c='g', linestyle='--')
    plt.legend()
    plt.show()
