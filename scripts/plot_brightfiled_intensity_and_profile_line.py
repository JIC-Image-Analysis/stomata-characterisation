"""Script for plotting bright field channel and profile line.

Basically used to make sure that the method of averaging the bright field
intensity channel is sane.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from jicimagelib.io import AutoWrite
from jicimagelib.geometry import Point2D

from util import (
    unpack_data,
    minor_and_major_lines_from_box,
    line_profile,
)

from find_stomata import find_stomata, ellipse_box

AutoWrite.on = False

#FIRST_REGION_ID, SERIES = 9, range(8, 14)   # Stomata id 1
#FIRST_REGION_ID, SERIES = 14, range(8, 14)  # Stomata id 2
#FIRST_REGION_ID, SERIES = 20, range(8, 14)  # Stomata id 3

FIRST_REGION_ID, SERIES = 8, range(15, 23)  # Stomata id 3

#FIRST_REGION_ID, SERIES = 8, range(24, 35)  # Stomata id?

def plot_brightfield_and_profile_line(im, line, lxs, lys, pxs, title, lw=1):

    # Line midpoint.
    def line_midpoint(lxs, lys):
        p1 = Point2D(lxs[0], lys[0])
        p2 = Point2D(lxs[1], lys[1])
        p3 = p1 - p2
        return p2 + p3/2.0
    lmx, lmy = line_midpoint(lxs, lys)

    # Profile midpoint.
    pmx = len(pxs) / 2.0


    # Make the figure wider.
    plt.figure(figsize=(16,6))

    # Plot the z-stack bright filed image.
    plt.subplot(1,2,1)
    plt.imshow(im, cmap=plt.get_cmap('gray'))
    plt.autoscale(False)  # Keep the image full size.
    plt.plot(lxs, lys, c='r')
    plt.plot(lmx, lmy, marker='x', c='g', markeredgewidth=1)

    # Plot the line profile.
    plt.subplot(1,2,2)
    plt.plot(pxs, line, linewidth=lw)
    plt.xlim((0, len(pxs)-1))
    plt.axvline(x=pmx, c='g', linestyle='--')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)

    # Find the region of interest from the first image.
    raw_zstack = image_collection.zstack_array(s=SERIES[0])
    stomata_region = find_stomata(raw_zstack, FIRST_REGION_ID)
    box = ellipse_box(stomata_region)

    # Work out the start and end of the profile line.
    p1, p2, p3, p4 = minor_and_major_lines_from_box(box)
    minor_xs, minor_ys = zip(p1, p2)

    # Determine the length of the intensity profile.
    im = image_collection.image()
    minor_profile, major_profile = line_profile(im, box)
    profile_xs = range(len(minor_profile))


    for i, s in enumerate(SERIES):

        num_z = 0
        profile_total = np.zeros(minor_profile.shape, dtype=float)
        im_total = np.zeros(im.shape, dtype=float)

#       if s != 34:
#           continue

        for z, proxy_im in enumerate(image_collection.zstack_proxy_iterator(s=s, c=2)):
            num_z = num_z + 1

            im = proxy_im.image
            im_total = im_total + im

            minor_profile, major_profile = line_profile(im, box, 10)
            profile_total = profile_total + minor_profile

            plot_brightfield_and_profile_line(im, minor_profile, minor_xs,
                minor_ys, profile_xs,
                title='Series {}; z-slice {}'.format(s, z))

        profile_average = profile_total / num_z
        im_average = im_total / num_z

        plot_brightfield_and_profile_line(im_average, profile_average, minor_xs,
            minor_ys, profile_xs, title='Average Series {}'.format(s), lw=4)
