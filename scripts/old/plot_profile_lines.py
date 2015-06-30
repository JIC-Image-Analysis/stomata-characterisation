"""Plot the profile lines for a stomata at a particular time point."""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from util import (
    unpack_data,
    ellipse_line_profiles,
    series_identifier,
)

from generate_augmented_zstack_images import box_of_interest

def plot_profile_lines(image_collection, stomata_id, timepoint):
    """Plot the individual z-stacks and the mean bright field profile lines."""
    series_id = series_identifier(stomata_id, timepoint)
    box = box_of_interest(image_collection, stomata_id)

    im = image_collection.image()
    minor_profile, major_profile = ellipse_line_profiles(im, box)

    xs = range(len(minor_profile))
    pmx = len(xs) / 2.0  # Profile midpoint.

    total = np.zeros(minor_profile.shape, dtype=float)
    tot_z = 0
    
    z_iter = image_collection.zstack_proxy_iterator(s=series_id, c=2)
    for z, proxy_im in enumerate(z_iter):
        im = proxy_im.image
        minor_profile, major_profile = ellipse_line_profiles(im, box, 10)
        plt.plot(xs, minor_profile)
        total = total + minor_profile
        tot_z += 1

    average = total / tot_z

    plt.plot(xs, average, linewidth=4)

    # Plot the midpoint
    plt.axvline(x=pmx, c='0.3', linestyle='--')

    plt.xlim((0, len(xs)-1))

    plt.title('Stomate {} timepoint {}'.format(stomata_id, timepoint))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('stomata_id', type=int, help='Zero based stomata index')
    parser.add_argument('timepoint', type=int, default=None,
        help='Zero based time point index')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    plot_profile_lines(image_collection, args.stomata_id, args.timepoint)
