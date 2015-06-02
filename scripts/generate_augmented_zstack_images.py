"""Find the stomata opening for a time point and plot annotated z-slices."""

import sys
import os
import os.path
import argparse

import numpy as np
import scipy.misc
import cv2

from jicimagelib.geometry import Point2D
from jicimagelib.io import AutoWrite
AutoWrite.on = False

from util import (
    unpack_data,
    stomata_lookup,
    minor_and_major_lines_from_box,
)
from calculate_opening import ellipse_of_interest, opening_points

def series_identifier(stomata_id, timepoint):
    """Return the series identifier."""
    region, series_identifiers = stomata_lookup(stomata_id)
    for i, series in enumerate(series_identifiers):
        if timepoint == i:
            return series
    raise(IndexError("No such timepoint: {}".format(timepoint)))

def box_of_interest(image_collection, stomata_id):
    """Return the stomata box of interest."""
    region, series_identifiers = stomata_lookup(stomata_id)
    first = series_identifiers[0]
    box = ellipse_of_interest(image_collection, first, region)
    return box

def minor_line(box):
    """Return the two points representing the minor line from the ellipse box."""
    p1, p2, p3, p4 = minor_and_major_lines_from_box(box)
    return p1, p2

def opening_line(image_collection, series_id, box):
    """Return the two point representing the stomata opening line."""
    minor_pt1, minor_pt2 = minor_line(box)
    cut1, cut2 = opening_points(image_collection, series_id, box)
    line = minor_pt2 - minor_pt1
    m = line.magnitude
    inner_pt1 = minor_pt1 + line * (cut1.x / line.magnitude)
    inner_pt2 = minor_pt1 + line * (cut2.x / line.magnitude)
    return inner_pt1, inner_pt2

def generate_augmented_zstack_images(image_collection, stomata_id, timepoint, output_dir):
    """Find the stomata opening write out image."""
    series_id = series_identifier(stomata_id, timepoint)
    box = box_of_interest(image_collection, stomata_id)
    minor_pt1, minor_pt2 = minor_line(box)
    inner_pt1, inner_pt2 = opening_line(image_collection, series_id, box)



# Loop over z-stacks and plot images...
    z_iter = image_collection.zstack_proxy_iterator(s=series_id, c=2)
    for z, proxy_im in enumerate(z_iter):
        fname = 'S{}_T{}_Z{}.png'.format(series_id, timepoint, z)
        fpath = os.path.join(output_dir, fname)
        im = np.dstack((proxy_im.image, proxy_im.image, proxy_im.image))
        cv2.ellipse(im, box, (0, 0, 255))
        cv2.line(im,
            minor_pt1.astype("int").astuple(),
            minor_pt2.astype("int").astuple(),
            (0, 255, 0), 1)
        cv2.line(im,
            inner_pt1.astype("int").astuple(),
            inner_pt2.astype("int").astuple(), (255, 0, 0), 1)
        scipy.misc.imsave(fpath, im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('stomata_id', type=int, help='Zero based stomata index')
    parser.add_argument('timepoint', type=int, default=None,
        help='Zero based time point index')
    parser.add_argument('-d', '--output_dir', default='.',
        help='Output directory')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    image_collection = unpack_data(args.confocal_file)
    generate_augmented_zstack_images(image_collection, args.stomata_id,
        args.timepoint, args.output_dir)
