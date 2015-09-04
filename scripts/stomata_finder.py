"""Fit ellipse to stomate identified using x,y coordinates."""

import os
import os.path
import argparse

import numpy as np
import scipy.misc
import cv2

import skimage.morphology

from jicimagelib.io import AutoName, AutoWrite
from jicimagelib.region import Region
from jicimagelib.util.array import normalise
from jicimagelib.transform import (
    transformation,
    max_intensity_projection,
    equalize_adaptive_clahe,
    smooth_gaussian,
    threshold_otsu,
    remove_small_objects,
)

from util import (
    unpack_data,
    stomata_timeseries_lookup,
    STOMATA,
)

from util.transform import (
    find_connected_components,
)

from util.geometry import (
    ellipse_box,
)

@transformation
def boolean_invert(image):
    """Return the negative image."""
    return image == False


@transformation
def single_stomate(components, x, y):
    """Return binary image of a single stomate from a point within it."""
    stomate = np.zeros(components.shape, dtype=np.bool)
    identifier = components[y,x]
    coords = np.where( components == identifier )
    stomate[coords] = True
    return stomate

@transformation
def convex_hull(image):
    """Return the convex hull as a binary image."""
    return skimage.morphology.convex_hull_image(image)

@transformation
def grow(convex_hull_im, small_removed_im):
    """Grow convex_hull_im to fill hole in small_removed_im."""
    target_im = np.logical_and(
        small_removed_im,
        np.logical_not(convex_hull_im))

    # Grow to fill the target hole.
    while True:
        prev_convex_hull_im = convex_hull_im
        convex_hull_im = skimage.morphology.binary_dilation(convex_hull_im)
        circonference = np.sum( np.logical_and(
            convex_hull_im,
            np.logical_not(prev_convex_hull_im)) * 1)
        overlap = np.sum( np.logical_and(convex_hull_im, target_im) * 1)
        if overlap > 2 * circonference:
            break

    # Remove pixels that overlap into the target hole.
    trimmed = np.logical_and(
        convex_hull_im,
        np.logical_not(target_im))

    return trimmed

def find_stomate_ellipse_box(raw_zstack, x, y):
    """Return stomate ellipse box."""
    projected = max_intensity_projection(raw_zstack)
    equalised = equalize_adaptive_clahe(projected)
    smoothed = smooth_gaussian(equalised)
    thresholded = threshold_otsu(smoothed)
    holes_filled = remove_small_objects(thresholded, min_size=100)
    inverted = boolean_invert(holes_filled)
    small_removed = remove_small_objects(inverted, min_size=100)
    components = find_connected_components(small_removed, background=None)
    stomate = single_stomate(components, x, y)
    hull = convex_hull(stomate)
    grown = grow(hull, small_removed)

    stomata_region = Region(grown)
    box = ellipse_box(stomata_region)

    return box

def annotate_with_ellipse_box(image, box):
    """Write out image annotated with ellipse box."""
    fname = 'annotated_projection.png'
    fpath = os.path.join(AutoName.directory, fname)
    gray_uint8 = normalise(image) * 255
    annotation_array = np.dstack([gray_uint8, gray_uint8, gray_uint8])
    cv2.ellipse(annotation_array, box, (255, 0, 0))
    scipy.misc.imsave(fpath, annotation_array)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("confocal_file", help="File containing confocal data")
    parser.add_argument('series', type=int, help='Zero based microscopy series index')
    parser.add_argument('x', type=int, help='x coordinate')
    parser.add_argument('y', type=int, help='y coordinate')
    parser.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    image_collection = unpack_data(args.confocal_file)
    raw_zstack = image_collection.zstack_array(s=args.series, c=0)

    box = find_stomate_ellipse_box(raw_zstack, args.x, args.y)

    projected = max_intensity_projection(raw_zstack)
    annotate_with_ellipse_box(projected, box)

def test_all():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("confocal_file", help="File containing confocal data")
    parser.add_argument("output_dir", help="Output directory")

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir
    AutoWrite.on = False

    image_collection = unpack_data(args.confocal_file)

    for i in range(len(STOMATA)):
        stomata_timeseries = stomata_timeseries_lookup(i)
        for stomate in stomata_timeseries:
            fname = 'annotated_projection_stomate_{}_series_{}.png'.format(
                stomate.stomate_id, stomate.timepoint_id)
            fpath = os.path.join(AutoName.directory, fname)
            raw_zstack = image_collection.zstack_array(s=stomate.series, c=0)
            projected = max_intensity_projection(raw_zstack)
            gray_uint8 = normalise(projected) * 255
            annotation_array = np.dstack([gray_uint8, gray_uint8, gray_uint8])
            box = find_stomate_ellipse_box(raw_zstack, stomate.x, stomate.y)
            cv2.ellipse(annotation_array, box, (255, 0, 0))
            scipy.misc.imsave(fpath, annotation_array)


if __name__ == "__main__":
    main()

