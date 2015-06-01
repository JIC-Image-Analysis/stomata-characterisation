"""Characterise the openings in stomata."""

import argparse

import scipy
import skimage.transform

from util import unpack_data
from find_stomata import (
    find_candidate_regions,
    ellipse_box,
    gaussian_filter,
    max_intensity_projection,
)

def rotate(image, angle, center):
    """Return rotated image."""
    return skimage.transform.rotate(image, angle, center=center, resize=False)

def crop(image, lower_left_coord, upper_right_coord):
    """Return the cropped image."""
    x1, y1 = lower_left_coord
    x2, y2 = upper_right_coord
    return image[y1:y2, x1:x2]

def crop_coordinates(bounds, angle, center):
    """Return the lower left and upper right coordinates after a rotation."""
    cx, cy = center
    width, height = bounds
    ll_coord = int(cx - width/2), int(cy - height/2)
    ur_coord = int(cx + width/2 + 2), int(cy + height/2 + 2)
    return ll_coord, ur_coord

def align_and_crop_box(image, box):
    """Return the aligned cropped box from the image."""

    center, bounds, angle = box
    rotated_array = rotate(image, angle, center)
    ll_coord, ur_coord = crop_coordinates(bounds, angle, center)
    return crop(rotated_array, ll_coord, ur_coord)

def find_inner_region(confocal_file):
    """Given the confocal image file, find stomata in it."""

    image_collection = unpack_data(confocal_file)
    raw_z_stack = image_collection.zstack_array(s=30)
    candidate_regions = find_candidate_regions(raw_z_stack)

    # We know that region 8 is a stomata.
    stomata_region = candidate_regions[8].convex_hull

    # Image to be aligned and cropped.
    smoothed_z_stack = gaussian_filter(raw_z_stack, (3, 3, 1))
    projection = max_intensity_projection(smoothed_z_stack)

    box = ellipse_box(stomata_region)
    cropped_stomata = align_and_crop_box(projection, box)
    scipy.misc.imsave('aligned_and_cropped_stomata.png', cropped_stomata)

    

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')

    args = parser.parse_args()

    unpack_data(args.confocal_file)
    find_inner_region(args.confocal_file)

if __name__ == "__main__":
    main()
