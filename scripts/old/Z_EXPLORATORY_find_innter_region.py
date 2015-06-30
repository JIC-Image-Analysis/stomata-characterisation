"""Script to calculate a line profiles across stomata."""

import argparse
import math

import numpy as np
import cv2
import scipy
import skimage
import skimage.transform
import skimage.measure
import skimage.morphology

from jicimagelib.geometry import Point2D
from jicimagelib.transform import transformation

from jicimagelib.region import Region

from util import (
    unpack_data,
    minor_and_major_lines_from_box,
    ellipse_line_profiles,
)

from find_stomata import (
    find_stomata,
    ellipse_box,
    smoothed_max_intensity_projection,
    threshold_otsu,
)


#############################################################################
# Working with the lines from the stomata ellipse fitting
# to find the inner region.
#############################################################################

def annotated_region_image(region):
    """Return an annotated region image to plot on top of."""
    xdim, ydim = region.bitmap.shape
    annotated_array = np.zeros((xdim, ydim, 3), dtype=np.uint8)
    annotated_array[region.border.index_arrays] = 255, 255, 255
    return annotated_array

def save_major_and_minor_lines(annotated_array, box, name):
    """Save image with minor and major lines."""
    p1, p2, p3, p4 = minor_and_major_lines_from_box(box)

    cv2.ellipse(annotated_array, box, (0, 0, 255))
    cv2.line(annotated_array, p1.astype("int").astuple(), p2.astype("int").astuple(), (0, 255, 0), 1)
    cv2.line(annotated_array, p3.astype("int").astuple(), p4.astype("int").astuple(), (255, 0, 0), 1)
    scipy.misc.imsave(name, annotated_array)
    
def find_relative_profile_length(profile):
    """Return relative cut points from a line profile."""
    new_profile = profile.copy()
    new_profile.shape = len(profile), 1
    sobel_filter = scipy.ndimage.filters.sobel(new_profile, axis=0)
    scipy.misc.imsave('sobel_line.png', sobel_filter)
    sobel_filter.shape = len(profile),
    sobel_min = np.min(sobel_filter)
    extra = abs(sobel_min)

    negative_sobel_cut_index = None
    positive_sobel_cut_index = None
    sobel_cutoff = 20
    for i, p in enumerate(sobel_filter):
        if p < -sobel_cutoff:
            negative_sobel_cut_index = i
        if p > sobel_cutoff:
            positive_sobel_cut_index = i
            break
#   # Visual validation.
#   for i, (p, f) in enumerate(zip(profile, sobel_filter)):
#       s = "X"*p
#       if i in [negative_sobel_cut_index, positive_sobel_cut_index]:
#           print '{:3.0f} {} ***'.format(f, s)
#       else:
#           print '{:3.0f} {}'.format(f, s)

    d = float(len(sobel_filter))
    rel1 = negative_sobel_cut_index/d
    rel2 = positive_sobel_cut_index/d
    relative_length = (rel1 + rel2) / 2.0
    return relative_length


def find_inner_region_using_lines(raw_zstack, stomata_region, out_fn):
    """Given an raw z-stack, identify the inner region of a stomata."""
    # We know that region 8 is a stomata.
    projection = smoothed_max_intensity_projection(raw_zstack)
    annotated_array = annotated_region_image(stomata_region)

    box = ellipse_box(stomata_region)
    center, bounds, angle = box
    width, height = bounds

#   save_major_and_minor_lines(annotated_array, box, 'quadrant_lines_image.png')
    minor_profile, major_profile = ellipse_line_profiles(projection, box)
    minor_rel_length = find_relative_profile_length(minor_profile)
    major_rel_length = find_relative_profile_length(major_profile)

    # Clean the annotated array.
    inner_bounds = width * minor_rel_length, height * major_rel_length
    inner_box = center, inner_bounds, angle

    save_major_and_minor_lines(annotated_array, inner_box, out_fn)

    return inner_box


#############################################################################
# Using edge detection on the stomata to find the inner region.
#############################################################################

@transformation
def cutout_region(image, region):
    """Cut out a region from an image."""
    return image * region.bitmap

@transformation
def find_edges(image, mask=None):
    """Find edges using sobel filter."""
    edges = skimage.filters.sobel(image, mask)
    return skimage.img_as_ubyte(edges)

@transformation
def remove_small_objects(image, min_size=50):
    binary_im = image > 0
    binary_im =  skimage.morphology.remove_small_objects(binary_im, min_size)
    return image*binary_im

@transformation
def dilation(image, salem=None):
    binary_im = image > 0
    binary_im =  skimage.morphology.binary_dilation(binary_im, salem)
    binary_im = binary_im.astype(np.uint8)
    return binary_im*255
    
@transformation
def fill_small_holes(image, max_size=50):
    binary_im = image > 0
    binary_im = binary_im == False  # Invert.
    binary_im =  skimage.morphology.remove_small_objects(binary_im, max_size)
    binary_im = binary_im == False  # And back to positive.
    binary_im = binary_im.astype(np.uint8)
    return binary_im*255

@transformation
def skeletonize(image):
    binary_im = image > 0
    binary_im = skimage.morphology.skeletonize(binary_im)
    binary_im = binary_im.astype(np.uint8)
    return binary_im*255
    
@transformation
def invert(image):
    tmp_max = np.max(image)
    tmp_im = np.ones(tmp_max.shape, dtype=image.dtype)
    return tmp_im * tmp_max - image
    
@transformation
def find_connected_components(image, connectivity=None, background=None):
    connected_components = skimage.measure.label(image,
        connectivity=connectivity, return_num=False)
    return connected_components.astype(np.uint8)


def find_inner_region_using_edge_detection(raw_zstack, stomata_region, out_fn):
    """Given an raw z-stack, identify the inner region of a stomata."""
    projection = smoothed_max_intensity_projection(raw_zstack)
    stomata_projection = cutout_region(projection, stomata_region)
    edges = find_edges(stomata_projection, stomata_region.bitmap)
    otsu = threshold_otsu(edges)
    no_small = remove_small_objects(otsu)
    dilated = dilation(no_small,
        salem=skimage.morphology.disk(2))
    filled = fill_small_holes(dilated, 100)
    skeleton = skeletonize(filled)
    thick_skeleton = dilation(skeleton)
    components = invert(thick_skeleton)
    connected_components = find_connected_components(components)

    stomata_box = ellipse_box(stomata_region)  # center, bounds, angle
    y, x = stomata_box[0]  # Transpose opencv point.
    center = Point2D(x, y).astype('int')
    inner_region_id = connected_components[center.x][center.y]

    inner_region = Region.select_from_array(connected_components, inner_region_id)
    inner_box = ellipse_box(inner_region)

    xdim, ydim = inner_region.bitmap.shape
    annotated_array = np.dstack((projection, projection, projection))
    cv2.ellipse(annotated_array, inner_box, (0, 255, 0), 2)
    annotated_array[inner_region.border.index_arrays] = 255 , 0, 0
    cv2.ellipse(annotated_array, stomata_box, (0, 255, 0), 2)
    annotated_array[stomata_region.border.index_arrays] = 255 , 0, 0

    scipy.misc.imsave(out_fn, annotated_array)


def find_inner_region(raw_zstack, stomata_region, out_fn):
    """Given an image collection, identify the inner region of a stomata."""
    find_inner_region_using_lines(raw_zstack, stomata_region, out_fn)
#   find_inner_region_using_edge_detection(raw_zstack, stomata_region, out_fn)
    



    

def main():

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('confocal_file', help='File containing confocal data')
    parser.add_argument('series', type=int, help='Zero based series index')
    parser.add_argument('region', type=int, help='Zero based region index')

    args = parser.parse_args()

    image_collection = unpack_data(args.confocal_file)
    raw_zstack = image_collection.zstack_array(s=args.series)
    # We know that region 8 is a stomata.
    stomata_region = find_stomata(raw_zstack, region_id=args.region)
    find_inner_region(raw_zstack, stomata_region, out_fn='annotated_inner_region.png')


if __name__ == "__main__":
    main()
