import SimpleITK as sitk
import numpy as np
import math
from numba import njit


@njit
def get_min(coords):
    return int(coords.min())


@njit
def get_max(coords):
    return int(coords.max())


def crop_mask_to_roi(mask_as_img: sitk.Image, xy_scaling_factor):
    # find indexes in all axis
    mask_as_array = sitk.GetArrayFromImage(mask_as_img)
    zs, ys, xs = np.where(mask_as_array != 0)

    # extract cube with extreme limits of where are the values != 0
    bounding_box = [[get_min(xs), get_max(xs) + 1], [get_min(ys), get_max(ys) + 1], [get_min(zs), get_max(zs) + 1]]
    meta = {
        "original_direction": list([int(i) for i in mask_as_img.GetDirection()]),
        "original_size": list([int(i) for i in mask_as_img.GetSize()]),
        "original_origin": list([float(i) for i in mask_as_img.GetOrigin()]),
        "original_spacing": list([float(i) for i in mask_as_img.GetSpacing()]),
        "bounding_box": bounding_box,
        "bounding_box_ct_grid": list([[math.floor(min_max[0] / xy_scaling_factor),
                                       math.ceil(min_max[1] / xy_scaling_factor)]
                                      for min_max in bounding_box])
    }

    cropped_img = mask_as_img[bounding_box[0][0]: bounding_box[0][1],
                  bounding_box[1][0]: bounding_box[1][1],
                  bounding_box[2][0]: bounding_box[2][1]]
    return cropped_img, meta
