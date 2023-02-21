import logging
import traceback

import SimpleITK as sitk
import numpy as np
from numba import njit
from skimage import draw

from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException


def scale_information_tuple(information_tuple: tuple, xy_scaling_factor: int, out_type: type, up: bool = True):
    scale_array = np.array([xy_scaling_factor, xy_scaling_factor, 1])
    if up:
        information_tuple = np.array(information_tuple) * scale_array
    else:
        information_tuple = np.array(information_tuple) / scale_array

    return tuple([out_type(info) for info in information_tuple])


def _get_transform_matrix(spacing, direction):
    """
    this returns the basics needed to run _transform_physical_point_to_continuous_index
    """
    s = np.array(spacing)
    d = np.array(direction).reshape(3, 3)
    m_IndexToPhysicalPoint = np.multiply(d, s)
    m_PhysicalPointToIndex = np.linalg.inv(m_IndexToPhysicalPoint)

    return m_PhysicalPointToIndex


def xor_update_np_mask(np_mask, filled_poly, z):
    overlay = np.logical_xor(np_mask[z, :, :], filled_poly)
    np_mask[z, :, :] = overlay


@njit
def _transform_physical_point_to_continuous_index(coords, m_PhysicalPointToIndex, origin):
    """
    This method does the same as SimpleITK's TransformPhysicalPointToContinuousIndex, but in a vectorized fashion.
    The implementation is based on ITK's code found in https://itk.org/Doxygen/html/itkImageBase_8h_source.html#l00497 and
    https://discourse.itk.org/t/solved-transformindextophysicalpoint-manually/1031/2
    """

    if m_PhysicalPointToIndex is None:
        raise Exception("Run set transform variables first!")

    pts = np.empty_like(coords)
    pts[:, 0] = coords[:, 0]  # Index of contour
    pts[:, 1] = coords[:, 1] - origin[0]  # x
    pts[:, 2] = coords[:, 2] - origin[1]  # y
    pts[:, 3] = coords[:, 3] - origin[2]  # z

    pts[:, 1:] = pts[:, 1:].copy() @ m_PhysicalPointToIndex

    return pts


def get_cropped_origin(stacked_coords):
    float_min = np.min(stacked_coords[:, 1:], axis=0)
    return float_min


def stack_coords(rtstruct_contours):
    coords = None
    for i, contour in enumerate(rtstruct_contours):
        if contour['type'].upper() not in ['CLOSED_PLANAR', 'INTERPOLATED_PLANAR']:
            if 'name' in contour:
                logging.info(f'Skipping contour {contour["name"]}, unsupported type: {contour["type"]}')
            else:
                logging.info(f'Skipping unnamed contour, unsupported type: {contour["type"]}')
            continue

        # Stack coordinate components to one array
        temp_coords = contour['points']
        stack = np.column_stack((
            [i for u in range(len(temp_coords["x"]))],
            temp_coords["x"],
            temp_coords["y"],
            temp_coords["z"])
        )
        # Stack column 0 is index of contour, then x, y, z.
        if coords is None:
            coords = stack
        else:
            coords = np.concatenate([coords, stack])

    return coords


def get_shape(idx_pts):
    maxs = np.ceil(np.max(idx_pts[:, 1:], axis=0)).astype(int) + 1

    return maxs


class DcmPatientCoords2Mask:
    def __init__(self):
        self.m_PhysicalPointToIndex = None
        self.origin = None
        self.shape = None
        self.spacing = None

    def convert(self,
                rtstruct_contours,
                dicom_image,
                xy_scaling_factor,
                crop_mask):
        self.spacing = scale_information_tuple(information_tuple=dicom_image.GetSpacing(),
                                               xy_scaling_factor=xy_scaling_factor,
                                               up=False,
                                               out_type=float)
        self.m_PhysicalPointToIndex = _get_transform_matrix(spacing=self.spacing,
                                                            direction=dicom_image.GetDirection())

        # Arrange contours into an array shape (n, 4), where column order is contour_index, x, y, z
        stacked_coords = stack_coords(rtstruct_contours=rtstruct_contours)

        # Origin set to minimum of x, y and z
        self.origin = get_cropped_origin(stacked_coords)

        # Index of coords
        idx_pts = _transform_physical_point_to_continuous_index(stacked_coords,
                                                                m_PhysicalPointToIndex=self.m_PhysicalPointToIndex,
                                                                origin=self.origin)
        # Get Shape for rastering
        self.shape = get_shape(idx_pts)

        np_mask = np.zeros(list(reversed(self.shape)), dtype=np.uint8)
        for idx in np.unique(idx_pts[:, 0]):
            pts = idx_pts[idx_pts[:, 0] == idx][:, 1:]  # Slice to only get coordinates
            z = int(pts[0, 2])  # Get z of the index

            try:
                # Draw the polygon and xor update np_mask
                filled_poly = draw.polygon2mask((self.shape[1], self.shape[0]), pts[:, 1::-1])
                xor_update_np_mask(np_mask=np_mask, filled_poly=filled_poly, z=z)
            except IndexError:
                # if this is triggered the contour is out of bounds
                raise ContourOutOfBoundsException()
            except RuntimeError as e:
                # this error is sometimes thrown by SimpleITK if the index goes out of bounds
                if 'index out of bounds' in str(e):
                    raise ContourOutOfBoundsException()
                raise e  # something serious is going on

        # np_mask to image
        mask = sitk.GetImageFromArray(np_mask.astype(np.uint8))  # Had trouble with the type. Use np.uint8!

        # Set image meta
        mask.SetDirection(dicom_image.GetDirection())
        mask.SetOrigin(self.origin)
        mask.SetSpacing(self.spacing)

        # If not crop, then resample to align with the original dicom image.
        if not crop_mask:
            mask = sitk.Resample(mask,
                                 scale_information_tuple(dicom_image.GetSize(),
                                                         xy_scaling_factor=xy_scaling_factor,
                                                         out_type=int,
                                                         up=True),
                                 sitk.Transform(),
                                 sitk.sitkNearestNeighbor,
                                 dicom_image.GetOrigin(),
                                 self.spacing,
                                 dicom_image.GetDirection())
        return mask
