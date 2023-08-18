import logging
import os.path

from dcmrtstruct2nii.adapters.convert.filenameconverter import FilenameConverter
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import PathDoesNotExistException, ContourOutOfBoundsException


def list_rt_structs(rtstruct_file):
    """
    Lists the structures in an DICOM RT Struct file by name.

    :param rtstruct_file: Path to the rtstruct file
    :return: A list of names, if any structures are found
    """
    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    rtreader = RtStructInputAdapter()
    rtstructs = rtreader.ingest(rtstruct_file, True)
    return [struct['name'] for struct in rtstructs]


def dcmrtstruct2nii(rtstruct_file,
                    dicom_file,
                    output_path,
                    structures=None,
                    gzip=True,
                    convert_original_dicom=True,
                    series_id=None,  # noqa: C901 E501
                    xy_scaling_factor=1,
                    crop_mask=False
                    ):
    """
    Converts A DICOM and DICOM RT Struct file to nii

    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param output_path: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :param series_id: Optional, the Series Instance UID. Use  to specify the ID corresponding to the image if there are
    dicoms from more than one series in `dicom_file` folder

    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    output_path = os.path.join(output_path, '')  # make sure trailing slash is there

    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    if not os.path.exists(dicom_file):
        raise PathDoesNotExistException(f'DICOM path does not exists: {dicom_file}')


    if structures is None:
        structures = []

    os.makedirs(output_path, exist_ok=True)

    filename_converter = FilenameConverter()
    rtreader = RtStructInputAdapter()

    rtstructs = rtreader.ingest(rtstruct_file)
    dicom_image = DcmInputAdapter().ingest(dicom_file, series_id=series_id)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()

    # Sort structures. If starts with ~ move to skip_structures
    skip_structures = []
    for structure in reversed(structures):
        if structure.startswith("~"):
            skip_structures.append(structure[1:])
            structures.remove(structure)

    for rtstruct in rtstructs:
        if len(structures) == 0 or rtstruct['name'] in structures:

            # Check if structure is in skipstructure
            if rtstruct["name"] in skip_structures:
                print(f"Skipping {rtstruct['name']}")
                continue

            if 'sequence' not in rtstruct:
                logging.info('Skipping mask {} no shape/polygon found'.format(rtstruct['name']))
                continue

            logging.info('Working on mask {}'.format(rtstruct['name']))
            try:
                mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'],
                                                          dicom_image,
                                                          xy_scaling_factor=xy_scaling_factor,
                                                          crop_mask=crop_mask)
            except ContourOutOfBoundsException:
                logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                continue
            except TypeError:
                logging.warning(f'Structure {rtstruct["name"]} is empty')
                continue
            except Exception as e:
                logging.warning(f'Structure {rtstruct["name"]} raises exception {str(e)}')
                continue
            
            #  Generate mask name
            mask_filename = filename_converter.convert(f'mask_{rtstruct["name"]}')
            nii_output_adapter.write(mask, f'{output_path}{mask_filename}', gzip)

    if convert_original_dicom:
        logging.info('Converting original DICOM to nii')
        nii_output_adapter.write(dicom_image, f'{output_path}image', gzip)

    logging.info('Success!')
