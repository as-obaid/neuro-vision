"""Read DICOM headers and pixels, limited to the tags Kaggle keeps at test time.

The list is copied from the competition Data page. TransferSyntaxUID lives in the file meta.
"""

from pathlib import Path

import pydicom
from pydicom.dataset import Dataset

TEST_TAGS: tuple[str, ...] = (
    "BitsAllocated",
    "BitsStored",
    "Columns",
    "FrameOfReferenceUID",
    "HighBit",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "InstanceNumber",
    "Modality",
    "PatientID",
    "PerFrameFunctionalGroupsSequence",
    "PhotometricInterpretation",
    "PixelRepresentation",
    "PixelSpacing",
    "PlanarConfiguration",
    "RescaleIntercept",
    "RescaleSlope",
    "RescaleType",
    "Rows",
    "SOPClassUID",
    "SOPInstanceUID",
    "SamplesPerPixel",
    "SliceThickness",
    "SpacingBetweenSlices",
    "StudyInstanceUID",
    "TransferSyntaxUID",
)


def read_header(path: Path) -> Dataset:
    return pydicom.dcmread(path, specific_tags=list(TEST_TAGS))


def read_series(series_dir: Path) -> list[Dataset]:
    tags = [*TEST_TAGS, "PixelData"]
    return [pydicom.dcmread(p, specific_tags=tags) for p in sorted(series_dir.glob("*.dcm"))]
