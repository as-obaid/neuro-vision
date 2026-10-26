"""Write synthetic DICOM series for tests."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"


def fake_pixels(shape: tuple[int, int, int]) -> np.ndarray:
    """Deterministic uint16 volume of shape (slices, rows, cols)."""
    return (np.arange(int(np.prod(shape))) % 4000).astype(np.uint16).reshape(shape)


def write_series(
    folder: Path,
    *,
    shape: tuple[int, int, int],
    pixel_spacing: tuple[float, float],
    slice_spacing: float,
    orientation: Sequence[float] = (1, 0, 0, 0, 1, 0),
    positions: Sequence[Sequence[float]] | None = None,
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
    rle: bool = False,
    multiframe: bool = False,
    series_uid: str | None = None,
    study_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    patient_id: str = "P0",
    study_date: str = "20200101",
    pixels: np.ndarray | None = None,
    top_level_geometry: bool = True,
    photometric: str = "MONOCHROME2",
    samples_per_pixel: int = 1,
) -> list[Path]:
    """Write one series into `folder` and return the files, sorted by name.

    `positions` is one ImagePositionPatient per slice; by default slices stack along the plane
    normal, `slice_spacing` apart. With `multiframe` all slices go into one file that carries
    NumberOfFrames, the first slice's position, and every slice's position, orientation and pixel
    spacing per frame. `top_level_geometry=False` leaves orientation, spacing and position out of
    the top level. `pixels` (slices, rows, cols) replaces the default pattern.
    """
    n, rows, cols = shape
    row_dir = np.array(orientation[:3], dtype=float)
    col_dir = np.array(orientation[3:], dtype=float)
    normal = np.cross(row_dir, col_dir)
    if positions is None:
        positions = [tuple(normal * slice_spacing * i) for i in range(n)]
    uids = (
        series_uid or generate_uid(),
        study_uid or generate_uid(),
        frame_of_reference_uid or generate_uid(),
    )
    pixels = fake_pixels(shape) if pixels is None else pixels
    folder.mkdir(parents=True, exist_ok=True)
    groups = [(0, pixels)] if multiframe else [(i, pixels[i : i + 1]) for i in range(n)]
    paths = []
    for index, data in groups:
        path = folder / f"{index:04d}.dcm"
        ds = _dataset(path, uids, patient_id, study_date)
        ds.PhotometricInterpretation = photometric
        ds.SamplesPerPixel = samples_per_pixel
        ds.Rows, ds.Columns = rows, cols
        ds.SliceThickness = slice_spacing
        ds.SpacingBetweenSlices = slice_spacing
        if top_level_geometry:
            ds.PixelSpacing = list(pixel_spacing)
            ds.ImageOrientationPatient = list(orientation)
            ds.ImagePositionPatient = list(positions[index])
        ds.InstanceNumber = index + 1
        ds.RescaleSlope = rescale_slope
        ds.RescaleIntercept = rescale_intercept
        ds.RescaleType = "HU"
        if multiframe:
            ds.NumberOfFrames = n
            ds.PerFrameFunctionalGroupsSequence = [
                _frame_group(p, orientation, pixel_spacing) for p in positions
            ]
        ds.PixelData = data.tobytes()
        if rle:
            ds.compress(RLELossless)
        ds.save_as(path)
        paths.append(path)
    return paths


def _frame_group(
    position: Sequence[float], orientation: Sequence[float], pixel_spacing: Sequence[float]
) -> Dataset:
    place, plane, measures = Dataset(), Dataset(), Dataset()
    place.ImagePositionPatient = list(position)
    plane.ImageOrientationPatient = list(orientation)
    measures.PixelSpacing = list(pixel_spacing)
    group = Dataset()
    group.PlanePositionSequence = [place]
    group.PlaneOrientationSequence = [plane]
    group.PixelMeasuresSequence = [measures]
    return group


def _dataset(
    path: Path, uids: tuple[str, str, str], patient_id: str, study_date: str
) -> FileDataset:
    series_uid, study_uid, frame_uid = uids
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CT_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.FrameOfReferenceUID = frame_uid
    ds.PatientID = patient_id
    ds.PatientName = "Synthetic^Test"
    ds.StudyDate = study_date
    ds.Modality = "CT"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    return ds
