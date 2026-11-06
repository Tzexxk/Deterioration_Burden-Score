"""Historical pixel transformation regression on generated DICOMs."""
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    ExplicitVRLittleEndian,
    RLELossless,
    generate_uid,
)

from src.common.paths import read_json as _json_file
from src.imaging.preprocessing import (
    process_dataset,
    validate_pixels,
    validate_settings,
)

PROJECT = Path(__file__).resolve().parents[2]
def dicom(values=None, photo='MONOCHROME2', *, frames=1, color=False):
    if values is None:
        values = np.arange(620, dtype=np.uint16).reshape(20, 31)
    values = np.asarray(values)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset('', {}, file_meta=meta, preamble=b'\x00' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = values.shape[-3] if color else values.shape[-2]
    ds.Columns = values.shape[-2] if color else values.shape[-1]
    ds.SamplesPerPixel = 3 if color else 1
    ds.NumberOfFrames = frames
    ds.PhotometricInterpretation = photo
    if color:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = int(values.dtype.kind == 'i')
    ds.PixelData = values.astype('<i2' if ds.PixelRepresentation else '<u2').tobytes()
    return ds

def lut(values, first=0, bits=16):
    item = Dataset()
    item.LUTDescriptor = [len(values), first, bits]
    item.add_new(2633734, 'US', list(map(int, values)))
    return item

class ImageTransformTests(unittest.TestCase):

    def setUp(self):
        self.settings = _json_file(PROJECT / 'configs/data.template/lenient/imaging/processing.json')

    def test_direction_presentation_and_odd_padding(self):
        a = process_dataset(dicom(), self.settings)
        b = process_dataset(dicom(photo='MONOCHROME1'), self.settings)
        ds = dicom(photo='MONOCHROME1')
        ds.PresentationLUTShape = 'INVERSE'
        c = process_dataset(ds, self.settings)
        self.assertEqual(c['processing_status'], 'complete')
        np.testing.assert_array_equal(b['pixels'], c['pixels'])
        self.assertEqual(c['presentation_path'], 'shape_inverse')
        self.assertTrue(c['inversion_applied'])
        self.assertEqual(a['resized_height'], 145)
        self.assertEqual(a['pad_top'], 39)
        self.assertEqual(a['pad_bottom'], 40)
        np.testing.assert_allclose(a['pixels'][39:184] + b['pixels'][39:184], 1, atol=2e-07)
        self.assertTrue((a['pixels'][:39] == 0).all())
        ds.PresentationLUTShape = 'IDENTITY'
        conflict = process_dataset(ds, self.settings)
        self.assertEqual(conflict['processing_status'], 'pending_review')
        self.assertIsNone(conflict['pixels'])
        self.assertIsNotNone(conflict['before'])
        ds = dicom()
        ds.PresentationLUTSequence = [lut(range(620))]
        presented = process_dataset(ds, self.settings)
        self.assertEqual(presented['presentation_path'], 'presentation_lut')
        ds.PhotometricInterpretation = 'MONOCHROME1'
        np.testing.assert_array_equal(process_dataset(ds, self.settings)['pixels'], presented['pixels'])
        ds.PresentationLUTShape = 'INVERSE'
        self.assertEqual(process_dataset(ds, self.settings)['error_code'], 'invalid_presentation_lut')

    def test_lut_priority_valid_fallbacks_and_rescale_defaults(self):
        ds = dicom()
        ds.RescaleSlope = 3
        ds.RescaleIntercept = 10
        ds.ModalityLUTSequence = [lut(range(620))]
        ds.VOILUTSequence = [lut(range(620))]
        ds.WindowCenter = 300
        ds.WindowWidth = 200
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['processing_status'], 'complete')
        self.assertEqual(r['modality_path'], 'modality_lut')
        self.assertEqual(r['voi_path'], 'voi_lut')
        ds.VOILUTSequence = [Dataset(), lut(range(620))]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['display_metadata']['voi_lut_index'], 1)
        self.assertIn('invalid_voi_lut_skipped', r['issues'])
        ds.VOILUTSequence = [Dataset()]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['voi_path'], 'window')
        self.assertIn('invalid_voi_lut_fallback', r['issues'])
        ds = dicom()
        ds.RescaleSlope = 2
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['modality_path'], 'rescale')
        self.assertEqual(r['display_metadata']['applied_rescale_intercept'], 0)
        self.assertEqual(r['voi_path'], 'none')
        ds.WindowCenter = [200, 500]
        ds.WindowWidth = [100, 500]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['display_metadata']['applied_window_center'], 200)
        ds.WindowWidth = [0, 500]
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['display_metadata']['window_index'], 1)
        self.assertEqual(r['display_metadata']['applied_window_center'], 500)
        self.assertIn('invalid_window_group_skipped', r['issues'])

    def test_padding_range_percentiles_and_mask(self):
        values = np.arange(-31, 589, dtype=np.int16).reshape(20, 31)
        ds = dicom(values)
        ds.PixelPaddingValue = -31
        ds.PixelPaddingRangeLimit = -1
        r = process_dataset(ds, self.settings)
        expected = np.percentile(values[values >= 0], [1, 99], method='linear')
        np.testing.assert_allclose([r['p1'], r['p99']], expected)
        self.assertEqual(r['valid_pixel_fraction'], 19 / 20)
        self.assertTrue((r['pixels'][r['pad_top']:r['pad_top'] + 3] == 0).all())
        ds.PixelPaddingValue = -1
        ds.PixelPaddingRangeLimit = -31
        np.testing.assert_array_equal(process_dataset(ds, self.settings)['pixels'], r['pixels'])
        ds = dicom(np.full((20, 31), 10, dtype=np.uint16))
        ds.PixelPaddingValue = 10
        self.assertEqual(process_dataset(ds, self.settings)['error_code'], 'no_valid_pixels')

    def test_dx_voi_depth_specialization_and_priority(self):
        for sop in (DigitalXRayImageStorageForPresentation, DigitalXRayImageStorageForProcessing):
            for bits in range(10, 17):
                with self.subTest(sop=sop, bits=bits):
                    ds = dicom()
                    ds.SOPClassUID = sop
                    ds.VOILUTSequence = [lut(np.arange(620) ** 2 // 619, bits=bits)]
                    ds.WindowCenter, ds.WindowWidth = (300, 40)
                    r = process_dataset(ds, self.settings)
                    self.assertEqual(r['processing_status'], 'complete')
                    self.assertEqual(r['voi_path'], 'voi_lut')
                    self.assertEqual(r['issues'], [])
                    self.assertEqual(r['display_metadata']['voi_lut_descriptor'][2], bits)
                    ds.WindowWidth = 500
                    np.testing.assert_array_equal(process_dataset(ds, self.settings)['pixels'], r['pixels'])
        ds = dicom()
        ds.SOPClassUID = DigitalXRayImageStorageForPresentation
        ds.VOILUTSequence = [lut(range(620), bits=9)]
        ds.WindowCenter, ds.WindowWidth = (300, 500)
        r = process_dataset(ds, self.settings)
        self.assertEqual(r['voi_path'], 'window')
        self.assertIn('invalid_voi_lut_fallback', r['issues'])
        ds.VOILUTSequence = [lut([0] * 619 + [2 ** 14], bits=14)]
        self.assertIn('invalid_voi_lut_fallback', process_dataset(ds, self.settings)['issues'])
        ds.SOPClassUID = ComputedRadiographyImageStorage
        ds.VOILUTSequence = [lut(range(620), bits=14)]
        self.assertIn('invalid_voi_lut_fallback', process_dataset(ds, self.settings)['issues'])

    def test_failures_never_return_substitute_pixels(self):
        cases = [dicom(np.ones((20, 31), dtype=np.uint16)), dicom(np.arange(1240, dtype=np.uint16).reshape(2, 20, 31), frames=2), dicom(np.arange(1860, dtype=np.uint16).reshape(20, 31, 3), photo='RGB', color=True)]
        missing = dicom()
        del missing.PixelData
        cases.append(missing)
        for ds in cases:
            r = process_dataset(ds, self.settings)
            self.assertEqual(r['processing_status'], 'fail')
            self.assertIsNone(r['pixels'])
        with patch('pydicom.dataset.Dataset.pixel_array', new_callable=PropertyMock, return_value=np.full((20, 31), np.nan)):
            r = process_dataset(dicom(), self.settings)
            self.assertEqual(r['error_code'], 'nonfinite_pixels')
        with patch('pydicom.dataset.Dataset.pixel_array', new_callable=PropertyMock, side_effect=ImportError('decoder missing')):
            self.assertIn('pixel_decode_error', process_dataset(dicom(), self.settings)['error_code'])

    def test_compressed_decode_and_numeric_contract(self):
        ds = dicom()
        expected = process_dataset(ds, self.settings)['pixels']
        ds.compress(RLELossless)
        result = process_dataset(ds, self.settings)
        self.assertEqual(result['processing_status'], 'complete')
        np.testing.assert_array_equal(expected, result['pixels'])
        validate_pixels(result['pixels'])
        with self.assertRaises(ValueError):
            validate_pixels(np.zeros((224, 224), dtype=np.float64))
        settings = {**self.settings, 'percentiles': [0, 100]}
        with self.assertRaises(ValueError):
            validate_settings(settings)
