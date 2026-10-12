import numpy as np
from numpy.testing import assert_allclose

from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.spatial_epsf import SpatialEPSFBuilder
from photutils.psf.spatial_epsf import SpatialEPSFModel


class _DummyStar:
    def __init__(self):
        self.center = (0.0, 0.0)
        self.flux = 1.0
        # mask has one masked pixel (middle); `_xidx_centered`/_yidx_centered
        # should list only the unmasked sample coordinates
        self.mask = np.array([False, True, False])
        self._xidx_centered = np.array([0.0, 0.0])
        self._yidx_centered = np.array([0.0, 0.0])

    def compute_residual_image(self, local_epsf):
        return np.array([0.1, 0.2, 0.3])


class _DummySpatialModel:
    def __init__(self):
        self.oversampling = (1, 1)
        self.shape = (2, 2)

    def make_image_psf(self, x, y):
        return None


def test_resample_residual_ignores_masked_pixels():
    builder = SpatialEPSFBuilder()
    star = _DummyStar()
    model = _DummySpatialModel()

    resampled, weights, x_coords, y_coords, saturation_mask = (
        builder._resample_residual(star, model))

    assert resampled.shape == model.shape
    assert weights.shape == model.shape
    assert x_coords.shape == model.shape
    assert y_coords.shape == model.shape
    assert saturation_mask.shape == model.shape


def test_spatial_resample_residual_marks_saturated_samples():
    star = EPSFStar(np.ones((3, 3), dtype=float),
                    cutout_center=(1.0, 1.0), origin=(4, 4),
                    exposure_time=2.0)
    star.flux = 9.0
    builder = SpatialEPSFBuilder(detector_shape=(10, 10), oversampling=1,
                                 degree=0, normalise_epsf=False)
    builder.saturation_value = 1.5
    model = SpatialEPSFModel(np.ones((1, 5, 5)), oversampling=1,
                             detector_shape=(10, 10), degree=0,
                             normalize_local_epsf=False)

    residual, weights, x_coords, y_coords, saturation = (
        builder._resample_residual(star, model))

    assert np.count_nonzero(saturation) == 9
    assert np.all(np.isfinite(residual[saturation]))
    assert np.all(weights[saturation] >= 0.0)


def test_spatial_resample_residuals_tracks_linked_star_groups():
    class _IdentityWCS:
        def pixel_to_world_values(self, x, y):
            return x, y

        def world_to_pixel_values(self, x, y):
            return x, y

    stars_in_group = [
        EPSFStar(np.ones((3, 3), dtype=float), cutout_center=(1.0, 1.0),
                 origin=origin, wcs_large=_IdentityWCS())
        for origin in ((2, 2), (5, 5))
    ]
    linked = LinkedEPSFStar(stars_in_group)
    excluded = EPSFStar(np.ones((3, 3), dtype=float),
                        cutout_center=(1.0, 1.0), origin=(8, 8))
    excluded._excluded_from_fit = True
    stars = EPSFStars([linked, excluded])
    builder = SpatialEPSFBuilder(detector_shape=(10, 10), oversampling=1,
                                 degree=0, normalise_epsf=False)
    model = SpatialEPSFModel(np.ones((1, 5, 5)), oversampling=1,
                             detector_shape=(10, 10), degree=0,
                             normalize_local_epsf=False)

    (residuals, weights, x_coords, y_coords, saturation, det_x, det_y,
     group_id) = builder._resample_residuals(stars, model)

    assert residuals.shape == (2, 5, 5)
    assert weights.shape == residuals.shape
    assert x_coords.shape == residuals.shape
    assert y_coords.shape == residuals.shape
    assert saturation.shape == residuals.shape
    assert_allclose(det_x, [3.0, 6.0])
    assert_allclose(det_y, [3.0, 6.0])
    assert_allclose(group_id, [0, 0])
