# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for ePSF model error estimation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.psf.epsf import EPSFFitter
from photutils.psf.epsf_model_error import EPSFErrorMap, calc_epsf_error
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.gridded_epsf import GriddedEPSFFitter
from photutils.psf.gridded_models import GriddedPSFModel
from photutils.psf.image_models import ImagePSF


class _SimpleWCS:
    def pixel_to_world_values(self, x, y):
        return x, y

    def world_to_pixel_values(self, x, y):
        return x, y


def test_calc_epsf_error_imagepsf():
    epsf = ImagePSF(np.zeros((5, 5), dtype=float), oversampling=2)

    data1 = np.zeros((3, 3), dtype=float)
    data2 = np.zeros((3, 3), dtype=float)
    data1[1, 1] = 1.0
    data2[1, 1] = 3.0

    star1 = EPSFStar(data1, weights=np.full((3, 3), 1.0e20),
                     cutout_center=(1.0, 1.0))
    star2 = EPSFStar(data2, weights=np.full((3, 3), 1.0e20),
                     cutout_center=(1.0, 1.0))
    star1.flux = 10.0
    star2.flux = 10.0
    stars = EPSFStars([star1, star2])

    error_map = calc_epsf_error(epsf, stars, fit_stars=False)

    assert isinstance(error_map, EPSFErrorMap)
    assert error_map.data.shape == epsf.data.shape
    assert_allclose(error_map.mean_residual[2, 2], 0.2)
    assert_allclose(error_map.residual_variance[2, 2], 0.01)
    assert_allclose(error_map.variance[2, 2], 0.01)
    assert_allclose(error_map.data[2, 2], 0.1)
    assert error_map.n_samples[2, 2] == 2
    assert_allclose(error_map.evaluate(0.0, 0.0), 0.1)


def test_calc_epsf_error_noise_subtraction():
    epsf = ImagePSF(np.zeros((5, 5), dtype=float), oversampling=2)

    data1 = np.zeros((3, 3), dtype=float)
    data2 = np.zeros((3, 3), dtype=float)
    data1[1, 1] = 1.0
    data2[1, 1] = 3.0

    weights = np.full((3, 3), 100.0)
    star1 = EPSFStar(data1, weights=weights, cutout_center=(1.0, 1.0))
    star2 = EPSFStar(data2, weights=weights, cutout_center=(1.0, 1.0))
    star1.flux = 10.0
    star2.flux = 10.0

    error_map = calc_epsf_error(epsf, EPSFStars([star1, star2]),
                                fit_stars=False)

    assert_allclose(error_map.noise_variance[2, 2], 1.0e-6)
    assert_allclose(error_map.variance[2, 2], 0.009999)


def test_calc_epsf_error_excludes_zero_weight_samples():
    epsf = ImagePSF(np.zeros((5, 5), dtype=float), oversampling=2)

    data = np.zeros((3, 3), dtype=float)
    data[1, 1] = 1.0
    weights1 = np.full((3, 3), 1.0e20)
    weights2 = np.full((3, 3), 1.0e20)
    weights2[1, 1] = 0.0

    star1 = EPSFStar(data, weights=weights1, cutout_center=(1.0, 1.0))
    star2 = EPSFStar(data, weights=weights2, cutout_center=(1.0, 1.0))
    star1.flux = 10.0
    star2.flux = 10.0

    error_map = calc_epsf_error(epsf, EPSFStars([star1, star2]),
                                fit_stars=False, min_samples=1)

    assert error_map.n_samples[2, 2] == 1
    assert error_map.variance[2, 2] == 0.0
    assert_allclose(error_map.mean_residual[2, 2], 0.1)


def test_calc_epsf_error_inputs():
    epsf = ImagePSF(np.zeros((5, 5), dtype=float), oversampling=2)

    with pytest.raises(TypeError, match='stars must be an EPSFStars'):
        calc_epsf_error(epsf, [], fit_stars=False)

    star = EPSFStar(np.ones((3, 3)), cutout_center=(1.0, 1.0))
    star._excluded_from_fit = True

    with pytest.raises(ValueError, match='non-excluded star'):
        calc_epsf_error(epsf, EPSFStars([star]), fit_stars=False)


def test_epsf_error_map_variance_interpolation_is_nonnegative():
    variance = np.zeros((5, 5), dtype=float)
    variance[2, 2] = 1.0
    error_map = EPSFErrorMap(np.sqrt(variance), variance=variance,
                             oversampling=2)

    x = np.linspace(-1.0, 1.0, 41)
    xx, yy = np.meshgrid(x, x)
    assert np.all(error_map.evaluate_variance(xx, yy) >= 0.0)


def test_forced_photometry_preserves_model_parameter_state():
    epsf = ImagePSF(np.ones((5, 5), dtype=float), oversampling=1)
    epsf.x_0.fixed = False
    epsf.y_0.fixed = False
    star = EPSFStar(np.ones((5, 5), dtype=float),
                    cutout_center=(2.0, 2.0))

    calc_epsf_error(epsf, EPSFStars([star]), forced_photometry=True)

    assert not epsf.x_0.fixed
    assert not epsf.y_0.fixed


def test_forced_photometry_gridded_model():
    from astropy.nddata import NDData

    psfs = np.ones((4, 5, 5), dtype=float)
    meta = {'grid_xypos': np.array([[0.0, 0.0], [1.0, 0.0],
                                   [0.0, 1.0], [1.0, 1.0]]),
            'oversampling': 1}
    model = GriddedPSFModel(NDData(psfs, meta=meta))
    linked_stars = []
    for origin in ((0, 0), (5, 5)):
        star = EPSFStar(np.ones((5, 5), dtype=float),
                        cutout_center=(2.0, 2.0), origin=origin,
                        wcs_large=_SimpleWCS())
        star.flux = 25.0
        linked_stars.append(star)
    linked = LinkedEPSFStar(linked_stars)
    stars = EPSFStars([linked])
    error_map = calc_epsf_error(
        model, stars, forced_photometry=True,
        fitter=GriddedEPSFFitter(
            (10, 10), (2, 2), fitter=EPSFFitter(fit_boxsize=None)))

    assert isinstance(error_map, EPSFErrorMap)
    assert_allclose(linked_stars[0].center, linked_stars[1].center)
