# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the gridded ePSF builder.
"""

import numpy as np
import pytest
from astropy.nddata import NDData
from numpy.testing import assert_allclose

from photutils.psf.epsf import EPSFFitter
from photutils.psf.epsf_stars import EPSFStar, LinkedEPSFStar
from photutils.psf.epsf_stars import EPSFStars
from photutils.psf.gridded_epsf import (GriddedEPSFBuilder,
                                        GriddedEPSFFitter)
from photutils.psf.gridded_models import GriddedPSFModel
from photutils.psf.image_models import ImagePSF


class SimpleWCS:
    def pixel_to_world_values(self, x, y):
        return x, y

    def world_to_pixel_values(self, lon, lat):
        return lon, lat


class DummyBuilder:
    def __init__(self):
        self.maxiters = 5
        self.calls = []

    def build_epsf(self, stars, *, init_model=None):
        self.calls.append((self.maxiters, stars, init_model))
        return 'epsf', stars


def make_star(x, y):
    data = np.ones((5, 5), dtype=float)
    return EPSFStar(data, cutout_center=(2, 2), origin=(x - 2, y - 2),
                    wcs_large=SimpleWCS())


def test_partition_preserves_linked_stars():
    builder = GriddedEPSFBuilder((100, 100), (1, 2), 2, maxiters=1,
                                 min_stars_per_gridcell=1)
    linked = LinkedEPSFStar([make_star(20, 50), make_star(30, 50)])

    grid_stars = builder._partition_stars_by_gridcell([linked])

    assert isinstance(grid_stars[0][0]._data[0], LinkedEPSFStar)
    assert grid_stars[0][0].n_all_stars == 2
    assert grid_stars[0][1] is None


def test_build_grid_cell_epsf_can_override_maxiters_temporarily():
    dummy_builder = DummyBuilder()

    epsf, stars = GriddedEPSFBuilder._build_grid_cell_epsf(
        dummy_builder, 'stars', init_model='init', maxiters=1)

    assert epsf == 'epsf'
    assert stars == 'stars'
    assert dummy_builder.calls == [(1, 'stars', 'init')]
    assert dummy_builder.maxiters == 5


def test_make_gridded_psf_model_fills_missing_cells_from_nearest_valid():
    builder = GriddedEPSFBuilder((100, 100), (2, 2), 2, maxiters=1,
                                 min_stars_per_gridcell=1)

    epsf1 = ImagePSF(np.full((5, 5), 1.0), oversampling=2)
    epsf2 = ImagePSF(np.full((5, 5), 2.0), oversampling=2)
    epsf3 = ImagePSF(np.full((5, 5), 3.0), oversampling=2)

    model = builder._make_gridded_psf_model([None, epsf1, epsf2, epsf3])

    assert_allclose(model.data[0], epsf1.data)
    assert_allclose(model.data[1], epsf1.data)
    assert_allclose(model.data[2], epsf2.data)
    assert_allclose(model.data[3], epsf3.data)


def test_gridded_epsf_builder_rejects_invalid_initial_model():
    builder = GriddedEPSFBuilder((100, 100), (1, 1), 2,
                                 progress_bar=False)
    with pytest.raises(TypeError, match='init_model must be a GriddedPSFModel'):
        builder.build_epsf(EPSFStars([]), init_model=object())


def test_gridded_epsf_builder_requires_at_least_one_populated_cell():
    builder = GriddedEPSFBuilder((100, 100), (1, 1), 2,
                                 min_stars_per_gridcell=2,
                                 progress_bar=False)

    with pytest.raises(ValueError, match='no grid cells have a valid ePSF'):
        builder.build_epsf(EPSFStars([make_star(50.0, 50.0)]))


def test_gridded_epsf_builder_grid_indices_and_samples():
    builder = GriddedEPSFBuilder((100, 100), (2, 2), 2,
                                 min_stars_per_gridcell=1)
    assert builder._get_grid_indices(-1, 50) is None
    assert builder._get_grid_indices(50, 50) == (1, 1)
    assert builder._get_grid_indices(100, 100) == (1, 1)

    star1 = make_star(25.25, 25.75)
    star2 = make_star(25.75, 25.25)
    linked = LinkedEPSFStar([star1, star2])
    samples = builder._collect_sample_arrays(EPSFStars([linked]))

    assert samples['linked_id'].tolist() == [0, 0]
    counts, unique_counts = builder._compute_subpixel_counts(samples)
    assert counts[0, 0].sum() == 2
    assert unique_counts[0, 0].sum() == 2

    flux_maps = builder._compute_flux_ppe_grid_maps(samples)
    assert_allclose(flux_maps[0, 0], 0.0)
    assert_allclose(flux_maps[1, 1], 0.0)
    assert builder._collect_sample_arrays(EPSFStars([])) is None


def test_gridded_epsf_builder_flux_ppe_fallbacks():
    builder = GriddedEPSFBuilder((100, 100), (1, 1), 2)
    samples = {
        'subpixel_x': np.array([0.1, 0.7, 0.7]),
        'subpixel_y': np.array([0.2, 0.8, 0.8]),
        'flux': np.array([2.0, 4.0, np.nan]),
        'linked_id': np.array([-1, -1, 3]),
        'grid_iy': np.array([0, 0, 0]),
        'grid_ix': np.array([0, 0, 0]),
        'x': np.array([1.0, 2.0, 3.0]),
        'y': np.array([1.0, 2.0, 3.0]),
    }

    flux_maps = builder._compute_flux_ppe_grid_maps(samples)

    assert_allclose(flux_maps[0, 0, 0, 0], -1 / 3)
    assert_allclose(flux_maps[0, 0, 1, 1], 1 / 3)


def test_gridded_epsf_fitter_input_and_fit_box_failures():
    fitter = GriddedEPSFFitter((10, 10), (1, 1))
    with pytest.raises(TypeError, match='must be a GriddedPSFModel'):
        fitter(None, EPSFStars([make_star(5, 5)]))

    model_data = np.ones((1, 5, 5), dtype=float)
    model = GriddedPSFModel(NDData(model_data, meta={
        'grid_xypos': np.array([[5.0, 5.0]]), 'oversampling': 1}))
    star = make_star(5, 5)
    fitter = GriddedEPSFFitter((10, 10), (1, 1),
                               fitter=EPSFFitter(fit_boxsize=(7, 7)))
    with pytest.warns(UserWarning, match='extends beyond the star cutout'):
        failed = fitter._fit_star_with_gridded_model(model, star, (7, 7))
    assert failed._fit_error_status == 1


def test_gridded_epsf_fitter_forced_photometry_keeps_centers_fixed():
    class _ForcedFitter:
        fit_info = {'ierr': 1}

        def __call__(self, model, x, y, z, weights=None, **kwargs):
            assert model.x_0.fixed
            assert model.y_0.fixed
            model.flux = 12.0
            return model

    data = np.ones((1, 5, 5), dtype=float)
    model = GriddedPSFModel(NDData(data, meta={
        'grid_xypos': np.array([[5.0, 5.0]]), 'oversampling': 1}))
    star = make_star(5.0, 5.0)
    initial_center = star.center
    fitter = GriddedEPSFFitter(
        (10, 10), (1, 1),
        fitter=EPSFFitter(fitter=_ForcedFitter(), fit_boxsize=None))

    fitted = fitter(model, EPSFStars([star]), forced_photometry=True)

    assert_allclose(fitted[0].center, initial_center)
    assert fitted[0].flux == 12.0
    assert fitted[0]._fit_error_status == 0


def test_gridded_epsf_builder_requires_valid_grid_psf():
    builder = GriddedEPSFBuilder((100, 100), (1, 1), 2)
    with pytest.raises(ValueError, match='no grid cells have a valid ePSF'):
        builder._make_gridded_psf_model([None])
