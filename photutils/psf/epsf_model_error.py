# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools to estimate ePSF model errors from fitted stars.
"""

import copy

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates

from photutils.psf.epsf import EPSFFitter
from photutils.psf.epsf_stars import EPSFStars, LinkedEPSFStar
from photutils.psf.gridded_epsf import GriddedEPSFFitter
from photutils.psf.gridded_models import GriddedPSFModel
from photutils.psf.image_models import ImagePSF
from photutils.psf.spatial_epsf import SpatialEPSFFitter, SpatialEPSFModel
from photutils.psf.variable_epsf import VariableEPSFFitter, VariableEPSFModel
from photutils.utils._round import py2intround

__all__ = ['EPSFErrorMap', 'calc_epsf_error']


class _NonnegativeImageInterpolator:
    """Shape-preserving interpolation of a nonnegative image."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or min(self.data.shape) < 1:
            raise ValueError('interpolated images must be a non-empty 2D '
                             'array')
        self._singleton_x = self.data.shape[1] == 1
        self._singleton_y = self.data.shape[0] == 1

    def evaluate(self, x, y):
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if self._singleton_x and self._singleton_y:
            return np.full(x.shape, self.data[0, 0], dtype=float)
        if self._singleton_x:
            values = np.interp(y.ravel(), np.arange(self.data.shape[0]),
                               self.data[:, 0])
            return np.maximum(values.reshape(x.shape), 0.0)
        if self._singleton_y:
            values = np.interp(x.ravel(), np.arange(self.data.shape[1]),
                               self.data[0])
            return np.maximum(values.reshape(x.shape), 0.0)
        coords = np.vstack((y.ravel(), x.ravel()))
        values = map_coordinates(self.data, coords, order=1, mode='nearest')
        return np.maximum(values.reshape(x.shape), 0.0)


class EPSFErrorMap:
    """
    An oversampled ePSF error map.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The ePSF standard-deviation error image in oversampled
        coordinates.

    variance : 2D `~numpy.ndarray`, optional
        The ePSF model variance image. If `None`, it is calculated as
        ``data**2``.

    mean_residual : 2D `~numpy.ndarray`, optional
        The mean normalized residual image.

    residual_variance : 2D `~numpy.ndarray`, optional
        The normalized residual variance image.

    noise_variance : 2D `~numpy.ndarray`, optional
        The normalized noise variance image.

    n_samples : 2D `~numpy.ndarray`, optional
        The number of residual samples contributing to each output pixel.

    oversampling : int or (int, int), optional
        Oversampling factor of the error map.

    origin : tuple of float, optional
        Origin of the oversampled grid in oversampled-pixel coordinates.
        If `None`, the geometric center of the array is used.

    fill_value : float, optional
        Value returned for evaluations outside the valid domain.

    fit_uncertainties: dict, optional
        The uncertainties in the star fit parameters (flux, x, y) for each of 
        the stars used in the error map construction.
    """

    def __init__(
        self,
        data,
        *,
        variance=None,
        mean_residual=None,
        residual_variance=None,
        noise_variance=None,
        n_samples=None,
        oversampling=1,
        origin=None,
        fill_value=0.0,
        fit_uncertainties=None
    ):
        self.data = np.asanyarray(data, dtype=float)

        if variance is None:
            variance = self.data**2

        self.variance = np.asanyarray(variance, dtype=float)
        self.mean_residual = mean_residual
        self.residual_variance = residual_variance
        self.noise_variance = noise_variance
        self.n_samples = n_samples

        self._oversampling = np.broadcast_to(
            np.asarray(oversampling, dtype=int), (2,)
        )

        if origin is None:
            ny, nx = self.data.shape
            origin = ((nx - 1) / 2.0, (ny - 1) / 2.0)

        self._origin = np.asarray(origin, dtype=float)
        self.fill_value = float(fill_value)

        self._data_interpolator = RectBivariateSpline(
            np.arange(self.data.shape[0], dtype=float),
            np.arange(self.data.shape[1], dtype=float), self.data,
            ky=min(3, self.data.shape[0] - 1),
            kx=min(3, self.data.shape[1] - 1))
        self._variance_interpolator = _NonnegativeImageInterpolator(
            self.variance)

        self.fit_uncertainties = fit_uncertainties

    @property
    def shape(self):
        """
        Shape of the oversampled error map.
        """
        return self.data.shape

    @property
    def extent(self):
        """
        Extent in detector-pixel coordinates suitable for imshow.

        Returns
        -------
        extent : tuple
            (xmin, xmax, ymin, ymax)
        """
        ny, nx = self.shape

        return (
            (-self.origin[0] - 0.5) / self.oversampling[0],
            (nx - self.origin[0] - 0.5) / self.oversampling[0],
            (-self.origin[1] - 0.5) / self.oversampling[1],
            (ny - self.origin[1] - 0.5) / self.oversampling[1],
        )

    @property
    def oversampling(self):
        if hasattr(self, "_oversampling"):
            return np.asarray(self._oversampling, dtype=int)

        return np.asarray(self.__dict__["oversampling"], dtype=int)


    @property
    def origin(self):
        if hasattr(self, "_origin"):
            return np.asarray(self._origin, dtype=float)

        return np.asarray(self.__dict__["origin"], dtype=float)

    @property
    def _interpolator(self):
        return self._data_interpolator

    def evaluate(self, x, y):
        """
        Evaluate the error map at detector-pixel coordinates.

        Parameters
        ----------
        x, y : array_like
            Coordinates in detector-pixel units relative to the PSF
            center. The origin (0, 0) corresponds to the PSF center.

        Returns
        -------
        values : ndarray
            Interpolated error values.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_img = x * self.oversampling[0] + self.origin[0]
        y_img = y * self.oversampling[1] + self.origin[1]

        values = self._data_interpolator.ev(y_img.ravel(), x_img.ravel())

        valid = (
            (x_img.ravel() >= 0)
            & (x_img.ravel() <= self.shape[1] - 1)
            & (y_img.ravel() >= 0)
            & (y_img.ravel() <= self.shape[0] - 1)
        )

        result = np.full(values.shape, self.fill_value, dtype=float)
        result[valid] = values[valid]

        return result.reshape(x.shape)

    def evaluate_variance(self, x, y):
        """
        Evaluate the variance map at detector-pixel coordinates.

        Parameters
        ----------
        x, y : array_like
            Coordinates in detector-pixel units relative to the PSF
            center. The origin (0, 0) corresponds to the PSF center.

        Returns
        -------
        values : ndarray
            Interpolated variance values.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_img = x * self.oversampling[0] + self.origin[0]
        y_img = y * self.oversampling[1] + self.origin[1]

        values = self._variance_interpolator.evaluate(x_img, y_img).ravel()

        valid = (
            (x_img.ravel() >= 0)
            & (x_img.ravel() <= self.shape[1] - 1)
            & (y_img.ravel() >= 0)
            & (y_img.ravel() <= self.shape[0] - 1)
        )

        result = np.full(values.shape, 0.0, dtype=float)
        result[valid] = values[valid]

        return result.reshape(x.shape)

    def plot(
        self,
        supersampling=None,
        ax=None,
        **imshow_kwargs,
    ):
        """
        Plot the standard-deviation error map.

        Parameters
        ----------
        supersampling : int or (int, int), optional
            Supersampling factor used for display. If `None`, use the
            native oversampling of the error map.

        ax : `~matplotlib.axes.Axes`, optional
            Axes on which to draw.

        **imshow_kwargs
            Additional keywords passed to `matplotlib.pyplot.imshow`.

        Returns
        -------
        image : `~matplotlib.image.AxesImage`
            The plotted image.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if supersampling is None:
            supersampling = self.oversampling

        supersampling = np.broadcast_to(
            np.asarray(supersampling, dtype=int), (2,)
        )

        ny, nx = self.shape

        if np.array_equal(supersampling, self.oversampling):
            image = self.data
        else:
            xmin, xmax, ymin, ymax = self.extent

            nx_out = int(
                np.round(
                    nx * supersampling[0] / self.oversampling[0]
                )
            )
            ny_out = int(
                np.round(
                    ny * supersampling[1] / self.oversampling[1]
                )
            )

            x = np.linspace(xmin, xmax, nx_out)
            y = np.linspace(ymin, ymax, ny_out)

            xx, yy = np.meshgrid(x, y)
            image = self.evaluate(xx, yy)

        artist = ax.imshow(
            image,
            origin="lower",
            extent=self.extent,
            **imshow_kwargs,
        )

        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        return artist


def calc_epsf_error(epsf, stars, *, fit_stars=True, fitter=None,
                    min_samples=2, forced_photometry=False):
    """
    Estimate an ePSF model error map from residuals to stars.

    Parameters
    ----------
    epsf : `ImagePSF`, `SpatialEPSFModel`, `VariableEPSFModel`, or \
            `GriddedPSFModel`
        The ePSF model.

    stars : `EPSFStars`
        The stars used to estimate the ePSF error.

    fit_stars : bool, optional
        If `True`, fit ``epsf`` to ``stars`` before calculating residuals.
        If `False`, use the fluxes and centroids already stored on the
        stars.

    fitter : object, optional
        The fitter object to use when ``fit_stars`` is `True`. If `None`,
        a default fitter appropriate for ``epsf`` is used.

    min_samples : int, optional
        The minimum number of residual samples required in an output
        pixel. Pixels with fewer samples are set to zero in the returned
        error map.

    forced_photometry: bool, optional
        Only applicable if fit_stars is `True`. If `True`, then star fitting
        uses forced photometry where linked stars share their average
        position and only the flux is fit. Default
        is `False`.

    Returns
    -------
    error_map : `EPSFErrorMap`
        The estimated ePSF error map. The ``data`` attribute stores the
        standard-deviation error map in oversampled coordinates, and the
        ``variance`` attribute stores the corresponding variance.
    """
    if not isinstance(stars, EPSFStars):
        raise TypeError('stars must be an EPSFStars object')

    min_samples = int(min_samples)
    if min_samples < 1:
        raise ValueError('min_samples must be >= 1')

    if fit_stars:
        if fitter is None:
            fitter = _default_fitter(epsf)

        if forced_photometry:
            stars.constrain_linked_centres(remove_outliers=True)
            if isinstance(epsf, GriddedPSFModel):
                if not isinstance(fitter, GriddedEPSFFitter):
                    raise TypeError('forced photometry for GriddedPSFModel '
                                    'requires a GriddedEPSFFitter')
                for item in stars:
                    if isinstance(item, LinkedEPSFStar):
                        item.constrain_centers(remove_outliers=True)
                stars = fitter(epsf, stars, forced_photometry=True)
            elif isinstance(epsf, VariableEPSFModel):
                if not isinstance(fitter, VariableEPSFFitter):
                    raise TypeError('forced photometry for '
                                    'VariableEPSFModel requires a '
                                    'VariableEPSFFitter')
                fit_fitter = copy.copy(fitter)
                fit_fitter.forced_photometry = True
                stars = fit_fitter(epsf, stars)
            elif isinstance(epsf, SpatialEPSFModel):
                if not isinstance(fitter, SpatialEPSFFitter):
                    raise TypeError('forced photometry for spatial ePSF '
                                    'models requires a SpatialEPSFFitter')
                fit_fitter = copy.copy(fitter)
                fit_fitter.forced_photometry = True
                stars = fit_fitter(epsf, stars)
            elif isinstance(epsf, ImagePSF):
                fit_model = epsf.copy()
                fit_model.x_0.fixed = True
                fit_model.y_0.fixed = True
                stars = fitter(fit_model, stars)
            else:
                raise TypeError('forced photometry requires an ImagePSF or '
                                'spatial ePSF model')
        else:
            stars = fitter(epsf, stars)

    good_stars = stars.all_good_stars
    if len(good_stars) == 0:
        raise ValueError('stars must contain at least one non-excluded star')

    local_epsf0 = _local_image_psf(epsf, good_stars[0])
    shape = local_epsf0.data.shape
    oversampling = local_epsf0.oversampling
    origin = local_epsf0.origin
    fill_value = local_epsf0.fill_value

    residuals = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
    noise_vars = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
    sigma_fluxes = []
    sigma_xs = []
    sigma_ys = []

    for star in good_stars:

        if not np.isfinite(star.flux) or star.flux == 0.0:
            continue

        local_epsf = _local_image_psf(epsf, star)
        if local_epsf.data.shape != shape:
            raise ValueError('all local ePSF models must have the same shape')
        if not np.array_equal(local_epsf.oversampling, oversampling):
            raise ValueError('all local ePSF models must have the same '
                             'oversampling')

        resid = star.compute_residual_image(local_epsf) / star.flux
        weights = np.asanyarray(star.weights, dtype=float)
        valid = (~star.mask & np.isfinite(resid) & np.isfinite(weights)
                 & (weights > 0.0))
        if not np.any(valid):
            continue

        yidx, xidx = np.indices(star.shape)
        x = oversampling[1] * (xidx[valid] - star.cutout_center[0])
        y = oversampling[0] * (yidx[valid] - star.cutout_center[1])
        xbin = py2intround(x + origin[0])
        ybin = py2intround(y + origin[1])

        in_bounds = ((xbin >= 0) & (xbin < shape[1])
                     & (ybin >= 0) & (ybin < shape[0]))
        xbin = xbin[in_bounds]
        ybin = ybin[in_bounds]
        resid_vals = resid[valid][in_bounds]
        noise_var_vals = ((1.0 / weights[valid][in_bounds])**2) / star.flux**2

        for yy, xx, value, noise_var in zip(ybin, xbin, resid_vals,
                                            noise_var_vals, strict=True):
            residuals[yy][xx].append(value)
            noise_vars[yy][xx].append(noise_var)

        sigma_flux = None
        sigma_x = None
        sigma_y = None
        if hasattr(star, '_fit_info'):
            if 'param_cov' in star._fit_info:
                cov = star._fit_info["param_cov"]
                if cov is not None:
                    sigma_flux = np.sqrt(cov[0, 0]) / star.flux
                    if not forced_photometry:
                        sigma_x = np.sqrt(cov[1, 1])
                        sigma_y = np.sqrt(cov[2, 2])
                    else:
                        sigma_x = None
                        sigma_y = None
                else:
                    jac = star._fit_info["jac"]
                    cost = star._fit_info["cost"]
                    n_data, n_par = jac.shape
                    sigma2 = 2 * cost / (n_data - n_par)
                    cov = sigma2 * np.linalg.inv(jac.T @ jac)
                    sigma_flux = np.sqrt(cov[0, 0]) / star.flux
                    if not forced_photometry:
                        sigma_x = np.sqrt(cov[1, 1])
                        sigma_y = np.sqrt(cov[2, 2])
                    else:
                        sigma_x = None
                        sigma_y = None
        sigma_fluxes.append(sigma_flux)
        sigma_xs.append(sigma_x)
        sigma_ys.append(sigma_y)

    fit_uncertainties = {
        "sigma_flux": sigma_fluxes,
        "sigma_x": sigma_xs,
        "sigma_y": sigma_ys,
    }
    mean_residual = np.full(shape, np.nan)
    residual_variance = np.full(shape, np.nan)
    noise_variance = np.full(shape, np.nan)
    model_variance = np.zeros(shape, dtype=float)
    n_samples = np.zeros(shape, dtype=int)

    for yy in range(shape[0]):
        for xx in range(shape[1]):
            values = np.asarray(residuals[yy][xx], dtype=float)
            if values.size < min_samples:
                continue
            noises = np.asarray(noise_vars[yy][xx], dtype=float)
            mean = np.mean(values)
            resid_var = np.mean((values - mean)**2)
            noise_var = np.mean(noises)

            mean_residual[yy, xx] = mean
            residual_variance[yy, xx] = resid_var
            noise_variance[yy, xx] = noise_var
            model_variance[yy, xx] = max(0.0, resid_var - noise_var)
            n_samples[yy, xx] = values.size

    error_data = np.sqrt(model_variance)
    return EPSFErrorMap(error_data, variance=model_variance,
                        mean_residual=mean_residual,
                        residual_variance=residual_variance,
                        noise_variance=noise_variance,
                        n_samples=n_samples, oversampling=oversampling,
                        origin=origin, fill_value=fill_value,
                        fit_uncertainties=fit_uncertainties)


def _default_fitter(epsf):
    if isinstance(epsf, VariableEPSFModel):
        return VariableEPSFFitter(fitter=TRFLSQFitter(calc_uncertainties=True))
    if isinstance(epsf, SpatialEPSFModel):
        return SpatialEPSFFitter(fitter=TRFLSQFitter(calc_uncertainties=True))
    if isinstance(epsf, GriddedPSFModel):
        return GriddedEPSFFitter(fitter=TRFLSQFitter(calc_uncertainties=True))
    if isinstance(epsf, ImagePSF):
        return EPSFFitter(fitter=TRFLSQFitter(calc_uncertainties=True))
    raise TypeError('epsf must be an ImagePSF, SpatialEPSFModel, '
                    'VariableEPSFModel, or GriddedPSFModel')


def _local_image_psf(epsf, star):
    if isinstance(epsf, VariableEPSFModel):
        flux = None
        if epsf.has_flux_dependency:
            flux = star.flux
            exposure_time = getattr(star, 'exposure_time', None)
            if exposure_time is not None:
                flux *= exposure_time
        fwhm = getattr(star, 'fwhm', None)
        return epsf.make_image_psf(star.center[0], star.center[1],
                                   flux=flux, fwhm=fwhm)
    if isinstance(epsf, SpatialEPSFModel):
        return epsf.make_image_psf(star.center[0], star.center[1])
    if isinstance(epsf, GriddedPSFModel):
        data = _interpolate_gridded_epsf(epsf, star.center[0],
                                         star.center[1])
        return ImagePSF(data, oversampling=epsf.oversampling,
                        fill_value=epsf.fill_value)
    if isinstance(epsf, ImagePSF):
        return epsf
    raise TypeError('epsf must be an ImagePSF, SpatialEPSFModel, '
                    'VariableEPSFModel, or GriddedPSFModel')


def _interpolate_gridded_epsf(epsf, x, y):
    data = epsf.data
    if data.shape[0] == 1:
        return data[0]

    grid_idx, grid_xy = epsf._find_bounding_points(x, y)
    weights = epsf._calc_bilinear_weights(x, y, grid_xy)

    result = np.zeros_like(data[grid_idx[0]], dtype=float)
    for idx, weight in zip(grid_idx, weights, strict=True):
        if weight != 0.0:
            result += weight * data[idx]
    return result
