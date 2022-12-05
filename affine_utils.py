import numpy as np

import astropy.units as u
import astropy.time as time
import astropy.coordinates as coord

import matplotlib.pyplot as plt

from affine_fitter import AffineTransformFitter


def make_bbox(lon_center, lat_center, bbox_size, n_samples):
    """Make a square bounding box centered around the given coordinates.

    No unit conversions occur for the given values, i.e. the given values are
    assumed to all be in the same units.

    Parameters
    ----------
    lon_center : `float`
        Longitude of the center coordinate.
    lat_center : `float`
        Latitude of the center coordinate.
    bbox_size : `float`
        Size of the bounding box side.
    n_samples : `int`
        Number of points to sample per side of the bounding box.

    Returns
    -------
    bbox : `numpy.array`
        Bounding box.
    """
    lon_range = np.linspace(
        lon_center-bbox_size/2,
        lon_center+bbox_size/2,
        n_samples
    )
    lat_range = np.linspace(
        lat_center-bbox_size/2,
        lat_center+bbox_size/2,
        n_samples
    )

    bbox = []
    for loni, lati in zip(lon_range, lat_range):
        bbox.append([lon_range[0], lati])
        bbox.append([lon_range[-1], lati])
        bbox.append([loni, lat_range[0]])
        bbox.append([loni, lat_range[-1]])

    # it might feel more correct to list points as [[x,y], [x,y]...] but it's
    # a lot easier to separate x and y into individual lists aligned by index
    bbox = np.array(bbox)
    return np.array([bbox[:,0], bbox[:, 1]])


def make_heliocentric_bbox(helio_center, bbox_size=1*u.degree, n_samples=100):
    """Create a bounding box in heliocentric True Ecliptic coordinate system.

    Parameters
    ----------
    helio_center : `astropy.coordinates.HeliocentricTrueEcliptic`
        Coordinates of the center of the bounding box.
    bbox_size : `astropy.units.Quantity`
        Size of the bounding box side.
    n_samples : `int`
        Number of points to sample per side of the bounding box.

    Returns
    -------
    bbox : `astropy.coordinates.HeliocentricTrueEcliptic`
        Bounding box in heliocentric true ecliptic coordinate system.
    """
    bbox = make_bbox(
        helio_center.lon.degree,
        helio_center.lat.degree,
        bbox_size.to(u.degree).value,
        n_samples
    )
    bbox_coords = coord.HeliocentricTrueEcliptic(
        lon=bbox[0]*u.degree,
        lat=bbox[1]*u.degree,
        distance=helio_center.distance
    )
    return bbox_coords


def geocentric_bbox_from_heliocentric(
        helio_center,
        obs_t,
        bbox_size=1*u.degree,
        n_samples=100
):
    """Create a bounding box in Geocentric True Ecliptic coordinate system by
    creating a bounding box around the given center coordinates in the
    Heliocentric True Ecliptic coordinates and then transforming them to
    geocentric true ecliptic coordinate system.

    Parameters
    ----------
    helio_center : `astropy.coordinates.HeliocentricTrueEcliptic`
        Coordinates of the center of the bounding box.
    obs_t : `astropy.time.Time`
        Time stamp when the bounding box was observed from Earth.
    bbox_size : `astropy.units.Quantity`
        Size of the bounding box side.
    n_samples : `int`
        Number of points to sample per side of the bounding box.

    Returns
    -------
    bbox : `astropy.coordinates.GeocentricTrueEcliptic`
        Bounding box in heliocentric true ecliptic coordinate system.
    """
    bbox_coords = make_heliocentric_bbox(helio_center, bbox_size, n_samples)
    geo_eclip = coord.GeocentricTrueEcliptic(obstime=obs_t)
    return bbox_coords.transform_to(geo_eclip)


def plot_bbox(
        ax,
        lon,
        lat,
        xlims=None,
        ylims=None,
        color=None,
        xlbl="",
        ylbl="",
        title="",
        aspect_ratio=None
):
    """Plots a bounding box on the given axis.

    The function expects the given coordinates and quantities are all
    transformed to the appropriate units.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axis`
        Axis to plot on
    lon : `array-like`
        Longitudes of points on the bounding box.
    lat : `array-like`
        Latitudes of points on the bounding box.
    xlims : `None` or `array-like`, optional
        When not `None` a length 2 array-like object setting the minimum and
        maximum limits of the x axis.
    ylims : `None` or `array-like`, optional
        When not `None` a length 2 array-like object setting the minimum and
        maximum limits of the y axis.
    color : `None` or `str`, optional
        Color of the points
    xlbl : `str`, optional
        Label of the x axis. Empty string by default.
    ylbl : `str`, optional
        Label of the y axis. Empty string by default.
    title : `str`, optional
        Plot title. Empty string by default.
    aspect_ratio : `None` or `str`
        Fix plot aspect ratio, one of the matplotlib's valid aspect ratio
        strings, f.e. `equal`

    Returns
    -------
    ax : `matplotlib.pyplot.Axis`
        Axis to plot on
    """
    ax.scatter(lon, lat, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)
    if aspect_ratio is not None:
        ax.axis(aspect_ratio)
    return ax


def plot_bbox_t_t180(
        helio_center,
        obs_t,
        bbox_size=1*u.degree,
        n_samples=100,
        xlims=None,
        ylims=None,
        figsize=(16, 4),
        aspect_ratio=None
):
    """Plots a a three pane figure displaying the heliocentric bounding box,
    geocentric bounding box at given time, and a geocentric bounding box at
    given time + 180 days.

    Parameters
    ----------
    helio_center : `astropy.coordinates.HeliocentricTrueEcliptic`
        Coordinates of the center of the bounding box.
    obs_t : `astropy.time.Time`
        Time stamp when the bounding box was observed from Earth.
    bbox_size : `astropy.units.Quantity`, optional
        Size of the bounding box side, 1x1 degrees by default
    n_samples : `int`, optional
        Number of points to sample per side of the bounding box, 100 by default
    xlims : `None` or `array-like`, optional
        When not `None` a length 2 array-like object setting the minimum and
        maximum limits of the x axis.
    ylims : `None` or `array-like`, optional
        When not `None` a length 2 array-like object setting the minimum and
        maximum limits of the y axis.
    figsize : `array-like`
        A length 2 array setting figure dimensions.
    aspect_ratio : `None` or `str`
        Fix plot aspect ratio, one of the matplotlib's valid aspect ratio
        strings, f.e. `equal`

    Returns
    -------
    fig : `matplotlib.pyplot.Figure`
        Figure holding the axes.
    ax : `matplotlib.pyplot.Axis`
        Axis to plot on
    """
    bbox_helio = make_heliocentric_bbox(
        helio_center,
        bbox_size,
        n_samples
    )

    geo_eclip1 = coord.GeocentricTrueEcliptic(obstime=obs_t)
    bbox_t = bbox_helio.transform_to(geo_eclip1)

    geo_eclip2 = coord.GeocentricTrueEcliptic(obstime=obs_t + 180*u.day)
    bbox_t180 = bbox_helio.transform_to(geo_eclip2)

    xlbl = "Longitude (degree)"
    ylbl = "Latitude (degree)"
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_bbox(
        axes[0],
        bbox_helio.lon,
        bbox_helio.lat,
        xlims=xlims,
        ylims=ylims,
        color="red",
        xlbl=xlbl,
        ylbl=ylbl,
        title="Heliocentric True Ecliptic"
    )
    plot_bbox(
        axes[1],
        bbox_t.lon,
        bbox_t.lat,
        xlims=xlims,
        ylims=ylims,
        color="blue",
        xlbl=xlbl,
        ylbl=ylbl,
        title="Geocentric True Ecliptic at t"
    )
    plot_bbox(axes[2],
              bbox_t180.lon,
              bbox_t180.lat,
              xlims=xlims,
              ylims=ylims,
              color="blue",
              xlbl=xlbl,
              ylbl=ylbl,
              title="Geocentric True Ecliptic at t+180 days"
    )

    return fig, axes


def coord_bbox_to_array(bbox, cast_units="degree"):
    """Casts given bounding box in a coordinate system to an array of floats.

    Parameters
    ----------
    bbox : `astropy.coordinates.BaseCoordinateFrame`
        Instance of one of Astropy's coordinate frames, must contain `lon` and
        `lat` attributes.
    cast_units : `str`
        One of Astropy's recognized units, f.e. `degree`, `arcsecond` etc.

    Returns
    -------
    bbox_cast : `numpy.array`
        Bounding box coordinate values as floats, cast in the given units.
    """
    if isinstance(bbox, coord.BaseCoordinateFrame):
        return np.array(
            [getattr(bbox.lon, cast_units), getattr(bbox.lat, cast_units)]
        )
    return bbox


def fit_affine(bbox1, bbox2, cast_units="degree", **kwargs):
    """Fits an affine transform (scale, rotation, shear and translation) that
    transforms bbox1 to bbox2.

    Parameters
    ----------
    bbox1 : `astropy.coordinates.BaseCoordinateFrame`
        Instance of one of Astropy's coordinate frames, must contain `lon` and
        `lat` attributes.
    bbox1 : `astropy.coordinates.BaseCoordinateFrame`
        Instance of one of Astropy's coordinate frames, must contain `lon` and
        `lat` attributes.
    cast_units : `str`
        One of Astropy's recognized units, f.e. `degree`, `arcsecond` etc.
    **kwargs : `dict`, optional
        Other keyword arguments are passed to the `AffineFitter.fit` method,
        basically to `scipy.optimize.minimize` function.

    Returns
    -------
    afft : `AffineFitter`
        The fitter.
    transformed_coords : `astropy.coordinates.GeocentricTrueEcliptic`
        Transformed coordinates in the geocentric true ecliptic frame.
    distances : `np.array`
        Distances between coordinates of the `bbox1` points and the transformed
        coordinates in arcseconds.
    """
    bbox1_arr = coord_bbox_to_array(bbox1, cast_units)
    bbox2_arr = coord_bbox_to_array(bbox2, cast_units)

    afft = AffineTransformFitter(bbox1_arr, bbox2_arr)
    fit = afft.fit(**kwargs)
    transformed = afft.transform()

    if isinstance(bbox1_arr, coord.GeocentricTrueEcliptic):
        dist2bbox = bbox1.distance
    else:
        dist2bbox = bbox2.distance

    # this can fail with ValueError when latitude angles are not within
    # -90deg <= angle <= 90 deg
    trans_coords = coord.GeocentricTrueEcliptic(
        lon=transformed[0]*getattr(u, cast_units),
        lat=transformed[1]*getattr(u, cast_units),
        distance=dist2bbox
    )

    # Afft transforms first set of points into the second. In this case bbox1
    # into bbox2, i.e. we look at distance between bbox2 and transformed
    distances = coord.angular_separation(
        bbox2.lon,
        bbox2.lat,
        trans_coords.lon,
        trans_coords.lat
    )
    distances = distances.to(u.arcsecond)

    return afft, trans_coords, distances


def plot_hist_distances(
        helio_center,
        obs_t,
        bbox_size=1*u.degree,
        n_samples=100,
        cast_units="degree",
        ax=None,
        bins=40,
        figsize=None,
):
    """Makes a bounding box in ecliptic coordinate frame around the given
    central coordinate, transforms it to geocentric coordinates, fits an affine
    transform between the two bounding boxes and plots them on the given
    axis.

    Parameters
    ----------
    helio_center : `astropy.coordinates.HeliocentricTrueEcliptic`
        Coordinates of the center of the bounding box.
    obs_t : `astropy.time.Time`
        Time stamp when the bounding box was observed from Earth.
    bbox_size : `astropy.units.Quantity`, optional
        Size of the bounding box side, 1x1 degrees by default
    n_samples : `int`, optional
        Number of points to sample per side of the bounding box, 100 by default
    cast_units : `str`, optional
        One of Astropy's recognized units, f.e. `degree`, `arcsecond` etc. The
        bounding box will be cast to these units before the fit is performed.
    ax : `matplotlib.pyplot.Axis` or `None`
        Axis to plot on. When `None` will create a new figure.
    bins : `int`, optional
        Number of bins in the histogram.
    figsize : `None` or `array-like`, optional
        A length 2 array setting figure dimensions, `None` by default.
    """
    bbox_helio = make_heliocentric_bbox(
        helio_center,
        bbox_size,
        n_samples
    )
    geo_eclip = coord.GeocentricTrueEcliptic(obstime=obs_t)
    bbox_geo = bbox_helio.transform_to(geo_eclip)

    afft, transformed, distances = fit_affine(bbox_helio, bbox_geo, cast_units)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.hist(distances.value, bins=bins)
    ax.set_title("Angular separation of Geocentric True Ecliptic at t and t+180 days")
    ax.set_xlabel("Angular separation (arcseconds)")

    return ax, afft, transformed, distances


