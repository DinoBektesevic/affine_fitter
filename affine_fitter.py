import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import astropy.coordinates as coord
import astropy.units as u

def transform_to_homogeneous_coordinates(points, w=1.0):
    r"""Transform a 2D set of points to homogeneous coordinates.

    Given by expression:

    ..math::

        (x_i, y_i) \rightarrow (x_i\w, y_i\w, w)

    Parameters
    ----------
    points : `array-like`
        Set of points. Has dimensions (2, N) where N is the number of points in
        the set; f.e. `[[x_0 ... x_N], [y_0 ... y_N]]`
    w : `float`
        Arbitrary reparameterization value.

    Returns
    -------
    homogeneous : `array-like`
        Set of points in homogeneous coordinate space. Has dimensions (3, N)
        where N is the number of points in the set; f.e.
        `[[x_0 ... x_N], [y_0 ... y_N], [w ... w]]`
    """
    homogeneous = np.ones((3, points.shape[-1]))
    homogeneous[0] = points[0]
    homogeneous[1] = points[1]
    return homogeneous


def transform_from_homogeneous_coordinates(homogeneous):
    r"""Transform a 3D set of points in homogeneous coordinates to 2D set of
    points.

    Given by expression

    ..math::
        (x_i, y_i, w) \rightarrow (x_i\w, y_i\w)

    Parameters
    ----------
    homogenous : `array-like`
        Set of points. Has dimensions (3, N) where N is the number of points in
        the set; f.e. `[[x_0 ... x_N], [y_0 ... y_N], [w ... w]]`

    Returns
    -------
    points : `array-like`
        Set of points with dimensions (2, N) where N is the number of points
        in the set; f.e. `[[x_0 ... x_N], [y_0 ... y_N]]`
    """
    points = np.zeros((2, homogeneous.shape[-1]))
    points[0] = homogeneous[0]/homogeneous[-1]
    points[1] = homogeneous[1]/homogeneous[-1]
    return points


class AffineTransformFitter:
    r"""Find the best fitting affine transform between two sets of points.

    Given two sets of points finds the affine transform that maps the one onto
    the other.

    ..math::

        A' &= TA

        \begin{bmatrix}
            x' \\
            y' \\
            z'
        \end{bmatrix} &=
        \begin{bmatrix}
            a & b & c \\
            d & e & f \\
            0 & 0 & 1
        \end{bmatrix}
        \begin{bmatrix}
            x \\
            y \\
            z
        \end{bmatrix}

    Affine transformation is performed in homogeneous coordinate
    space and can account for translation.

    Parameters
    ----------
    points1 : `array-like` or `None`, optional
        Set of points ``{[x_0 ... x_n], [y_0 ... y_n]}`` that is the target of
        the transformation.
    points2 : `array-like` or `None`, optional
        Set of points ``{[x'_0 ... x'_n], [y'_0 ... y'_n]}`` that will be
        transformed to points1, target, of the fitter.

    Attributes
    ----------
    points1 : `array-like` or `None`, optional
        Set of points ``{[x_0 ... x_n], [y_0 ... y_n]}`` that is the target of
        the transformation.
    points2 : `array-like` or `None`, optional
        Set of points ``{[x'_0 ... x'_n], [y'_0 ... y'_n]}`` that will be
        transformed to points2.
    bestfit : `scipy.optimize.OptimizeResult` or `None`
        Stores the minimizers result, once the fit method has been called.
        `None` otherwise. Evaluated when `fit` is called.
    affineT : `array-like`
        A 3 dimensional square matrix that represents the affine transform of
        `points1` to `points2` in the heterogeneous coordinate space. Evaluated
        when call gto `fit` is made.
    invAffineT : `array-like`
       A 3 dimensional square matrix that represents the inverse affine
       transform, i.e. the matrix that transforms `points2` to `points1`.
       Evaluated only if `transform` is called with `inverse=True`.

    Notes
    -----
    Class finds the affine transform that maps `points1` onto `points2`, so
    that multiplying `affineT` with `points1` results in `points2`.
    """
    def __init__(self, points1=None, points2=None):
        self.points1 = np.array(points1)
        self.points2 = np.array(points2)
        self.bestfit = None
        self.affineT = None
        self.invAffineT = None
        self.__costf_map = {
            "L1": self.norm_L1,
            "L2": self.norm_L2,
            "ang_sep": self.ang_sep,
        }

    def norm_L1(self, points1, points2):
        """Sum of absolute-valued element-wise differences of two sets of
        points, i.e. the L1 norm of vector difference.

        ..math::

            \sum |x'_i - x_i|

        Returns
        -------
        L1_norm : `float`
            Sum of differences between two sets of points
        """
        return abs(points2 - points1).sum()

    def norm_L2(self, points1, points2):
        """Square roof of the sum of squared differences of vector elements,
        i.e. Ruclidean distance between two vectors

        ..math::

            || x' - x ||

        Returns
        -------
        L2_norm : `float`
            Euclidean distance of the given points
        """
        return np.linalg.norm(points2 - points1)

    def ang_sep(self, points1, points2):
        """Square roof of the sum of squared differences of vector elements,
        i.e. Ruclidean distance between two vectors

        ..math::

            || x' - x ||

        Returns
        -------
        L2_norm : `float`
            Euclidean distance of the given points
        """
        return coord.angular_separation(
            (points1[0]*u.degree).to(u.radian),
            (points1[1]*u.degree).to(u.radian),
            (points2[0]*u.degree).to(u.radian),
            (points2[1]*u.degree).to(u.radian)
        ).sum()

    def fit(
            self,
            points1=None,
            points2=None,
            costf = "L2",
            method="SLSQP",
            x0=(1, 0, 0, 0, 1, 0),
            **kwargs
    ):
        """Finds the best fit affine transform between the given point sets.

        The fit solves for the minimum of given cost function. By default that
        is the minimal absolute difference between given `points1` and the
        affine transformation of `points2`:

        ..math::

            \min(A' - TA)
            \min(\sum_{i,j} | a_i' - t^i_j a^j_k |)

        Parameters
        ----------
        points1 : `array-like` or `None`, optional
            Set of points ``{[x_0 ... x_n], [y_0 ... y_n]}`` that will be
            transformed to ``points2``.
        points2 : `array-like` or `None`, optional
            Set of points ``{[x'_0 ... x'_n], [y'_0 ... y'_n]}``, the target
            of the transform.
        costf : `str` or callable
            Cost function that will be evaluated at each minimization iteration.
            By default `L2`. Available: `L1` and `L2`.
        method : `str`, optional
            One of `scipy.optimize.minimize` minimizer methods. Default: SLSQP
        x0 : `array-like`, optional
            Initial guesses for the elements of the affine transformation
            matrix. By default `[1, 0, 0, 0, 1, 0]` so that the initial affine
            transformation matrix is an identity matrix.
        **kwargs : `dict`, optional
            Passed onto the `scipy.optimize.minimize` function.

        Raises
        ------
        ValueError : raised when two sets of points were not provided.

        Note
        ----
        Both `points1` and `points2` arguments must be provided when calling
        this method or at initialization time. Error is raised otherwise.
        """
        if self.points1 is not None:
            points1 = self.points1
        if self.points2 is not None:
            points2 = self.points2

        if isinstance(costf, str):
            costf = self.__costf_map[costf]

        # to homogeneous coordinates
        hpoints1 = transform_to_homogeneous_coordinates(points1)
        hpoints2 = transform_to_homogeneous_coordinates(points2)

        def objective(x):
            a, b, c, d, e, f = x
            affineT = np.array([
                [a, b, c],
                [d, e, f],
                [0, 0, 1]
            ], dtype=float)
            guess = affineT @ hpoints1
            return costf(hpoints2, guess)

        # perform the fit
        res = minimize(objective, x0, method=method, **kwargs)

        # store crucial results
        self.bestfit = res
        self.affineT = np.array([
            [res.x[0], res.x[1], res.x[2]],
            [res.x[3], res.x[4], res.x[5]],
            [0,        0,        1       ]
        ], dtype=float)

        return res

    def transform(self, points=None, inverse=False):
        """Transform set of points using the fitted affine transformation.

        Parameters
        ----------
        points : `None` or `array-like`, optional
            Points to transform. Assumes `self.points1` if `None`. If
            `inverse=True` assumes `self.points2`
        inverse : `bool`, optional
            Use fitted affine transform or its inverse. Default is `False` for
            fitted affine transform.

        Returns
        -------
        transformed : `numpy.array`
            Transformed points, matrix multiplication of the transform matrix
            `T` and `points`.

        Note
        ----
        The inverse of the affine transform matrix will be computed and stored
        the first time the method is called with `inverse=True`. This can
        potentially fail.
        """
        if self.affineT is None:
            raise RuntimeError("Run fit method first!")

        if points is None:
            points = self.points1
        else:
            points = np.array(points)

        if inverse:
            if self.invAffineT is None:
                self.invAffineT = np.linalg.inv(self.affineT)
            T = self.invAffineT
            points = self.points2
        else:
            T = self.affineT

        homp = transform_to_homogeneous_coordinates(points)
        htransformed =T @ homp
        return transform_from_homogeneous_coordinates(htransformed)

    def __str__(self):
        if self.bestfit is None:
            return f"{self.__class__.__name__}('Call `fit` method to see results.')"
        return f"{self.__class__.__name__}(\n{self.affineT})"


############################################################
#                           Example
############################################################
if __name__ == "__main__":
    np.random.seed(0)

    square1 = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ], dtype=float)

    square2= np.array([
        [1, 1.5, 2.5, 2],
        [1,   2,   2, 1]
    ], dtype=float)

    afft = AffineTransformFitter(square1, square2)
    print(afft.fit())
    print(afft)
    # transformed = afft.transform(square2, inverse=True)
    transformed = afft.transform()

    fig, ax = plt.subplots()
    ax.plot(afft.points1[0], afft.points1[1], color="red")
    ax.plot(afft.points2[0], afft.points2[1], color="blue")
    ax.plot(transformed[0], transformed[1], color="green")
    plt.show()
