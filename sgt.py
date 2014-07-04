#!/usr/bin/python
# vim: set fileencoding=utf-8 :
"""
Simple good-turing estimation, as described in 

W. Gale. Good-Turing smoothing without tears. Journal of
Quantitative Linguistics, 2:217-37, 1995.
"""

from __future__ import division

import unittest
import collections
import copy
from math import log, exp, fsum
import numpy

from . import averaging_transform
from .memo import memoize


class Estimator(object):
    """
    The estimator created by SGT method.
    Slope and intercept define the linear estimate. The linear_cutoff parameter
    defines the r value strictly below which the estimate will be computed with
    the unsmoothed Turing estimate. N is the array of values N_r; N[0] should
    be equal to the sum of all N values thereafter.

    'b' is used as defined in the paper and is the log-log slope of N[r] with
    respect to r. It needs to be < -1 for SGT smoothing to be applicable.
    """
    def __init__(self, *args, **kwargs):
        """
        Create a Simple Good Turing estimator for the given N_r values (as
        specified in the 'N' kwarg). N[r] == N_r, so N[0] should be blank.

        The maximum r value is assumed to be len[N] unless specified otherwise
        with max_r. This enables use of defaultdict instead of list to specify
        N_r values.
        """
        self.N = copy.copy(kwargs.pop('N'))
        self.max_r = kwargs.pop('max_r', len(self.N))
        super(Estimator, self).__init__(*args, **kwargs)
        assert not self.N[0]
        self._precompute()

    def _precompute(self):
        """
        Do the necessary precomputation to compute estimates
        """
        N = self.N
        # Store 'N' as used by Gale in N[0]
        N[0] = sum(N[r] * r for r in range(1, self.max_r + 1))
        self.Z = Z = averaging_transform.transform(N, self.max_r)  # Z[r] = Z_r
        self.b, self.a = self._regress(Z)    # 'a' and 'b' as used in (Gale); a
                                             # is intercept and b is slope.
        # Find the transition point between linear Good-Turing estimate and the
        # Turing estimate.
        self.linear_cutoff = self._find_cutoff()
        self.norm_constant = self._find_norm_constant()

    def _find_norm_constant(self):
        N = self.N
        return ((1 - self.rstar_unnorm(0)) /
                fsum(N[r] * self.p_unnorm(r)
                     for r in range(1, self.max_r + 1)))

    def _regress(self, Z):
        """
        Perform linear regression on the given points in loglog space, return
        result
        """
        # Make a set of the nonempty points in log scale
        x, y = zip(*[(log(r), log(Z[r]))
                               for r in range(1, self.max_r + 1) if Z[r]])
        self.x, self.y = x, y
        matrix = numpy.array((x, numpy.ones(len(x)))).T
        return numpy.linalg.lstsq(matrix, y)[0]

    def _find_cutoff(self):
        """
        Find first r value s.t. the linear and turing estimates of r* are not
        significantly different.
        """
        cutoff = 1
        while ((self.linear_rstar_unnorm(cutoff) -
                self.turing_rstar_unnorm(cutoff))**2
                > self.approx_turing_variance(cutoff)):
            cutoff += 1
        return cutoff

    def approx_turing_variance(self, r):
        """
        Compute the approximate variance of the turing estimate for r* given r,
        using the approximation given in (Gale):

        var(r^{*}_T) ≈ (r + 1)^2 (N_{r+1}/N_r^2) (1 + N_{r+1}/N_r)
        """
        N = self.N
        return (r + 1)**2 * (N[r+1] / N[r]**2) * (1 + N[r+1] / N[r])

    def linear_rstar_unnorm(self, r):
        """
        Linear Good-Turing estimate of r* for given r:
             log(N_r) = a + b log(r)
        -->  N_r = A * r^b
        and  r* = (r + 1)(N_{r+1})/N_r
        -->  r* = (r + 1)(A * (r+1)^b)/(A * r^b)
                = (r+1)^{b+1) / r^b
                = r (1 + 1/r)^{b+1}
        """
        return r * (1 + 1/r)**(self.b + 1) if r > 0 else None

    def turing_rstar_unnorm(self, r):
        """
        simple Turing estimate of r* for given r (unsmoothed):
             r* = (r + 1)(N_{r+1})/N_r
        """
        return ((r + 1) * self.N[r + 1] / self.N[r]
                if self.N[r + 1] > 0 and self.N[r] > 0
                else None)

    @memoize(dict)
    def rstar_unnorm(self, r):
        return (self.linear_rstar_unnorm(r) if r >= self.linear_cutoff
                else self.turing_rstar_unnorm(r))

    @memoize(dict)
    def rstar(self, r):
        return (self.rstar_unnorm(0) if r == 0
                else self.rstar_unnorm(r) * self.norm_constant)

    def p_unnorm(self, r):
        return self.rstar_unnorm(r) / self.N[0]

    def p(self, r):
        return self.rstar(r) / self.N[0]


class ChinesePluralsTest(unittest.TestCase):
    max_r = 1918
    maxDiff = None
    input = collections.defaultdict(lambda:0)
    input.update([
        (1, 268),
        (2, 112),
        (3, 70),
        (4, 41),
        (5, 24),
        (6, 14),
        (7, 15),
        (8, 14),
        (9, 8),
        (10, 11),
        (11, 9),
        (12, 6),
        (13, 6),
        (14, 3),
        (15, 7),
        (16, 9),
        (17, 4),
        (18, 4),
        (19, 8),
        (20, 2),
        (21, 4),
        (22, 2),
        (23, 2),
        (24, 3),
        (25, 4),
        (26, 4),
        (27, 4),
        (28, 1),
        (29, 1),
        (31, 2),
        (33, 1),
        (39, 3),
        (41, 1),
        (46, 1),
        (47, 1),
        (50, 1),
        (52, 2),
        (53, 1),
        (55, 1),
        (57, 1),
        (60, 1),
        (74, 1),
        (84, 1),
        (108, 1),
        (109, 1),
        (177, 1),
        (400, 1),
        (1918, 1),
    ])
    output = copy.copy(input)
    output.update([
        (0, 0.04090978),
        (1, 0.8414893),
        (2, 1.887716),
        (3, 2.288452),
        (4, 3.247259),
        (5, 4.222094),
        (6, 5.206074),
        (7, 6.195765),
        (8, 7.189259),
        (9, 8.185414),
        (10, 9.183503),
        (11, 10.18304),
        (12, 11.1837),
        (13, 12.18523),
        (14, 13.18746),
        (15, 14.19026),
        (16, 15.19353),
        (17, 16.19719),
        (18, 17.20118),
        (19, 18.20545),
        (20, 19.20996),
        (21, 20.21467),
        (22, 21.21956),
        (23, 22.22462),
        (24, 23.22981),
        (25, 24.23512),
        (26, 25.24054),
        (27, 26.24606),
        (28, 27.25167),
        (29, 28.25735),
        (31, 30.26893),
        (33, 32.28073),
        (39, 38.31721),
        (41, 40.32964),
        (46, 45.36113),
        (47, 46.36749),
        (50, 49.38667),
        (52, 51.39953),
        (53, 52.40597),
        (55, 54.41891),
        (57, 56.43188),
        (60, 59.45142),
        (74, 73.54344),
        (84, 83.60977),
        (108, 107.7701),
        (109, 108.7768),
        (177, 177.2346),
        (400, 401.744),
        (1918, 1930.037),
    ])

    norm_constant = 1.006782
    a, b = (6.683387, -1.964591)

    def assertAlmostEqual(self, left, right, msg=None, places=3):
        if msg:
            msg = msg + (" (%r ≠ %r)" % (left, right))
        unittest.TestCase.assertAlmostEqual(
            self, left, right, msg=msg, places=places)

    def test_unnorm_output(self):
        estimator = Estimator(N=self.input, max_r=self.max_r)
        keys = sorted(self.output.keys())
        for key in keys :
            self.assertAlmostEqual(estimator.rstar_unnorm(key),
                                   self.output[key] / 
                                   (self.norm_constant if key > 0 else 1),
                                   msg=("%d* (unnormalized)" % (key,)))


    def test_output(self):
        estimator = Estimator(N=self.input, max_r=self.max_r)
        keys = sorted(self.output.keys())
        for key in keys:
            self.assertAlmostEqual(estimator.rstar(key),
                                   self.output[key],
                                   msg=("%d* (normalized)" % (key,)))

    def test_constant(self):
        estimator = Estimator(N=self.input, max_r=self.max_r)
        self.assertAlmostEqual(estimator.norm_constant,
                               self.norm_constant,
                               msg="Normalization constant")

    def test_linear(self):
        estimator = Estimator(N=self.input, max_r=self.max_r)
        self.assertAlmostEqual(estimator.a,
                               self.a,
                               places=6,
                               msg="Linear regression intercept")
        self.assertAlmostEqual(estimator.b,
                               self.b,
                               places=6,
                               msg="Linear regression slope")


class ProsodyTest(ChinesePluralsTest):
    max_r = 7846
    input = collections.defaultdict(lambda:0)
    input.update([
        (1, 120),
        (2, 40),
        (3, 24),
        (4, 13),
        (5, 15),
        (6, 5),
        (7, 11),
        (8, 2),
        (9, 2),
        (10, 1),
        (12, 3),
        (14, 2),
        (15, 1),
        (16, 1),
        (17, 3),
        (19, 1),
        (20, 3),
        (21, 2),
        (23, 3),
        (24, 3),
        (25, 3),
        (26, 2),
        (27, 2),
        (28, 1),
        (31, 2),
        (32, 2),
        (33, 1),
        (34, 2),
        (36, 2),
        (41, 3),
        (43, 1),
        (45, 3),
        (46, 1),
        (47, 1),
        (50, 1),
        (71, 1),
        (84, 1),
        (101, 1),
        (105, 1),
        (121, 1),
        (124, 1),
        (146, 1),
        (162, 1),
        (193, 1),
        (199, 1),
        (224, 1),
        (226, 1),
        (254, 1),
        (257, 1),
        (339, 1),
        (421, 1),
        (456, 1),
        (481, 1),
        (483, 1),
        (1140, 1),
        (1256, 1),
        (1322, 1),
        (1530, 1),
        (2131, 1),
        (2395, 1),
        (6925, 1),
        (7846, 1),
    ])
    output = copy.copy(input)
    output.update([
        (0, 0.003883244),
        (1, 0.7628079),
        (2, 1.706448),
        (3, 2.679796),
        (4, 3.663988),
        (5, 4.653366),
        (6, 5.645628),
        (7, 6.63966),
        (8, 7.634856),
        (9, 8.63086),
        (10, 9.627446),
        (12, 11.62182),
        (14, 13.61725),
        (15, 14.61524),
        (16, 15.61336),
        (17, 16.6116),
        (19, 18.60836),
        (20, 19.60685),
        (21, 20.6054),
        (23, 22.60264),
        (24, 23.60133),
        (25, 24.60005),
        (26, 25.5988),
        (27, 26.59759),
        (28, 27.59639),
        (31, 30.59294),
        (32, 31.59183),
        (33, 32.59073),
        (34, 33.58964),
        (36, 35.58751),
        (41, 40.58235),
        (43, 42.58035),
        (45, 44.57836),
        (46, 45.57738),
        (47, 46.57641),
        (50, 49.57351),
        (71, 70.55399),
        (84, 83.54229),
        (101, 100.5272),
        (105, 104.5237),
        (121, 120.5097),
        (124, 123.507),
        (146, 145.4879),
        (162, 161.474),
        (193, 192.4472),
        (199, 198.4421),
        (224, 223.4205),
        (226, 225.4188),
        (254, 253.3947),
        (257, 256.3922),
        (339, 338.3218),
        (421, 420.2514),
        (456, 455.2215),
        (481, 480.2),
        (483, 482.1983),
        (1140, 1138.636),
        (1256, 1254.537),
        (1322, 1320.48),
        (1530, 1528.302),
        (2131, 2128.788),
        (2395, 2392.562),
        (6925, 6918.687),
        (7846, 7838.899),
    ])
    norm_constant = 0.9991445
    a, b = (4.468558, -1.389374)
