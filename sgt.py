#!/usr/bin/python
# vim: set fileencoding=utf-8 :
"""
Simple good-turing estimation, as described in 

W. Gale. Good-Turing smoothing without tears. Journal of
Quantitative Linguistics, 2:217-37, 1995.
"""

import unittest
from math import log, exp
import numpy
from __future__ import division

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
        self.N = kwargs.pop('N')
        self.max_r = kwargs.pop('max_r', len(self.N))
        super(Estimator, self).__init__(*args, **kwargs)
        assert not self.N[0]
        self._precompute()

    def _precompute():
        """
        Do the necessary precomputation to compute estimates
        """
        self.sum_N = sum(self.N[r] for r in range(1, max_r + 1))
        self.Z = averaging_transform.transform(N, max_r)   # Z[r] = Z_r
        self.a, self.b = self._regress(Z)   # 'a' and 'b' as used in the paper
        self.linear_cutoff = self._find_cutoff()

    def _regress(self, Z):
        # Make a set of the nonempty points in log scale
        self.x, self.y = unzip(log(r), log(Z[r])
                               for r in range(1, max_r + 1) if Z[r])
        matrix = numpy.array((x, numpy.ones(len(x)))).T
        return numpy.linalg.lstsq(matrix, y)[0]

    def _find_cutoff(self):
        """
        Find first r value s.t. the linear and turing estimates of r* are not
        significantly different.
        """
        cutoff = 1
        while ((self.linear_estimate(cutoff) -
                self.turing_estimate(cutoff))**2
                > self._approx_turing_variance(cutoff)):
            cutoff += 1
        return cutoff

    def _approx_turing_variance(r):
        """
        Compute the approximate variance of the turing estimate for r* given r,
        using the approximation given in (Gale):

        var(r^{*}_T) ≈ (r + 1)^2 (N_{r+1}/N_r^2) (1 + N_{r+1}/N_r)
        """
        return (r + 1)**2 * (N[r+1] / N[r]**2) * (1 + N[r+1] / N[r])

    @memoize(dict)
    def linear_estimate(self, r):
        """
        Linear Good-Turing estimate of r* for given r:
             log(N_r) = a + b log(r)
        -->  N_r = A * r^b
        and  r* = (r + 1)(N_{r+1})/N_r
        -->  r* = (r + 1)(A * (r+1)^b)/(A * r^b)
                = (r+1)^{b+1) / r^b
                = r (1 + 1/r)^{b+1}
        """
        return r * (1 + 1/r)**(self.b + 1)

    @memoize(dict)
    def turing_estimate(self, r):
        """
        simple Turing estimate of r* for given r (unsmoothed):
             r* = (r + 1)(N_{r+1})/N_r
        """
        return (r + 1) * N[r + 1] / N[r]

    @memoize(dict)
    def estimate(self, r):
        return (self.linear_estimate(r)
                if r >= self.linear_cutoff
                else self.turing_estimate(r))

