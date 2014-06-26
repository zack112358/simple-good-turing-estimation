"""
Implement the averaging transform for simple Good-Turing smoothing as described
in 

W. Gale. Good-Turing smoothing without tears. Journal of
Quantitative Linguistics, 2:217-37, 1995.
"""

from __future__ import division
import unittest
import collections
import functools
import copy

def transform(N, size=None):
    """
    (Nondestructively) transform the list of values by averaging them with
    neighboring zeros.

    If size is specified, only generate Z[r] for r in range(size).  Otherwise,
    size defaults to len(N).

    Identifiers are chosen to match those used by 

    W. Gale. Good-Turing smoothing without tears. Journal of
    Quantitative Linguistics, 2:217-37, 1995.
    """

    size = len(N) if size is None else size

    Z = copy.copy(N)

    for r in (r for r in range(size)[1:] if N[r] != 0):
        q = r - 1
        while q > 0 and N[q] == 0:
            q -= 1
        t = r + 1
        while t < size and N[t] == 0:
            t += 1

        # The paper doesn't really specify a way to handle the edge case where
        # there is no defined q or t, because this is the lowest or hightest R.
        # I am choosing to handle the lower case by setting q to 0 (which falls
        # out of the above code naturally) , and the higher case by assuming
        # that t/r likely equals r/q. This would be correct for the case where
        # the log-log slope is -1, which won't really be true, but it'll be an
        # okay guess.
        if t == size and q != 0:
            # import pdb; pdb.set_trace()
            t = r**2 / q

        Z[r] = N[r] * 2 / (t - q)

    return Z


class TrivialTransformTest(unittest.TestCase):
    input = [0, 1]
    output = [0, 1]
    def test(self):
        self.assertEqual(transform(self.input), self.output)


class MadeUpTransformTest(TrivialTransformTest):
    input = [
        0.0,
        300.0,
        100.0,
        50.0,
        25.0,
        12.0,
        6.0,
        3.0,
        2.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    ]
    output = [
        0.0,
        300.0,
        100.0,
        50.0,
        25.0,
        12.0,
        6.0,
        3.0,
        2.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.4,
        0.0,
        0.4,
        0.0,
        0.0,
        0.303030303030303,
    ]


class ChinesePluralTransformTest(unittest.TestCase):
    maxDiff=None
    def test(self):
        self.assertEqual(dict(transform(self.input, size=1919)),
                         dict(self.output))

    input = collections.defaultdict(lambda:0)
    input.update([
        (1, 268.0),
        (2, 112.0),
        (3, 70.0),
        (4, 41.0),
        (5, 24.0),
        (6, 14.0),
        (7, 15.0),
        (8, 14.0),
        (9, 8.0),
        (10, 11.0),
        (11, 9.0),
        (12, 6.0),
        (13, 6.0),
        (14, 3.0),
        (15, 7.0),
        (16, 9.0),
        (17, 4.0),
        (18, 4.0),
        (19, 8.0),
        (20, 2.0),
        (21, 4.0),
        (22, 2.0),
        (23, 2.0),
        (24, 3.0),
        (25, 4.0),
        (26, 4.0),
        (27, 4.0),
        (28, 1.0),
        (29, 1.0),
        (31, 2.0),
        (33, 1.0),
        (39, 3.0),
        (41, 1.0),
        (46, 1.0),
        (47, 1.0),
        (50, 1.0),
        (52, 2.0),
        (53, 1.0),
        (55, 1.0),
        (57, 1.0),
        (60, 1.0),
        (74, 1.0),
        (84, 1.0),
        (108, 1.0),
        (109, 1.0),
        (177, 1.0),
        (400, 1.0),
        (1918, 1.0),
    ])
    output = copy.copy(input)
    output.update(dict([
        (1, 268.0),
        (2, 112.0),
        (3, 70.0),
        (4, 41.0),
        (5, 24.0),
        (6, 14.0),
        (7, 15.0),
        (8, 14.0),
        (9, 8.0),
        (10, 11.0),
        (11, 9.0),
        (12, 6.0),
        (13, 6.0),
        (14, 3.0),
        (15, 7.0),
        (16, 9.0),
        (17, 4.0),
        (18, 4.0),
        (19, 8.0),
        (20, 2.0),
        (21, 4.0),
        (22, 2.0),
        (23, 2.0),
        (24, 3.0),
        (25, 4.0),
        (26, 4.0),
        (27, 4.0),
        (28, 1.0),
        (29, 0.6666666666666666),
        (31, 1.0),
        (33, .25),
        (39, .75),
        (41, 0.2857142857142857),
        (46, 1/3),
        (47, .5),
        (50, .4),
        (52, 4/3),
        (53, 2/3),
        (55, .5),
        (57, .4),
        (60, 2/17),
        (74, 2/24),
        (84, 2/34),
        (108, 2/25),
        (109, 0.028985507246376812),
        (177, 0.006872852233676976),
        (400, 2/(1918-177)),
        (1918, 0.00022735514351225047),
    ]))
