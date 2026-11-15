import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_analyses import clustering, estimate, ranks


class AnalysisVerificationTests(unittest.TestCase):
    def test_rank_ties_receive_equal_average_ranks(self):
        np.testing.assert_array_equal(ranks(np.array([2., 1., 1., 3.])), [2., .5, .5, 3.])

    def test_weightings_preserve_their_different_populations(self):
        value = estimate(np.array([0., 0., 1.]), np.array(['A', 'A', 'B']))
        np.testing.assert_allclose(value[:, 0], [100/3, 50])

    def test_fractional_ties_are_preserved_in_the_sibling_oracle(self):
        value = clustering(np.array([0., .5, 1., 1.]), np.array(['A', 'A', 'B', 'B']), 2)
        self.assertEqual(value['baseline'], .625)
        self.assertEqual(value['sibling_oracle'], .75)
        self.assertEqual(value['oracle_gain'], .125)
        self.assertEqual(value['all_wrong_molecule_share'], 0.)
        self.assertEqual(value['mixed_molecule_share'], .5)
