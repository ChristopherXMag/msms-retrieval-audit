import unittest
import numpy as np
from audit_metrics import paired_cluster_interval,summarize,uniform_ties


class MetricsTests(unittest.TestCase):
    def test_uniform_ties_cross_recall_cutoffs(self):
        out=uniform_ties(np.array([0,4,19]),np.array([2,3,2]))
        np.testing.assert_allclose(out[:,0],[.5,0,0])
        np.testing.assert_allclose(out[:,1],[1,1/3,0])
        np.testing.assert_allclose(out[:,2],[1,1,.5])
        np.testing.assert_allclose(out[0,3],.75)
        with self.assertRaises(ValueError):uniform_ties([0],[0])

    def test_group_balancing_changes_unequal_population_weight(self):
        v=np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0]],float)
        r=summarize(v,np.array(['a','a','b']))
        self.assertAlmostEqual(r['query_weighted']['recall@1'],200/3)
        self.assertAlmostEqual(r['connectivity_balanced']['recall@1'],50)

    def test_paired_cluster_constant_effect(self):
        before=np.full((3,4),.1);after=before+.2
        r=paired_cluster_interval(after,before,np.array(['a','a','b']),draws=64,seed=3)
        for kind in ('query_weighted_ci95_pp','connectivity_balanced_ci95_pp'):
            for interval in r[kind].values():np.testing.assert_allclose(interval,[20,20])


if __name__=='__main__':unittest.main()
