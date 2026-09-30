import unittest
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from pilot import significance,generate,BINS

class PhysicsChecks(unittest.TestCase):
    def test_no_signal(self):
        for norm,shape in [(0,0),(.15,0),(.15,.35)]:
            self.assertAlmostEqual(significance(np.zeros(15),np.full(15,100.),norm,shape),0,places=6)
    def test_known_background(self):
        s=np.full(15,5.);b=np.full(15,100.)
        expected=np.sqrt(2*15*(105*np.log(1.05)-5))
        self.assertAlmostEqual(significance(s,b),expected,places=9)
    def test_normalization_against_independent_scalar_solution(self):
        s=np.linspace(1,12,15);b=np.linspace(80,120,15);delta=.15
        n=s+b;k=np.log1p(delta)
        theta=brentq(lambda t:k*(b.sum()*np.exp(k*t)-n.sum())+t,-10,10)
        q=b*np.exp(k*theta)
        z=np.sqrt(2*np.sum(q-n+xlogy(n,n/q))+theta**2)
        self.assertAlmostEqual(significance(s,b,delta,0),z,places=6)
    def test_nuisance_cannot_improve_nominal_sensitivity(self):
        s=np.zeros(15);s[6:9]=30;b=np.full(15,100.)
        self.assertLessEqual(significance(s,b,.15,.35),significance(s,b)+1e-7)
    def test_zero_background_rejected(self):
        with self.assertRaises(ValueError): significance(np.ones(15),np.zeros(15))
    def test_generation_reproducibility_and_split_independence(self):
        a=generate(12,100);b=generate(12,100);other=generate(13,100)
        np.testing.assert_array_equal(a[0],b[0]);self.assertFalse(np.array_equal(a[0],other[0]))
        self.assertEqual(a[0].shape,(200,4));self.assertEqual(len(BINS),16)

if __name__=='__main__': unittest.main()
