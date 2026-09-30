import unittest
import numpy as np
from detector_pilot import response,migration,morph,response_profile_z,hist_components
from pilot import generate,significance

class DetectorChecks(unittest.TestCase):
    def test_shared_response_draws(self):
        latent=generate(4,100)
        a=response(latent,8);b=response(latent,8)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        up=response(latent,8,.01,1)
        np.testing.assert_allclose(up[1]-a[1],.01*latent[1])
    def test_response_does_not_read_labels(self):
        x,m,y=generate(4,100)
        a=response((x,m,y),8);b=response((x,m,1-y),8)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    def test_migration_conservation(self):
        y=np.array([0,0,0,1,1,1]);m=np.array([100,110,120,125,126,127])
        a=np.array([.2,.6,.8,.4,.7,.9]);b=np.array([.7,.3,.8,.6,.7,.2])
        for row in migration(m,m,y,a,b,.5):
            self.assertEqual(row['selected_varied']-row['selected_nominal'],row['entered']-row['exited'])
        for row in migration(m,m,y,a,a,.5):self.assertEqual(row['entered']+row['exited']+row['mass_bin_changed'],0)
    def test_endpoint_morphs(self):
        b=np.full(15,100.);down=b*.9;up=b*1.2
        np.testing.assert_allclose(morph(b,down,up,0),b)
        np.testing.assert_allclose(morph(b,down,up,-1),down)
        np.testing.assert_allclose(morph(b,down,up,1),up)
    def test_profile_identity_and_nested_bound(self):
        b=np.full(15,100.);s=np.zeros(15);s[7]=20
        z,_=response_profile_z(s,b,{'scale':(b,b),'resolution':(b,b)})
        self.assertAlmostEqual(z,significance(s,b,.15,0),places=5)
        self.assertLessEqual(z,significance(s,b)+1e-6)
