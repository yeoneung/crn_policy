import unittest
import numpy as np
from threadpoolctl import threadpool_limits
import coupled_support_core as c
from coupling_sensitivity import FrozenSphereFamily,DS,D0,simulate


class GeometryChecks(unittest.TestCase):
    def setUp(self): threadpool_limits(limits=1)

    def family(self,n=4):
        rng=np.random.default_rng(990001)
        return FrozenSphereFamily(rng.normal(size=(2*n+3)*n+2),n)

    def test_preserves_radius_and_global_chord_constraint(self):
        env=c.StablePlant(4,9100);fam=self.family()
        rng=np.random.default_rng(43)
        x=rng.normal(size=(250,4))*np.geomspace(1e-8,10,250)[:,None]
        x[0]=0
        for d in DS:
            a,_=fam.action(x,np.full(len(x),20),env.baseline,np.full(len(x),d))
            b=env.baseline(x)
            self.assertLessEqual(np.max(np.linalg.norm(a-b,axis=1)),d+1e-10)
            np.testing.assert_allclose(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),atol=1e-10)

    def test_reference_recovers_exact_frozen_policy(self):
        env=c.StablePlant(4,9101);fam=self.family();x=np.random.default_rng(9).normal(size=(100,4));rem=np.arange(100)%40+1
        a,_=fam.action(x,rem,env.baseline,np.full(len(x),D0))
        np.testing.assert_array_equal(a,fam.original.action(x,rem,env.baseline))

    def test_no_repair_stays_no_repair(self):
        env=c.StablePlant(4,9102);fam=FrozenSphereFamily(np.zeros(46),4)
        x=np.random.default_rng(11).normal(size=(100,4));rem=np.full(len(x),20)
        for d in DS:
            a,_=fam.action(x,rem,env.baseline,np.full(len(x),d))
            np.testing.assert_array_equal(a,env.baseline(x))

    def test_direction_preserved_and_chord_scaled_when_uncapped(self):
        env=c.StablePlant(4,9103);fam=self.family();x=np.random.default_rng(12).normal(size=(100,4));rem=np.full(len(x),20)
        b=env.baseline(x);u=c.unit(b);ref=fam.original.action(x,rem,env.baseline)
        tangent0=ref-u*np.sum(u*ref,axis=1)[:,None]
        for d in [.0125,.025,.1,.2]:
            a,cap=fam.action(x,rem,env.baseline,np.full(len(x),d))
            tangent=a-u*np.sum(u*a,axis=1)[:,None]
            good=(np.linalg.norm(tangent0,axis=1)>1e-8)&(~cap)
            np.testing.assert_allclose(c.unit(tangent[good]),c.unit(tangent0[good]),atol=1e-8)
            np.testing.assert_allclose(np.linalg.norm(a-b,axis=1)[good],np.linalg.norm(ref-b,axis=1)[good]*d/D0,atol=1e-10)

    def test_baseline_crn_stream_and_reference_exactly_match_original_evaluator(self):
        env=c.StablePlant(4,9104);fam=self.family();got=simulate(env,fam,64,991000,batch=32)
        ref=c.evaluate(env,[None,fam.original],64,c.gen(991000,env.n,env.seed),batch=32)
        np.testing.assert_allclose(got['returns'][[0,3]],ref['returns'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(got['reference_state_distances'].max(axis=0),ref['max_state_distance'][1],atol=1e-12)

    def test_independent_no_repair_has_noise_while_crn_is_zero(self):
        env=c.StablePlant(4,9105);fam=FrozenSphereFamily(np.zeros(46),4)
        got=simulate(env,fam,512,991100)
        np.testing.assert_allclose(got['returns'][3],got['returns'][0],atol=1e-12)
        self.assertGreater(np.var(got['returns'][6]-got['returns'][0]),.001)


if __name__=='__main__':unittest.main()
