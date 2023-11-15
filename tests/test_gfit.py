import unittest

import numpy as np
from gfit import initialise, evaluate, gfit, refine
from gfit.internal import fit_mgauss, fit_amgauss, amgauss, mgauss
from gfit.util import split_coeff, stack_coeff, rand_signal

class MyTestCase(unittest.TestCase):


    def test_01_forward(self):
        x = np.linspace(-10,10)
        y = np.zeros_like(x)

        # run symmetric gaussian
        mgauss(x,y, np.array([1,2,]), np.array([-2,2.5]), np.array([5,5] ) )
        self.assertTrue( (y >= 0).all() )

        # run asymmetric gaussian
        amgauss(x, y, np.array([1, 2, ]), np.array([-2, 2.5]), np.array([5, 5]), np.array([1, 1]) )
        self.assertTrue((y >= 0).all())

        # test batch evaluation
        M = np.array( [ [1., -2., 5., 2., 2.5, 5. ] for i in range(100)] )
        Y = evaluate( x, M, sym=True )
        self.assertTrue( M.shape[0] == Y.shape[0] ) # check shape of output is sensible
        self.assertTrue( Y.shape[-1] == x.shape[0] )
        self.assertTrue( np.max( Y ) > 1. )

        M = np.array( [ [ np.array([1., -2., 5., 10.,  2., 2.5, 5., 2.]) for i in range(100)] for n in range(10) ] )
        Y = evaluate(x, M, sym=False)
        self.assertTrue(M.shape[0] == Y.shape[0])  # check shape of output is sensible
        self.assertTrue(M.shape[1] == Y.shape[1])  # check shape of output is sensible
        self.assertTrue(Y.shape[-1] == x.shape[0])
        self.assertTrue(np.max(Y) > 1.)

    def test_02_backward_asym(self):
        # generate test signal
        x = np.linspace(-10, 10)
        y = np.zeros_like(x)
        a = np.array([2., 1.])
        b = np.array([-5., 2.5])
        c1 = np.array([3., 2.5])
        c2 = np.array([1., 2.])
        amgauss(x, y, a, b, c1, c2)  # evaluate model

        # generate initial guess
        n = 2
        x0 = initialise(x, np.array([y]), n, sym=False, d=4)  # compute initial values

        # fit it
        c = np.array(([0., -11., 0, 0] * n, [11., 11., np.inf, np.inf] * n))
        fit = fit_amgauss(x, y, x0[0, :], n, c=c, verbose=False, ftol=1e-6, xtol=1e-6)
        fa, fb, fc1, fc2 = split_coeff(fit)

        # sort results by position and check they're similar
        idxA = np.argsort(b)  # first three features return by rand_signal are the deep ones
        idxB = np.argsort(fb)

        for i in range(n):
            #print(a[idxA[i]], fa[idxB[i]])
            #print(b[idxA[i]], fb[idxB[i]])
            #print(c1[idxA[i]], fc1[idxB[i]])
            #print(c2[idxA[i]], fc2[idxB[i]])
            self.assertTrue( np.abs(a[idxA[i]] - fa[idxB[i]]) < 0.1 )
            self.assertTrue( np.abs(b[idxA[i]] - fb[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c1[idxA[i]] - fc1[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c2[idxA[i]] - fc2[idxB[i]]) < 0.1 )

    def test_03_backward_sym(self):
        # generate test signal
        x = np.linspace(-10, 10)
        y = np.zeros_like(x)
        a = np.array([2., 1.])
        b = np.array([-5., 2.5])
        c1 = np.array([3., 2.5])
        mgauss(x, y, a, b, c1)  # evaluate model

        # generate initial guess
        n = 2
        x0 = initialise(x, np.array([y]), n, sym=True, d=4)  # compute initial values

        # fit it
        c = np.array(([0., -11., 0] * n, [11., np.inf, np.inf] * n))
        fit = fit_mgauss(x, y, x0[0, :], n, c=c, verbose=False, ftol=1e-6, xtol=1e-6)
        fa, fb, fc1 = split_coeff(fit, sym=True)

        # sort results by position and check they're similar
        idxA = np.argsort(b)  # first three features return by rand_signal are the deep ones
        idxB = np.argsort(fb)

        for i in range(n):
            #print(a[idxA[i]], fa[idxB[i]])
            #print(b[idxA[i]], fb[idxB[i]])
            #print(c1[idxA[i]], fc1[idxB[i]])

            self.assertTrue( np.abs(a[idxA[i]] - fa[idxB[i]]) < 0.1 )
            self.assertTrue( np.abs(b[idxA[i]] - fb[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c1[idxA[i]] - fc1[idxB[i]]) < 0.1 )

    def test_04_splitting(self):
        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(10)])  # create random array
        x0 = initialise(x, X, 3, sym=False, d=4)  # compute initial values

        # check that split and stack functions work for assymetric functions
        a, b, c1, c2 = split_coeff(x0, sym=False)
        x1 = stack_coeff(a, b, c1, c2)

        assert (x0 == x1).all(), "Error - 2D stacking or splitting doesn't work"
        assert (stack_coeff(*split_coeff(x0[0, :])) == x0[0, :]).all(), "Error - 1D stacking or splitting doesn't work"

        # check that split and stack functions work for symmetric functions
        x0 = initialise(x, X, 3, sym=True, d=4)  # compute initial values
        a, b, c = split_coeff(x0, sym=True)
        x1 = stack_coeff(a, b, c)
        assert (x0 == x1).all(), "Error - 2D stacking or splitting doesn't work"

    def test_05_gfit_single_sym(self):
        from gfit import gfit

        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=True, d=4)  # compute initial values
        F = gfit(x, X, x0, 3, sym=True, nthreads=1, vb=True) # run optimisation

    def test_06_gfit_single_asym(self):
        from gfit import gfit

        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=False, d=4)  # compute initial values
        F = gfit(x, X, x0, 3, sym=False, nthreads=1, vb=True) # run optimisation

    def test_07_gfit_multi_sym(self):
        from gfit import gfit

        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=True, d=4)  # compute initial values
        F = gfit(x, X, x0, 3, sym=True, nthreads=-1, vb=True)  # run optimisation

    def test_08_gfit_multi_asym(self):
        from gfit import gfit

        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=False, d=4)  # compute initial values
        F = gfit(x, X, x0, 3, sym=False, nthreads=-1, vb=True)  # run optimisation

    def test_09_hull(self):
        from gfit.util import remove_hull
        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array
        X = np.abs(X) # remove negatives as this breaks the whole logic of a hull correction

        # divide upper hull (should result in all values  <= 1)
        Xh = remove_hull( X, upper=True )
        self.assertTrue( np.max(Xh) <= 1.0 )

        # divide lower hull (should result in all values  >= 1)
        Xh = remove_hull(X, upper=False)
        self.assertTrue(np.min(Xh) >= 1.0)

        # subtract upper hull (should result in all values  <= 0)
        Xh = remove_hull(X, upper=True, div=False)
        self.assertTrue(np.max(Xh) <= 0.0)

        # subtract lower hull (should result in all values  >= 0)
        Xh = remove_hull(X, upper=False, div=False)
        self.assertTrue(np.max(Xh) >= 0.0)

        # test on tricky dataset
        y = [0.1239837, 0.1, 0.20490336, 0.14806974, 0.31640035, 0.25220016, 0.32135254, 0.31456614, 0.40553832, 0.37361073, 0.39339942, 0.43066925, 0.4069156, 0.4247079, 0.45245135, 0.46885866, 0.48674428, 0.48791993, 0.5260081, 0.5245605, 0.51537675, 0.54603475, 0.55500835, 0.5740631, 0.56682056, 0.57287896, 0.5824296, 0.57922035, 0.59330124, 0.5860449, 0.5997907, 0.6002794, 0.60646814, 0.6128575, 0.61719114, 0.62185663, 0.62157637, 0.6234595, 0.6226736, 0.6325355, 0.6279977, 0.62992775, 0.63424915, 0.63554955, 0.63604695, 0.64096737, 0.6367344, 0.63740903, 0.64112204, 0.63816667, 0.6391656, 0.63690025, 0.6385414, 0.6381255, 0.6359338, 0.6370522, 0.6383992, 0.6390894, 0.63658357, 0.6359382, 0.6343395, 0.63758564, 0.6346627, 0.63511676, 0.63576424, 0.6359313, 0.6353944, 0.6379161, 0.6405646, 0.64042956, 0.6417708, 0.6439112, 0.6453111, 0.6467698, 0.6476673, 0.6501449, 0.6508637, 0.65072024, 0.6526123, 0.65590256, 0.65646774, 0.65446895, 0.6584839, 0.65691423, 0.66074634, 0.6606266, 0.6597198, 0.6606694, 0.66148394, 0.6632531, 0.6649953, 0.66756576, 0.668348, 0.6709612, 0.6708359, 0.6719639, 0.67367834, 0.6753634, 0.67567265, 0.672398, 0.6681845, 0.6779837, 0.68129355, 0.6819616, 0.6808597, 0.6811762, 0.6824197, 0.6824909, 0.68302137, 0.6835325, 0.6850223, 0.6870871, 0.68295497, 0.6854926, 0.6847698, 0.6848163, 0.6796797, 0.6811815, 0.6784403, 0.6799515, 0.6819219, 0.67809504, 0.67774135, 0.6753158, 0.6763439, 0.6714739, 0.66670376, 0.66939765, 0.6670862, 0.66759217, 0.6670961, 0.66295534, 0.6608094, 0.6606232, 0.6592001, 0.658899, 0.65656465, 0.6572743, 0.655487, 0.6554295, 0.65490097, 0.65232563, 0.6523706, 0.65305597, 0.6521162, 0.6528243, 0.65339154, 0.6516251, 0.6551416, 0.6539611, 0.6565033, 0.6556545, 0.6531398, 0.6569324, 0.65636593, 0.6564284, 0.6578926, 0.66152304, 0.659339, 0.6619823, 0.6626976, 0.66249555, 0.6641095, 0.6673027, 0.6687721, 0.66166776, 0.66415054, 0.6672054, 0.663602, 0.6642634, 0.662708, 0.6617951, 0.6624955, 0.6691956, 0.63417834, 0.6643856, 0.66675204, 0.66994125, 0.6711732, 0.6704542, 0.66920006, 0.6700372, 0.66996485, 0.67020035, 0.66937405, 0.66724926, 0.66956705, 0.6689852, 0.6699607, 0.6723215, 0.6707719, 0.67158103, 0.67164904, 0.67259747, 0.6730639, 0.6743521, 0.67493427, 0.67799324, 0.678164, 0.67811733, 0.67341685, 0.6806647, 0.68299437, 0.6850762, 0.687522, 0.68838686, 0.69101757, 0.6942636, 0.695708, 0.69859916, 0.7006682, 0.7062822, 0.7105276, 0.71442586, 0.72049, 0.7262016, 0.7299139, 0.7371605, 0.7401443, 0.7443576, 0.74625623, 0.7487325, 0.7529951, 0.7533893, 0.75447005, 0.75555843, 0.7604399, 0.76494193, 0.7648293, 0.76959735, 0.7702201, 0.7735309, 0.776223, 0.78068763, 0.7837493, 0.78792614, 0.7903506, 0.79405475, 0.7969008, 0.8005414, 0.803647, 0.8035068, 0.80074054, 0.7990194, 0.79925394, 0.79531044, 0.778262, 0.7415418, 0.6766072, 0.61712134, 0.57309663, 0.5492524, 0.557204, 0.61647147, 0.68787235, 0.736258, 0.75022537, 0.7579229, 0.76160413, 0.7698144, 0.7772672, 0.781874, 0.78493434, 0.793581, 0.7965664, 0.80047065, 0.79982895, 0.79870105, 0.7965914, 0.7938431, 0.7914435, 0.78996074, 0.7909289, 0.7924568, 0.79067487, 0.7903498, 0.79182345, 0.7929495, 0.7958183, 0.79922104, 0.8012536, 0.8052773, 0.8111827, 0.81539327, 0.81902224, 0.8236818, 0.8273287, 0.8291526, 0.83546793, 0.8361641, 0.8417408, 0.84319586, 0.84802115, 0.85046464, 0.8521055, 0.853567, 0.85650444, 0.85889757, 0.85876304, 0.86312383, 0.86307865, 0.86297244, 0.8638952, 0.86594635, 0.8675197, 0.8700436, 0.869609, 0.8712807, 0.8712113, 0.8720377, 0.87205213, 0.8735412, 0.8738168, 0.8739564, 0.8718329, 0.87287617, 0.8709547, 0.86982316, 0.8668764, 0.8669053, 0.8668702, 0.8602684, 0.8610498, 0.8580871, 0.85431874, 0.846709, 0.84202063, 0.8354382, 0.82940173, 0.8188476, 0.81285304, 0.8067151, 0.7993155, 0.7915049, 0.78116065, 0.77147764, 0.7585811, 0.74997044, 0.73259, 0.7130727, 0.69762045, 0.69062036, 0.68994194, 0.6905525, 0.69115466, 0.6891177, 0.6812801, 0.678976, 0.67653245, 0.67338145, 0.6692361, 0.6652579, 0.65969497, 0.6509336, 0.64422745, 0.64022106, 0.6357146, 0.63439286, 0.6344535, 0.63741946, 0.64155215, 0.6488119, 0.65522486, 0.6620279, 0.6655814, 0.667128, 0.66651195, 0.6658018, 0.6643455, 0.6626653, 0.6594522, 0.6532271, 0.6447912, 0.634679, 0.62214285, 0.6056647, 0.5941905, 0.582815, 0.5735137, 0.5689071, 0.56796247, 0.56960475, 0.5705489, 0.5721457, 0.5687864, 0.56301385, 0.5526402, 0.54224265, 0.5244536, 0.49796063, 0.4613698, 0.41747725, 0.37864846, 0.3477264, 0.3286198, 0.32111302, 0.3246804, 0.3427289, 0.36920154, 0.3968661, 0.4113295, 0.407757, 0.39948273, 0.39763212, 0.40967605, 0.4192701, 0.42848194, 0.43667692, 0.4309867, 0.42483348, 0.41425547, 0.40539527, 0.39354664, 0.3808289, 0.36753434, 0.3547973, 0.33783865, 0.32595378, 0.31119287, 0.30263528, 0.2970759, 0.2949585, 0.29867494, 0.30995667, 0.31720763, 0.32531708, 0.3321113, 0.33387694, 0.3347053, 0.33117285, 0.33188576, 0.32787344, 0.32412857, 0.3191431, 0.3142073, 0.30612236, 0.30121887, 0.29699335, 0.29323727, 0.29029375, 0.29306513, 0.29420495, 0.29781437, 0.2991016, 0.29780495, 0.29824835, 0.30275083, 0.3019504, 0.30542812, 0.30566156]
        Yh = remove_hull(np.array([y]), upper=True, div=True)

    def test_10_refine(self):
        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=False, d=4)  # compute initial values
        x1 = refine(x, X, x0, 5 )

    def test_11_benchmark(self):
        from gfit.util import benchmark
        benchmark(size=25, vb=True) # run benchmark


if __name__ == '__main__':
    unittest.main()
