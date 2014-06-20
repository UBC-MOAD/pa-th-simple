""" This test module is set to test the output of an advection scheme given the following input parameters:
"""
from __future__ import division
import numpy as np
import ThPa2D

# run for following outputs
time = (0, 0.001, 2, 3, 5, 10)
t = time[0]
T = time[1]
V = 1  
xmin = 0
xmax = 1e6
zmin = 0
zmax = 5e3
nz = 10
nx = 10
ng = 1
g = ThPa2D.FDgrid(nx, nz, ng)
h = ThPa2D.FDgrid(nx, nz, ng)
# chemistry
k_ad, k_de, Q = ThPa2D.k_sorp('Th', xmin, xmax, zmin, zmax, nx, nz)
S = 500
# velocity = u1
u1, flowfig, init = ThPa2D.u_simple(g, h, xmin, xmax, zmin, zmax, nx, nz, V, 'Th')
# velocity = u2
u2, flowfig, init = ThPa2D.u_complex(g, h, xmin, xmax, zmin, zmax, nx, nz, V, 'Th')


def test_DTh_u1():
    g = ThPa2D.FDgrid(nx, nz, ng)
    h = ThPa2D.FDgrid(nx, nz, ng)
    afterg, afterh = ThPa2D.adflow(g, h, t, T, u1, k_ad, k_de, Q, ThPa2D.upwind)

    beforeg = [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.00023934,  0.00026581,  0.00026581,  0.00026581,  0.00026581,
         0.00026581,  0.00026581,  0.00026581,  0.00026581,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]

    np.testing.assert_allclose(afterg.a, beforeg, rtol=0, atol = 1e-8) 


def test_PTh_u1():
    g = ThPa2D.FDgrid(nx, nz, ng)
    h = ThPa2D.FDgrid(nx, nz, ng)
    afterg, afterh = ThPa2D.adflow(g, h, t, T, u1, k_ad, k_de, Q, ThPa2D.upwind)

    beforeh = [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.19225355e-06,   1.45579434e-06,   1.45579435e-06,
          1.45579435e-06,   1.45579435e-06,   1.45579433e-06,
          1.45579432e-06,   1.45579433e-06,   1.45579433e-06,
          0.00000000e+00],
       [  6.00847926e-07,   7.34217655e-07,   7.34217641e-07,
          7.34217641e-07,   7.34217652e-07,   7.34217667e-07,
          7.34217678e-07,   7.34217677e-07,   7.34217675e-07,
          0.00000000e+00],
       [  5.98896935e-07,   7.31620949e-07,   7.31620944e-07,
          7.31620945e-07,   7.31620951e-07,   7.31620959e-07,
          7.31620965e-07,   7.31620965e-07,   7.31620966e-07,
          0.00000000e+00],
       [  5.98891666e-07,   7.31613353e-07,   7.31613348e-07,
          7.31613350e-07,   7.31613357e-07,   7.31613366e-07,
          7.31613373e-07,   7.31613374e-07,   7.31613374e-07,
          0.00000000e+00],
       [  5.98891654e-07,   7.31613334e-07,   7.31613331e-07,
          7.31613334e-07,   7.31613342e-07,   7.31613351e-07,
          7.31613357e-07,   7.31613358e-07,   7.31613357e-07,
          0.00000000e+00],
       [  5.98891656e-07,   7.31613336e-07,   7.31613338e-07,
          7.31613342e-07,   7.31613350e-07,   7.31613357e-07,
          7.31613361e-07,   7.31613359e-07,   7.31613358e-07,
          0.00000000e+00],
       [  5.98891660e-07,   7.31613340e-07,   7.31613342e-07,
          7.31613346e-07,   7.31613352e-07,   7.31613358e-07,
          7.31613359e-07,   7.31613354e-07,   7.31613353e-07,
          0.00000000e+00],
       [  5.98891660e-07,   7.31613341e-07,   7.31613342e-07,
          7.31613346e-07,   7.31613352e-07,   7.31613357e-07,
          7.31613358e-07,   7.31613353e-07,   7.31613353e-07,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]]

    np.testing.assert_allclose(afterh.a, beforeh, rtol=0, atol = 1e-10) 

def test_DTh_u2():
    g = ThPa2D.FDgrid(nx, nz, ng)
    h = ThPa2D.FDgrid(nx, nz, ng)
    afterg, afterh = ThPa2D.adflow(g, h, t, T, u2, k_ad, k_de, Q, ThPa2D.upwind)

    beforeg = [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.00023934,  0.00026581,  0.00026581,  0.00026581,  0.00026581,
         0.00026581,  0.00026581,  0.00026581,  0.00026581,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]

    np.testing.assert_allclose(afterg.a, beforeg, rtol=0, atol = 1e-8)

def test_PTh_u2():
    g = ThPa2D.FDgrid(nx, nz, ng)
    h = ThPa2D.FDgrid(nx, nz, ng)
    afterg, afterh = ThPa2D.adflow(g, h, t, T, u2, k_ad, k_de, Q, ThPa2D.upwind)

    beforeh = [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.19225354e-06,   1.45579433e-06,   1.45579436e-06,
          1.45579436e-06,   1.45579434e-06,   1.45579434e-06,
          1.45579435e-06,   1.45579434e-06,   1.45579435e-06,
          0.00000000e+00],
       [  6.00847947e-07,   7.34217681e-07,   7.34217673e-07,
          7.34217673e-07,   7.34217659e-07,   7.34217659e-07,
          7.34217637e-07,   7.34217659e-07,   7.34217666e-07,
          0.00000000e+00],
       [  5.98896950e-07,   7.31620966e-07,   7.31620964e-07,
          7.31620964e-07,   7.31620955e-07,   7.31620955e-07,
          7.31620943e-07,   7.31620955e-07,   7.31620959e-07,
          0.00000000e+00],
       [  5.98891684e-07,   7.31613374e-07,   7.31613361e-07,
          7.31613361e-07,   7.31613361e-07,   7.31613361e-07,
          7.31613348e-07,   7.31613361e-07,   7.31613361e-07,
          0.00000000e+00],
       [  5.98891672e-07,   7.31613355e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613329e-07,   7.31613342e-07,   7.31613342e-07,
          0.00000000e+00],
       [  5.98891670e-07,   7.31613354e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613336e-07,   7.31613351e-07,   7.31613351e-07,
          0.00000000e+00],
       [  5.98891667e-07,   7.31613349e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613355e-07,   7.31613355e-07,
          0.00000000e+00],
       [  5.98891667e-07,   7.31613349e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613355e-07,   7.31613355e-07,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]]

    np.testing.assert_allclose(afterh.a, beforeh, rtol=0, atol = 1e-10)


def test_jvecu1g():
        g = ThPa2D.FDgrid(nx, nz, ng)
        h = ThPa2D.FDgrid(nx, nz, ng)
        gjvec, hjvec = ThPa2D.adflow(g, h, t, T, u1, k_ad, k_de, Q, ThPa2D.upwind)

        gup = [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.00023934,  0.00026581,  0.00026581,  0.00026581,  0.00026581,
         0.00026581,  0.0002658 ,  0.00026581,  0.00026581,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]


        np.testing.assert_allclose(gjvec.a, gup, rtol=0, atol = 1e-8)


def test_jvecu1h():
        g = ThPa2D.FDgrid(nx, nz, ng)
        h = ThPa2D.FDgrid(nx, nz, ng)
        gjvec, hjvec = ThPa2D.adflow(g, h, t, T, u1, k_ad, k_de, Q, ThPa2D.upwind)

        hup = [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.19225355e-06,   1.45579434e-06,   1.45579850e-06,
          1.45580380e-06,   1.45579890e-06,   1.45578978e-06,
          1.45578488e-06,   1.45579018e-06,   1.45579086e-06,
          0.00000000e+00],
       [  6.00845691e-07,   7.34214768e-07,   7.34206346e-07,
          7.34205872e-07,   7.34212744e-07,   7.34222574e-07,
          7.34229447e-07,   7.34228973e-07,   7.34227630e-07,
          0.00000000e+00],
       [  5.98893906e-07,   7.31617247e-07,   7.31614279e-07,
          7.31614817e-07,   7.31618493e-07,   7.31623416e-07,
          7.31627093e-07,   7.31627631e-07,   7.31628110e-07,
          0.00000000e+00],
       [  5.98887309e-07,   7.31608031e-07,   7.31605076e-07,
          7.31606043e-07,   7.31610467e-07,   7.31616256e-07,
          7.31620679e-07,   7.31621646e-07,   7.31621727e-07,
          0.00000000e+00],
       [  5.98887292e-07,   7.31608006e-07,   7.31606703e-07,
          7.31608452e-07,   7.31613332e-07,   7.31619124e-07,
          7.31623096e-07,   7.31623285e-07,   7.31622900e-07,
          0.00000000e+00],
       [  5.98888634e-07,   7.31609645e-07,   7.31610677e-07,
          7.31613324e-07,   7.31618206e-07,   7.31623090e-07,
          7.31625508e-07,   7.31623942e-07,   7.31623409e-07,
          0.00000000e+00],
       [  5.98890857e-07,   7.31612359e-07,   7.31613332e-07,
          7.31615983e-07,   7.31619968e-07,   7.31623291e-07,
          7.31623950e-07,   7.31620975e-07,   7.31620380e-07,
          0.00000000e+00],
       [  5.98890857e-07,   7.31612459e-07,   7.31613363e-07,
          7.31615931e-07,   7.31619764e-07,   7.31622906e-07,
          7.31623415e-07,   7.31620378e-07,   7.31620210e-07,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]]

        np.testing.assert_allclose(hjvec.a, hup, rtol=0, atol = 1e-10)


def test_jvecu2g():
        g = ThPa2D.FDgrid(nx, nz, ng)
        h = ThPa2D.FDgrid(nx, nz, ng)
        gjvec, hjvec = ThPa2D.adflow(g, h, t, T, u2, k_ad, k_de, Q, ThPa2D.upwind)

        gup = [[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.00023934,  0.0002658 ,  0.00026581,  0.00026581,  0.00026581,
         0.00026581,  0.00026581,  0.00026581,  0.00026581,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.00023982,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,
         0.0002664 ,  0.0002664 ,  0.0002664 ,  0.0002664 ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]

        np.testing.assert_allclose(gjvec.a, gup, rtol=0, atol = 1e-8)

        
def test_jvecu2h():

        g = ThPa2D.FDgrid(nx, nz, ng)
        h = ThPa2D.FDgrid(nx, nz, ng)
        gjvec, hjvec = ThPa2D.adflow(g, h, t, T, u2, k_ad, k_de, Q, ThPa2D.upwind)

        hup = [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00],
       [  1.19224745e-06,   1.45578595e-06,   1.45580598e-06,
          1.45580598e-06,   1.45579434e-06,   1.45579434e-06,
          1.45580272e-06,   1.45579434e-06,   1.45580009e-06,
          0.00000000e+00],
       [  6.00858644e-07,   7.34231497e-07,   7.34225977e-07,
          7.34225977e-07,   7.34217659e-07,   7.34217659e-07,
          7.34203821e-07,   7.34217659e-07,   7.34221756e-07,
          0.00000000e+00],
       [  5.98902949e-07,   7.31628303e-07,   7.31626826e-07,
          7.31626826e-07,   7.31620955e-07,   7.31620955e-07,
          7.31613607e-07,   7.31620955e-07,   7.31623844e-07,
          0.00000000e+00],
       [  5.98898460e-07,   7.31621649e-07,   7.31613382e-07,
          7.31613382e-07,   7.31613361e-07,   7.31613361e-07,
          7.31605073e-07,   7.31613361e-07,   7.31613364e-07,
          0.00000000e+00],
       [  5.98898450e-07,   7.31621633e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31605051e-07,   7.31613342e-07,   7.31613342e-07,
          0.00000000e+00],
       [  5.98897632e-07,   7.31620634e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31609681e-07,   7.31619183e-07,   7.31619183e-07,
          0.00000000e+00],
       [  5.98895488e-07,   7.31618016e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613329e-07,   7.31621624e-07,   7.31621624e-07,
          0.00000000e+00],
       [  5.98895488e-07,   7.31617688e-07,   7.31613342e-07,
          7.31613342e-07,   7.31613342e-07,   7.31613342e-07,
          7.31613402e-07,   7.31621383e-07,   7.31621384e-07,
          0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00]]

        np.testing.assert_allclose(hjvec.a, hup, rtol=0, atol = 1e-10)

