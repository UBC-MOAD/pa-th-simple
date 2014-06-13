###################################################TIME LOOP##########################################################################
""" 
Plots of Profiles
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def plot_profiles(zmin,zmax,nz,nx,Conc):
    """
    Plots of Profiles, dissolved and particulate

    :arg zmin: depth at top of model [m]
    :type zmin: float

    :arg zmax: depth at bottom of model [m]
    :type zmax: float

    :arg nz: number of grid points in z dimension
    :type nz: int

    :arg nx: number of grid points in z dimension
    :type nx: int

    :arg Conc: 2D tensor of shape (nz,nx), dissolved concentation in domain
    :type Conc: float

    """

    dz = (zmax-zmin)/nz
    theplot = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    center = np.int(nx/2)
    plt.plot (Conc[:,center-2],dz*np.arange(20),'bo-',label="Upwelling Side")
    plt.plot (Conc[:,center],dz*np.arange(20),'rx-',label="Center")
    plt.plot (Conc[:,center+2],dz*np.arange(20),'k+-',label="Downwelling Side")
    plt.title("Concentration")
    plt.ylim (zmax,zmin)
    plt.legend(loc='center left')
    plt.subplot(1,2,2)
    plt.plot (1e4*(Conc[:,center-2]-Conc[:,center+2]),dz*np.arange(20),'gs-',label = "Upwelling-Downwelling")
    plt.plot (1e4*(Conc[:,center]-Conc[:,center+2]),dz*np.arange(20),'m^-',label = "Center-Downwelling")
    plt.legend(loc='lower left')
    plt.ylim (zmax,zmin)
    plt.title("Difference in Concentration (x1e-4)")

    return theplot
