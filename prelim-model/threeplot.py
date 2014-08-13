"""plotting module
"""
from __future__ import division
import numpy as np
import pylab as plb
import copy
import matplotlib.pyplot as plt
from math import pi
from mpl_toolkits.axes_grid1 import make_axes_locatable


def init(D, P, u, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, string, n):

        # plot initial dist.
	x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
	z = np.linspace(zmin, zmax, nz)
        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)
        dz = (zmax - zmin) / (nz - 1)
	# remove NaNs
	idx = np.isnan(D)
	clean_D = np.zeros([nz, ny, nx])
	clean_D[~idx] = D[~idx]

	idx = np.isnan(P)
	clean_P = np.zeros([nz, ny, nx])
	clean_P[~idx] = P[~idx]

        init = plb.subplots(1, 2, figsize = (25, 5))

        plb.subplot(121) 
        plb.pcolormesh(1e-3 * x, z, clean_D[:, n, :])
        if string == 'Th': 
	        plb.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        plb.title('Initial Dissolved [Pa]')
        if string == 'ThPa':
                plb.title('Initial Dissolved [Th]/[Pa]')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')
	plb.colorbar()

        plb.subplot(122) 
        plb.pcolormesh(1e-3 * x, z, clean_P[:, n, :])
        if string == 'Th':
	        plb.title('Initial Particulate [Th]')
        if string == 'Pa':
	        plb.title('Initial Particulate [Pa]')
        if string == 'ThPa':
                plb.title('Initial Particulate [Th]/[Pa]')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')
	plb.colorbar()



        flow = plb.subplots(1, 2, figsize = (25, 10))
	# plot a z-slice of the velocity field        
        plb.subplot(121)
        im = plb.imshow(100*u[0,n,:,:], interpolation='none',extent=[0,1000,0,1000])
        plb.quiver(1e-3*x, 1e-3*y, u[1,n,:,:], u[2,n,:,:], pivot = 'mid')
        plb.xlabel('x [km]')
        plb.ylabel('y [km]')
	plt.title('x-y velocity quiver with z velocity colormesh')
        plb.gca().invert_yaxis()
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
	# plot a y-slice of the velocity field        
        plb.subplot(122)
        plb.quiver(1e-3*x, z, u[1,:,n,:], -100*u[0,:,n,:], pivot = 'mid')
	plb.gca().invert_yaxis()
	plt.title('x-z Velocity Field')
	plt.xlabel('x [km]')
	plt.ylabel('z [m]')

        return init, flow
