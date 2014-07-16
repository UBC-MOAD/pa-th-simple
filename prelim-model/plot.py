"""plotting module
"""
from __future__ import division
import numpy as np
import pylab as plb
import copy
import matplotlib.pyplot as plt
from math import pi

def init(D, P, u, xmin, xmax, zmin, zmax, nx, nz, string):

        # plot initial dist.
	x = np.linspace(xmin, xmax, nx)
	z = np.linspace(zmin, zmax, nz)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz

	# remove NaNs
	idx = np.isnan(D)
	clean_D = np.zeros([nz, nx])
	clean_D[~idx] = D[~idx]

	idx = np.isnan(P)
	clean_P = np.zeros([nz, nx])
	clean_P[~idx] = P[~idx]

        init = plb.subplots(1, 2, figsize = (25, 5))

        plb.subplot(121) 
        mesh1 = plb.pcolormesh(1e-3 * x, z, clean_D)
        if string == 'Th': 
	        plb.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        plb.title('Initial Dissolved [Pa]')
        if string == 'ThPa':
                plb.title('Initial Dissolved [Th]/[Pa]')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')

        plb.subplot(122) 
        mesh2 = plb.pcolormesh(1e-3 * x, z, clean_P)
        if string == 'Th':
	        plb.title('Initial Particulate [Th]')
        if string == 'Pa':
	        plb.title('Initial Particulate [Pa]')
        if string == 'ThPa':
                plb.title('Initial Particulate [Th]/[Pa]')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')

	# plot the velocity field        
	flowfig = plb.figure(figsize = (25, 5))	
	plb.quiver(1e-3*x, z, u[:,:,1], -100*u[:,:,0], pivot = 'mid')
	plb.gca().invert_yaxis()
	plt.title('Velocity Field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        return init

def ratio(DT, DP, PT, PP, xmin, xmax, zmin, zmax, T):
	""" Plots the ratio T/P and outputs to notebook

	:arg DTh: 2D profile of dissolved Th

	:arg PTh: 2D profile of particulate Th

	:arg DPa: 2D profile of dissolved Pa	

	:arg PPa: 2D profile of particulate Pa

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""

        # extract grid points
        nz = DT.nz
        nx = DT.nx

	# define grid
	x = np.linspace(xmin, xmax, nx)
	z = np.linspace(zmin, zmax, nz)
        tmax = 10*T

	# replace NaNs with 0
	Dratio = copy.copy(DT.a/DP.a)                               # use copy fnctn to avoid changing DTh.a outside plot module
	idx = np.isnan(Dratio)
	clean_Dratio = np.zeros([nz, nx])
	clean_Dratio[~idx] = Dratio[~idx]

	Pratio = copy.copy(PT.a/PP.a)
	idx = np.isnan(Pratio)
	clean_Pratio = np.zeros([nz, nx])
	clean_Pratio[~idx] = Pratio[~idx]

	# plot 
	TPratio = plb.subplots(1, 2, figsize = (25, 5))
	
	plb.subplot(121)
	D = plb.pcolormesh(x*1e-3, z, clean_Dratio)
	plb.gca().invert_yaxis()
	plt.title('Dissolved [Th]/[Pa],' + str(tmax) + 'years elapsed')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	plb.colorbar(D)


	plb.subplot(122)
	P = plb.pcolormesh(x*1e-3, z, clean_Pratio)
	plb.gca().invert_yaxis()
	plt.title('Particulate [Th]/[Pa],' + str(tmax) + 'years elapsed')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	plb.colorbar(P)


	return TPratio
	
def prof(g, h, xmin, xmax, zmin, zmax, nx, nz, T, string):
        """Plot the dissolved and particulate profile of either [Th] or [Pa]

        :arg slowfig: figure with 3 subplots. 1 - vel. field; 2 - dissolved initial distribution; 3 - particulate initial distribution

        :arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	:arg string: string, either 'Th' or 'Pa' which determines which title to use on figures
        """

        # define grid
	x_plt = np.linspace(xmin, xmax, nx)
	z_plt = np.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = np.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 1)*dx
        zmax_plt = (nz - 1)*dz
        tmax = 10*T

        meshTh = plb.subplots(1, 2, figsize = (25, 5)) 
        plb.subplot(121) 
        mesh3 = plb.pcolormesh(xx_plt/1e3, zz_plt, g.a)
        if string == 'Th':
	        plb.title('Final Dissolved [Th],' + str(tmax) + 'years elapsed')
        if string == 'Pa':
	        plb.title('Final Dissolved [Pa],' + str(tmax) + 'years elapsed')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')
        plb.colorbar(mesh3)

        plb.subplot(122) 
        mesh4 = plb.pcolormesh(xx_plt/1e3, zz_plt, h.a)
        if string == 'Th':
	        plb.title('Final Particulate [Th],' + str(tmax) + 'years elapsed')
        if string == 'Pa':
	        plb.title('Final Particulate [Pa],' + str(tmax) + 'years elapsed')
        plb.gca().invert_yaxis()
        plb.ylabel('depth [m]')
        plb.xlabel('x [km]')
        plb.colorbar(mesh4)

	return meshTh
