"""Module containing velocity fields to test the preliminary model
"""

from __future__ import division
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from math import pi

def zero(nz, nx):
	""" Produce a matrix of zeros on the input grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""

	# store the solution in a matrix
	u = np.zeros([2, nz, nx])

	return u


def onecell_up(xmin, xmax, zmin, zmax, nx, nz, V):
        """ u_simple computes a simple rotational, divergenceless flow field on a specified grid

        :arg xmin: minimum x on the grid
        :arg xmax: maximum x on the grid
        :arg zmin: minimum z on the grid
        :arg zmax: maximum z on the grid
        :arg nx: number of points in x dimension
        :arg nz: number of points in z dimension	

        """
        # define velocity on square grid, then scale onto rectangular grid. 
        a = zmax
        b = zmax
        x = np.linspace(-a/2, a/2, nx)
        hdz = 0.5*b/nz
        z = np.linspace(-b/2-hdz, b/2+hdz, nz+1)
        [xx, zz] = np.meshgrid(x, z)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        rr = np.sqrt(xx**2 + zz**2)
        idx = rr < a/2
        #ux = np.zeros([nz+1, nx])
        uz = np.zeros([nz+1, nx])

        #ux[idx] = -np.sin(2*pi*rr[idx] / a) / rr[idx] * zz[idx]
        uz[idx] = np.sin(2*pi*rr[idx] / a) / rr[idx] * xx[idx]

        # remove nans
        nanfill = np.zeros((nz, nx))
        #id_nan = np.isnan(ux)
        #ux[id_nan] = nanfill[id_nan]
        id_nan = np.isnan(uz)
        uz[id_nan] = nanfill[id_nan]

        # make sure top two and bottom two rows are zero
        uz[0:2,:] = 0.
        uz[nz-1:nz+1,:] = 0.

        # scale & store the solution in a matrix, shifting up and down
        u = np.zeros([2, nz, nx])
        u[0, :, :nx/2] = uz[0:nz, :nx/2] / np.max(uz) * V * zmax/xmax
        u[0, :, nx/2:] = uz[1:, nx/2:] / np.max(uz) * V * zmax/xmax
        #u[1, :, :nx/2] = ux[0:nz, :nx/2] / np.max(ux) * V 
        #u[1, :, nx/2:] = ux[1:, nx/2:] / np.max(ux) * V

        # extract velocity components for upstream correction
        uz = u[0, :, :]
        ux = np.zeros((nz,nx))
        
        # define upstream as the sum of two adjacent grid point vel.s
        p_upz = np.sign(uz[:-1]+uz[1:])*0.5*( np.sign(uz[:-1]+uz[1:]) - 1)
        n_upz = np.sign(uz[:-1]+uz[1:])*0.5*( np.sign(uz[:-1]+uz[1:]) + 1)

        # z > 0, ux > 0
        i = np.arange(1, nz/2, 1, dtype = int)
        j = 1
        while j <= nx-2:
            # note shift in p_upz and n_upz
            ux[i, j] = ux[i, j - 1] + dx/dz* (( uz[i,j] - uz[i + 1, j])*p_upz[i, j] 
                                               + (uz[i - 1, j] - uz[i,j])*n_upz[i-1, j])
            j += 1

        # z < 0, ux < 0
        i = np.arange(nz/2, nz - 1, 1, dtype = int)
        j = nx - 2
        while j >= 1:
            
            ux[i, j] = ux[i, j + 1] - dx/dz* ((uz[i,j] - uz[i + 1, j]) *p_upz[i, j] + (uz[i - 1, j] - uz[i,j])*n_upz[i-1, j])
            j -= 1

        # store result
        u[1, :, :] = ux

        return u


def twocell(xmin, xmax, zmin, zmax, nx, nz, V):
	""" u_complex complex computes a rotational, downwelling velocity field

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""

	# define a grid that will produce downwelling
	a = zmax
	b = zmax
	x = np.zeros(nx)

        #nx should always be odd
        x[0:nx/2] = np.linspace(-a/2, a/2, nx/2)
        x[nx/2:] = np.linspace(a/2, -a/2, nx/2)
        hdz = 0.5*b/nz
        z = np.linspace(-b/2-hdz, b/2+hdz, nz+1)
	[xx, zz] = np.meshgrid(x, z)
	zz[0:, nx/2:] = - zz[0:, nx/2:]  
	rr = np.sqrt(xx**2 + zz**2)
	ux = np.zeros((nz+1, nx))
	uz = np.zeros((nz+1, nx))

	# use logical indexing to define points of non-zero velocity
	idx = rr < a/2

        ux[idx] = np.sin(2*pi*rr[idx] / a) / rr[idx] * -zz[idx]

        uz[idx] = -np.sin(2*pi*rr[idx] / a) / rr[idx] * -xx[idx]

        # remove nans
        nanfill = np.zeros((nz, nx))
        id_nan = np.isnan(ux)
        ux[id_nan] = nanfill[id_nan]
        id_nan = np.isnan(uz)
        uz[id_nan] = nanfill[id_nan]

        # make sure top two and bottom two rows are zero
        uz[0:2,:] = 0.
        uz[nz-1:nz+1,:] = 0.

        # scale & store the solution in a matrix, shifting up and down
        u = np.zeros([2, nz, nx])
        u[0, :, :nx/4] = uz[0:nz, :nx/4] / np.max(uz) * V * zmax/xmax
        u[0, :, nx/4:] = uz[1:, nx/4:] / np.max(uz) * V * zmax/xmax
        u[0, :, nx/2:] = uz[1:, nx/2:] / np.max(uz) * V * zmax/xmax
        u[0, :, 3*nx/4:] = uz[0:nz, 3*nx/4:] / np.max(uz) * V * zmax/xmax

        u[1, :, :nx/4] = ux[0:nz, :nx/4] / np.max(ux) * V 
        u[1, :, nx/4:] = ux[1:, nx/4:] / np.max(ux) * V
        u[1, :, nx/2:] = ux[1:, nx/2:] / np.max(ux) * V 
        u[1, :, 3*nx/4:] = ux[0:nz, 3*nx/4:] / np.max(ux) * V

	return u

def twocell_c(u, xmin, xmax, zmin, zmax, nx, nz):
        """Correct the complex velocity field to conserve mass on grid-by-grid basis
        """

        # extract velocities for redefinition
        ux = u[1, :,:]
        uz = u[0, :,:]

        # define upstream
        p_upz = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) - 1)
        n_upz = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) + 1)
        p_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) - 1)
        n_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) + 1)

        #define ux = 0 everywhere, define it using uz 
        ux = np.zeros((nz, nx))

        # spatial step
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)

        # vectorize region z > 0
        i = np.arange(1, nz/2, 1, dtype = int)
        j = 1
        while j <= nx/2:
            ux[i, j] = ux[i, j - 1] + dx/dz * ((uz[i - 1, j] - uz[i, j])*n_upz[i - 1, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j])
            j += 1

        j = nx - 2
        while j >= nx/2:
            ux[i, j] = ux[i, j + 1] - dx/dz * ((uz[i - 1, j] - uz[i, j])*n_upz[i - 1, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j])
            j -= 1

        # vectorize region z < 0
        i = np.arange(nz/2, nz - 1, 1, dtype = int)
        j = nx/2 - 2
        while j >= 1:
            ux[i, j] = ux[i, j + 1] - dx/dz * ((uz[i - 1, j] - uz[i, j])*n_upz[i - 1, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j])
            j -= 1

        j = nx/2 + 1
        while j <= nx - 2:
            ux[i, j] = ux[i, j - 1] + dx/dz * ((uz[i - 1, j] - uz[i, j])*n_upz[i - 1, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j])
            j += 1
            
        # store solution
        u[1, :,:] = ux
        
        return u

def divtest(u, xmax, xmin, zmax, zmin, nx, nz, n_upz, p_upz, n_upx, p_upx):
        """compute the divergence of any field on any grid in an upstream scheme
        """
        ux = u[1, :,:]
        uz = u[0, :,:]

        # set up vectorized correction \n",
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1) 

        # QUAD 1
        i = np.arange(1, nz - 1, 1, dtype = int)
        j = 1
        div = np.zeros((nz, nx))
        while j <= nx - 2:

            div[i,j] = dz * ( (ux[i, j] - ux[i, j + 1])*p_upx[i, j] + (ux[i, j - 1] - ux[i, j])*n_upx[i, j - 1] ) + dx * ( (uz[i, j] - uz[i + 1, j])*p_upz[i, j] + (uz[i - 1, j] - uz[i, j])*n_upz[i-1, j] )
            j += 1    


        # plot the results
        plb.figure(figsize = (25, 5))
        divplot = plb.pcolormesh(div)
        plb.colorbar(divplot)
        plb.gca().invert_yaxis()

        return divplot
