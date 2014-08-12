"""Module containing velocity fields to test the preliminary model
"""

from __future__ import division
import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from math import pi

def zero(nz, nx, ny):
	""" Produce a matrix of zeros on the input grid
	:arg nx: number of points in x dimension
	:arg nz: number of points in z dimension
	"""
	# store the solution in a matrix
	u = np.zeros([2, nz, nx, ny])
	return u

def onecell_cen(xmin, xmax, zmin, zmax, nx, nz, ny, V):
        """ u_simple computes a simple rotational, divergenceless flow field on a specified grid
        :arg xmin: minimum x on the grid
        :arg xmax: maximum x on the grid
        :arg zmin: minimum z on the grid
        :arg zmax: maximum z on the grid
        :arg nx: number of points in x dimension
        :arg nz: number of points in z dimension	
        """
        # define velocity on square grid, then scale onto rectangular grid.
        # define velocity on square grid, then scale onto rectangular grid. 
        a = zmax
        b = zmax
        c = zmax
        x = np.linspace(-a/2, a/2, nx)
        y = np.linspace(-b/2, b/2, ny)
        z = np.linspace(-c/2, c/2, nz)
        # the order of mesh
        [xx, zz, yy] = np.meshgrid(x, z, y)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        rr = np.sqrt(xx**2 + zz**2)
        # scale the flow in closer to centre of domain
        scale = 0.9
        idx = rr < (a/2) * scale
        u = np.zeros((3, nz, nx, ny))
        uz = u[0,:,:,:]
        # apply a Gaussian to fade edges
        uz[idx] = np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * xx[idx] * np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
        # scale solution
        uz = uz / np.max(uz) * V * zmax/xmax 

        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int)
        j = 1
        ux = np.zeros((nz,nx,ny))
        while j <= nx-2:

            ux[i, j + 1,:] = ux[i, j - 1,:] - dx/dz* ( uz[i+1, j,:] - uz[i-1,j,:])

            j += 1

        # store result
        u[1,:,:,:] = ux

        return u


def twocell_cen(xmin, xmax, zmin, zmax, nx, nz, ny, V):
        """ u_simple computes a simple rotational, divergenceless flow field on a specified grid
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
c = zmax

x = np.zeros(nx)
x[0:nx/2] = np.linspace(-a/2, a/2, nx/2)
x[nx/2:] = np.linspace(a/2, -a/2, nx/2)

y = np.zeros(nx)
y[0:nx/2] = np.linspace(-c/2, c/2, ny/2)
y[nx/2:] = np.linspace(c/2, -c/2, ny/2)

z = np.linspace(-b/2, b/2, nz)
[xx, zz, yy] = np.meshgrid(x, z, y)
zz[:, nx/2:,:] = - zz[:, nx/2:,:]  

rr = np.sqrt(xx**2 + zz**2)
# scale the flow in closer to centre of domain
scale = 0.9
idx = rr < (a/2) * scale
# apply a Gaussian to fade edges
uz = np.zeros((nz, nx, ny))
uz[idx] = -np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * -xx[idx]* np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
# scale solution
uz = uz / np.max(uz) * V * zmax/xmax 
# store solution in a tensor
u = np.zeros([3, nz, nx, ny])
ux = u[1,:,:,:]
u[0,:,:,:] = uz
# spatial step
dx = (xmax - xmin) / (nx - 1)
dz = (zmax - zmin) / (nz - 1)

# vectorize in z, redefine u[j+1] with u[j-1] 
i = np.arange(1, nz - 1, 1, dtype = int)
j = 1
while j <= nx-2:

    ux[i, j + 1, :] = ux[i, j - 1, :] - dx/dz* ( uz[i + 1,j, :] - uz[i - 1, j, :])

    j += 1

# store solution
u[1,:,:,:] = ux


