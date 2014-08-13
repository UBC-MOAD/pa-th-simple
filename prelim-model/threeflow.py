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

def onecell_cen_xz(xmin, xmax, zmin, zmax, nx, nz, ny, V):
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
        c = zmax
        x = np.linspace(-a/2, a/2, nx)
        y = np.linspace(-b/2, b/2, ny)
        z = np.linspace(-c/2, c/2, nz)
        # the order of mesh
        [yy, zz, xx] = np.meshgrid(y,z,x)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        rr = np.sqrt(xx**2 + zz**2)
        # scale the flow in closer to centre of domain
        scale = 0.9
        idx = rr < (a/2) * scale
        # store result
        u = np.zeros((3, nz, ny, nx))
        uz = u[0,:,:,:]
        # apply a Gaussian to fade edges
        uz[idx] = np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * xx[idx] * np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
        # scale solution
        uz = uz / np.max(uz) * V * zmax/xmax 
                
        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int)
        k = np.arange(1, ny - 1, 1, dtype = int)
        j = 1
        ux = u[1,:,:,:]
        while j <= nx-2:

            ux[i,k,j+1] = ux[i,k,j-1] - dx/dz* ( uz[i+1,k,j] - uz[i-1,k,j])

            j += 1

        return u


def twocell_cen_xz(xmin, xmax, zmin, zmax, nx, nz, ny, V):
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
        x = np.zeros(nx)
        x[0:nx/2] = np.linspace(-a/2, a/2, nx/2)
        x[nx/2:] = np.linspace(a/2, -a/2, nx/2)
        y = x
        z = np.linspace(-b/2, b/2, nz)
        [yy, zz, xx] = np.meshgrid(y, z, x)
        zz[:, nx/2:,:] = - zz[:, nx/2:,:]  

        rr = np.sqrt(xx**2 + zz**2)
        # scale the flow in closer to centre of domain
        scale = 0.9
        idx = rr < (a/2) * scale
        # store solution in a tensor
        u = np.zeros([3, nz, ny, nx])
        uz = u[0,:,:,:]
        ux = u[1,:,:,:]
        # apply a Gaussian to fade edges
        uz[idx] = -np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * -xx[idx]* np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
        # scale solution
        uz = uz / np.max(uz) * V * zmax/xmax 
        # spatial step
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int)
        k = np.arange(1, ny - 1, 1, dtype = int)
        j = 1
        while j <= nx-2:

            ux[i,k,j+1] = ux[i,k,j-1] - dx/dz* ( uz[i+1,k,j] - uz[i-1,k,j])

            j += 1

        return u

def onecell_cen_xyz(xmin, xmax, ymin, ymax, zmin, zmax, nx, nz, ny, V):


        # define velocity on square grid, then scale onto rectangular grid. 
        # overturning in x-z plane
        a = zmax
        b = zmax
        c = zmax
        x = np.linspace(-a/2, a/2, nx)
        y = np.linspace(-b/2, b/2, ny)
        z = np.linspace(-c/2, c/2, nz)
        # the order of mesh
        [yy, zz, xx] = np.meshgrid(y, z, x)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        dy = (ymax - ymin) / (ny - 1)
        rr = np.sqrt(xx**2 + zz**2)
        # scale the flow in closer to centre of domain
        scale = 0.9
        idx = rr < (a/2) * scale
        uz = np.zeros([nz, ny, nx])
        # apply a Gaussian to fade edges
        uz[idx] = np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * xx[idx] * np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
        # scale solution
        uz = uz / np.max(uz) * V * zmax/xmax 
        # rotate by 90 degrees into y-z plane
        rr = np.sqrt(yy**2 + zz**2)
        # scale the flow in closer to centre of domain
        idx = rr < (a/2) * scale
        uz90 = np.zeros([nz, nx, ny])
        # apply a Gaussian to fade edges
        uz90[idx] = np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * yy[idx] * np.exp(-2*rr[idx]**2/(a/2*scale)**2) 
        # scale solution
        uz90 = uz90 / np.max(uz90) * V * zmax/ymax 
        u = np.zeros((3, nz, nx, ny))
        u[0,:,:,:] = uz + uz90
        # define ux for uz in x-z
        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int)
        k = np.arange(1, ny - 1, 1, dtype = int)
        j = 1
        ux = u[1,:,:,:]
        while j <= nx-2:

                ux[i,k,j+1] = ux[i,k,j-1] - dx/dz* ( uz[i+1,k,j] - uz[i-1,k,j])

                j += 1

        # define uy for uz in y-z
        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int)
        k = np.arange(1, nx - 1, 1, dtype = int)
        j = 1
        uy= u[2,:,:,:]
        while j <= ny-2:

                uy[i,j+1,k] = uy[i,j-1,k] - dy/dz* ( uz90[i+1,j,k] - uz90[i-1,j,k])

                j += 1

        return u


def twocell_cen_xyz(xmin, xmax, ymin, ymax, zmin, zmax, nx, nz, ny, V):

        # define a grid that will produce downwelling
        a = zmax; b = zmax; c = zmax
        x = np.zeros(nx); x[0:nx/2] = np.linspace(-a/2, a/2, nx/2); x[nx/2:] = np.linspace(a/2, -a/2, nx/2)
        y = np.zeros(nx); y[0:nx/2] = np.linspace(-c/2, c/2, ny/2); y[nx/2:] = np.linspace(c/2, -c/2, ny/2)
        z = np.linspace(-b/2, b/2, nz)
        [yy, zz, xx] = np.meshgrid(y, z, x)
        zz[:, nx/2:,:] = - zz[:, nx/2:,:]  
        rr = np.sqrt(xx**2 + zz**2)
        # spatial step
        dx = (xmax - xmin) / (nx - 1); dy = (ymax - ymin) / (ny - 1); dz = (zmax - zmin) / (nz - 1)
        # scale the flow in closer to centre of domain
        scale = 0.9
        idx = rr < (a/2) * scale
        # apply a Gaussian to fade edges and scale solution
        uz = np.zeros((nz, nx, ny)); uz[idx] = -np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * -xx[idx]* np.exp(-2*rr[idx]**2/(a/2*scale)**2); uz = uz / np.max(uz) * V * zmax/xmax 
        # store solution in a tensor
        u = np.zeros([3, nz, nx, ny])
        ux = u[1,:,:,:]
        # vectorize in z, redefine u[j+1] with u[j-1] 
        i = np.arange(1, nz - 1, 1, dtype = int); j = 1;         k = np.arange(1, ny - 1, 1, dtype = int)
        while j <= nx-2:
                ux[i,k,j+1] = ux[i,k,j-1] - dx/dz* ( uz[i+1,k,j] - uz[i-1,k,j])
                j += 1

        # define a grid that will produce downwelling
        rr = np.sqrt(yy**2 + zz**2)
        idx = rr < (a/2) * scale
        # apply a Gaussian to fade edges and scale solution
        uz90 = np.zeros((nz, nx, ny)); uz90[idx] = -np.sin(2*pi*rr[idx] / (a*scale)) / rr[idx] * -yy[idx]* np.exp(-2*rr[idx]**2/(a/2*scale)**2); uz90 = uz90 / np.max(uz90) * V * zmax/ymax 
        # store solution in a tensor
        u[0,:,:,:] = uz90 + uz
        uy = u[2,:,:,:]
        j = 1
        k = np.arange(1, nx - 1, 1, dtype = int)
        while j <= ny-2:
                uy[i,j+1,k] = uy[i,j-1,k] - dy/dz* ( uz90[i + 1,j,k] - uz90[i-1,j,k])
                j += 1

        return u


