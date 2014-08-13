""" 
Finite-difference implementation of upwind sol. for coupled linear advection.

We are solving  a_t = Q - k_ad*a + k_de*b + <u> * <del>a

                b_t = k_ad*a - k_de*b + S*b_z + <u> * <del>b
                
The FTCS discretization is: a_new = a_old + (C/2) * (aold_{i+1} - aold_{i-1})
The upwind discretization is: a_new = a_old + C *(aold_{i+1} - aold_{i-1})
 
"""
from __future__ import division
import numpy as np
import pylab as plb
from math import pi

class Fgrid:

	def __init__(self, nx, ny, nz, ng, dt, xmin = 0, xmax = 1e6, ymin = 0, ymax = 1e6, zmin = 0, zmax = 5e3):

		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		self.zmin = zmin
		self.zmax = zmax
		self.ng = ng
		self.nx = nx
		self.ny = ny
		self.nz = nz

		# python is zero-based
		self.ilo = 0
		self.ihi = nz - 1
		self.jlo = 0
		self.jhi = ny - 1
		self.klo = 0
		self.khi = nx - 1

		# physical coords
		self.dx = (xmax - xmin) / (nx - 1)
		self.dx_i = 1/self.dx
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dy = (ymax - ymin) / (ny - 1)
		self.dy_i = 1/self.dy
		self.y = ymin + (np.arange(ny) - ng) * self.dy
		self.dz = (zmax - zmin) / (nz - 1)
		self.dz_i = 1/self.dz
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.yy, self.zz, self.yy] = np.meshgrid(self.y, self.z, self.x)
                self.dt = dt
		# storage for the solution 
		self.a = np.zeros((nz, ny, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.ny, self.nx), dtype=np.float64)

	def fillBCs_d(self, k_ad, Q):    
		self.a[self.ilo, :] = self.a[self.ilo, :] + (Q - k_ad[0, :]*self.a[self.ilo, :] ) * self.dt
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]
		self.a[:, self.klo] = self.a[:, self.klo + 1]
		self.a[:, self.khi] = self.a[:, self.khi - 1]

	def fillBCs_p(self):             
		self.a[self.ilo, :] = 0
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]
		self.a[:, self.klo] = self.a[:, self.klo + 1]
		self.a[:, self.khi] = self.a[:, self.khi - 1]



	
