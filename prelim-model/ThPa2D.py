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

class FDTgrid:

	def __init__(self, nx, nz, ng, xmin = 0, xmax = 1e6, zmin = 0, 
		 zmax = 5e3):

		self.xmin = xmin
		self.xmax = xmax
		self.zmin = zmin
		self.zmax = zmax
		self.ng = ng
		self.nx = nx
		self.nz = nz

		# python is zero-based
		self.ilo = 0
		self.ihi = nz - 1
		self.jlo = 0
		self.jhi = nx - 1

		# physical coords
		self.dx = (xmax - xmin) / (nx - 1)
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = np.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs(self):             
		self.a[self.ilo, :] = self.a[self.ilo, :] + (0.0267 - self.a[self.ilo, :] ) * 0.001
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]

class FPTgrid:

	def __init__(self, nx, nz, ng, xmin = 1, xmax = 1e6, zmin = 0, 
		 zmax = 5e3):

		self.xmin = xmin
		self.xmax = xmax
		self.zmin = zmin
		self.zmax = zmax
		self.ng = ng
		self.nx = nx
		self.nz = nz

		# python is zero-based
		self.ilo = 0
		self.ihi = nz - 1
		self.jlo = 0
		self.jhi = nx - 1

		# physical coords
		self.dx = (xmax - xmin) / (nx - 1)
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = np.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs(self):             
		self.a[self.ilo, :] = 0
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]
class FDPgrid:

	def __init__(self, nx, nz, ng, xmin = 0, xmax = 1e6, zmin = 0, 
		 zmax = 5e3):

		self.xmin = xmin
		self.xmax = xmax
		self.zmin = zmin
		self.zmax = zmax
		self.ng = ng
		self.nx = nx
		self.nz = nz

		# python is zero-based
		self.ilo = 0
		self.ihi = nz - 1
		self.jlo = 0
		self.jhi = nx - 1

		# physical coords
		self.dx = (xmax - xmin) / (nx - 1)
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = np.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs(self):             
		self.a[self.ilo, :] = self.a[self.ilo, :] + ( 0.00246  - 0.08*self.a[self.ilo, :] ) * 0.001
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]

class FPPgrid:

	def __init__(self, nx, nz, ng, xmin = 1, xmax = 1e6, zmin = 0, 
		 zmax = 5e3):

		self.xmin = xmin
		self.xmax = xmax
		self.zmin = zmin
		self.zmax = zmax
		self.ng = ng
		self.nx = nx
		self.nz = nz

		# python is zero-based
		self.ilo = 0
		self.ihi = nz - 1
		self.jlo = 0
		self.jhi = nx - 1

		# physical coords
		self.dx = (xmax - xmin) / (nx - 1)
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = np.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs(self):             
		self.a[self.ilo, :] = 0
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]
		

def adflow(g, h, t, T, u, k_ad, k_de, Q, adscheme):
	"""
	Compute and store the dissolved and particulate [Th] profiles, write them to a file, plot the results.

        :arg t: scale for time at which code is initiated
        :type t: int

	:arg T: scale for time at which code is terminated
	:typeT: int

	:arg V: scale for ux, uz, which are originally order 1.
	:type V: int

	:arg u: 3D tensor of shape (nz, nx, 2), z component of velocity in (:, :, 1), x component of velocity in (:, :, 2) 
	:type u: float

	:arg k_ad: nz x nx adsorption rate matrix
	:type k_ad: float

	:arg k_de: nz x nx adsorption rate matrix
	:type k_de: float

	"""

	# define the CFL, sink velocity, and reaction constant
        S = 500
	S_i = 1/S 
        # time info (yr)
	dt = 0.001   
        t = t * (g.zmax - g.zmin)*S_i
	T = T * (g.zmax - g.zmin)*S_i            

        g, h = adscheme(g, h, t, T, u, k_ad, k_de, Q, S, dt)

        return g, h

def upstream(g, h, t, T, u, k_ad, k_de, Q, S, dt):

	# extract the velocities
	uz = u[:, :, 0]
	ux = u[:, :, 1]

	# evolution loop
	anew = g.a
	bnew = h.a

        # define upstream for particulate phase (contains sinking vel.)
        p_upz_p = np.sign(uz[:-1, :]+uz[1:, :] + S)*0.5*( np.sign(uz[:-1, :]+uz[1:, :] + S) - 1)
        n_upz_p = np.sign(uz[:-1, :]+uz[1:, :] + S)*0.5*( np.sign(uz[:-1, :]+uz[1:, :] + S) + 1)
        # define upstream for dissolved phase
        p_upz_d = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) - 1)
        n_upz_d = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) + 1)
        # define upstream in x
        p_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) - 1)
        n_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) + 1)

        # save inverses for speed
        g.dx_i = 1/g.dx
        g.dz_i = 1/g.dz
        h.dx_i = 1/h.dx
        h.dz_i = 1/h.dz

	while (t < T):

		# fill the boundary conditions (g will be defined by FDgrid, h by FPgrid)
		g.fillBCs()
		h.fillBCs()

		# vectorize spatial indices 
                i = np.arange(g.ilo + 1, g.ihi, 1, dtype = int)
                j = np.arange(g.jlo + 1, g.jhi, 1, dtype = int)
                [i , j] = np.meshgrid(i,j)

                # dissolved:
                anew[i, j] = g.a[i, j] + ( Q - k_ad[i, j] * g.a[i, j] + k_de[i, j] * h.a[i, j] +
                    ux[i, j] * ( n_upx[i, j - 1]*(g.a[i, j - 1] - g.a[i, j]) + p_upx[i, j]*(g.a[i, j] - g.a[i, j + 1]) ) * g.dx_i + 
                    uz[i, j] * ( n_upz_d[i - 1, j]*(g.a[i - 1, j] - g.a[i, j]) + p_upz_d[i, j]*(g.a[i, j] - g.a[i + 1, j]) ) * g.dz_i ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S * ( (n_upz_p[i, j]*h.a[i - 1, j] - h.a[i, j]) + p_upz_p[i, j]*(h.a[i, j] - h.a[i + 1, j]) ) * h.dz_i + k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                    ux[i, j] * ( n_upx[i, j - 1]*(h.a[i, j - 1] - h.a[i, j]) + p_upx[i, j]*(h.a[i, j] - h.a[i, j + 1]) ) * h.dx_i +
                    uz[i, j] * ( n_upz_p[i - 1, j]*(h.a[i - 1, j] - h.a[i, j]) + p_upz_p[i, j]*(h.a[i, j] - h.a[i + 1, j]) ) * h.dz_i ) * dt

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt
        return g, h

def flux(g, h, t, T, u, k_ad, k_de, Q, S, dt):
        """Flux based advection scheme
        """

	# extract the velocities
	uz = u[:, :, 0]
	ux = u[:, :, 1]

	# evolution loop
	anew = g.a
	bnew = h.a

        # define upstream for particulate phase (contains sinking vel.)
        p_upz_p = np.sign(uz[:-1, :]+uz[1:, :] + S)*0.5*( np.sign(uz[:-1, :]+uz[1:, :] + S) - 1)
        n_upz_p = np.sign(uz[:-1, :]+uz[1:, :] + S)*0.5*( np.sign(uz[:-1, :]+uz[1:, :] + S) + 1)
        # define upstream for dissolved phase
        p_upz_d = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) - 1)
        n_upz_d = np.sign(uz[:-1, :]+uz[1:, :])*0.5*( np.sign(uz[:-1, :]+uz[1:, :]) + 1)
        # define upstream in x
        p_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) - 1)
        n_upx = np.sign(ux[:, :-1]+ux[:, 1:])*0.5*( np.sign(ux[:, :-1]+ux[:, 1:]) + 1)

        # save inverses for speed
        g.dx_i = 1/g.dx
        g.dz_i = 1/g.dz
        h.dx_i = 1/h.dx
        h.dz_i = 1/h.dz

	while (t < T):

		# fill the boundary conditions
		g.fillBCs()
		h.fillBCs()

		# vectorize spatial indices
                i = np.arange(g.ilo + 1, g.ihi, 1, dtype = int)
                j = np.arange(g.jlo + 1, g.jhi, 1, dtype = int)
                [i , j] = np.meshgrid(i,j)

                # dissolved:
                anew[i, j] = g.a[i, j] + ( Q - k_ad[i, j] * g.a[i, j] + k_de[i, j] * h.a[i, j] +                 
                                ( n_upx[i, j - 1]*(g.a[i, j - 1]*ux[i, j - 1] - g.a[i, j]*ux[i, j]) + 
                                  p_upx[i, j]*(g.a[i, j]*ux[i, j] - g.a[i, j + 1]*ux[i, j + 1]) ) * g.dx_i + 
                                ( n_upz_d[i - 1, j]*(g.a[i - 1, j]*uz[i - 1, j] - g.a[i, j]*uz[i, j]) + 
                                  p_upz_d[i, j]*(g.a[i, j]*uz[i, j] - g.a[i + 1, j]*uz[i + 1, j]) ) * g.dz_i ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S *( n_upz_p[i, j]*(h.a[i - 1, j] - h.a[i, j]) + p_upz_p[i, j]*(h.a[i, j] - h.a[i + 1, j]) )* h.dz_i + k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] +      
                                ( n_upx[i, j - 1]*( h.a[i, j - 1]*ux[i, j - 1] - h.a[i, j]*ux[i, j] ) + 
                                  p_upx[i, j]*( h.a[i, j]*ux[i, j] - h.a[i, j + 1]*ux[i, j + 1]) ) * h.dx_i +
                                ( n_upz_p[i - 1, j]*( h.a[i - 1, j]*uz[i - 1, j] - h.a[i, j]*uz[i, j] ) + 
                                  p_upz_p[i, j]*( h.a[i, j]*uz[i, j] - h.a[i + 1, j]*uz[i + 1, j] ) ) * h.dz_i ) * dt

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt

        return g, h

def k_sorp(string, xmin, xmax, zmin, zmax, nx, nz):
	""" Computes adsorption,desorption, & production constants for either Th or Pa

	:arg string: a string, either 'Th' or 'Pa'

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension
	
	"""
	# physical coords
	dx = (xmax - xmin) / (nx - 1)
	x = xmin + (np.arange(nx) - 1) * dx
	dz = (zmax - zmin) / (nz - 1)
	z = zmin + (np.arange(nz) - 1) * dz
	[xx, zz] = np.meshgrid(x, z)

	if string == 'Pa':

		k_ad = np.ones(np.shape(zz))
		k_ad[:, :] = 0.08
		k_ad[251 <= z, :] = 0.06
		k_ad[500 <= z, :] = 0.04

		k_de = np.zeros((np.shape(zz)))
		k_de[:] = 1.6
		
		Q = 0.00246

	if string == 'Th':
		k_ad = np.ones(np.shape(zz))
		k_ad[251 <= z, :] = 0.75
		k_ad[500 <= z, :] = 0.5

		k_de = np.ones(np.shape(zz))

		Q = 0.0267
	
	return k_ad, k_de, Q	



