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
		self.dx_i = 1/self.dx
		self.x = xmin + (np.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.dz_i = 1/self.dz
		self.z = zmin + (np.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = np.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs_d(self):             
		self.a[self.ilo, :] = 2*self.a[self.ilo + 1, :] - self.a[self.ilo + 2, :]
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi] = self.a[:, self.jhi - 1]

	def fillBCs_p(self):             
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

	:arg nx, nz: size of the arrays
	:type nx, nz: int

	"""

	# define the CFL, sink velocity, and reaction constant
        S = 500 

        # time info (yr)
	dt = 0.001   
        gS = (g.zmax - g.zmin)/S
	t *= gS
	T *= gS    

	# extract the velocities
	u_rolled = np.rollaxis(u, axis=2)
	uz, ux = u_rolled

	# upstream factors
	sign_uz_S = np.sign(uz[:-1, :] + uz[1:, :] + S)
	sign_uz = np.sign(uz[:-1, :] + uz[1:, :])
	sign_ux = np.sign(ux[:, :-1] + ux[:, 1:])
        # define upstream for particulate phase (contains sinking vel.)
	p_upz_p = sign_uz_S * (sign_uz_S - 1)/2
	n_upz_p = sign_uz_S * (sign_uz_S + 1)/2
        # define upstream for dissolved phase
	p_upz_d = sign_uz * (sign_uz - 1)/2
	n_upz_d = sign_uz * (sign_uz + 1)/2
	# define upstream in x
	p_upx = sign_ux * (sign_ux - 1)/2
	n_upx = sign_ux * (sign_ux + 1)/2


        while (t < T):

                # dissolved:
                g.a += ( Q - k_ad * g.a + k_de * h.a 
                             + adscheme(g, u_rolled, p_upz_d, n_upz_d, p_upx, n_upx) ) * dt

                # particulate:
                h.a += ( k_ad * g.a - k_de * h.a 
                             + adscheme(h, u_rolled, p_upz_p, n_upz_p, p_upx, n_upx, sinkrate=S) ) * dt


                g.fillBCs_d()
                h.fillBCs_p()

                t += dt

        return g, h


def flux(conc, u, p_upz, n_upz, p_upx, n_upx, sinkrate=0):
        """Flux based advection scheme
        """
	nz, nx = conc.nz, conc.nx
	
	vert_flux = conc.a * u[0]
	horz_flux = conc.a * (u[1] + sinkrate)
	
	left_flux =  n_upx[1:nz-1, 0:nx-2] * ( horz_flux[1:nz-1, 0:nx-2] - horz_flux[1:nz-1, 1:nx-1] )
	right_flux = p_upx[1:nz-1, 1:nx-1] * ( horz_flux[1:nz-1, 1:nx-1] - horz_flux[1:nz-1, 2:nx] )
	up_flux = n_upz[0:nz-2, 1:nx-1] * ( vert_flux[0:nz-2, 1:nx-1] - vert_flux[1:nz-1, 1:nx-1] )
        down_flux = p_upz[1:nz-1, 1:nx-1] * ( vert_flux[1:nz-1, 1:nx-1] - vert_flux[2:nz, 1:nx-1] )
		    
	adv = np.empty_like(conc.a)
        adv[1:nz-1, 1:nx-1] = (left_flux + right_flux) * conc.dx_i + (up_flux + down_flux) * conc.dz_i
                 
        return adv

