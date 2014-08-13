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
        gS = (g.zmax - g.zmin)/S
	t *= gS
	T *= gS    

	# extract the velocities
	uz = u[0,:,:,:]
	ux = u[1,:,:,:]
        uy = u[2,:,:,:]

	# upstream factors
	sign_uz_S = np.sign(uz[:-1,:,:] + uz[1:,:,:] + S)
	sign_uz = np.sign(uz[:-1,:,:] + uz[1:,:,:])
	sign_ux = np.sign(ux[:,:,:-1] + ux[:,:,1:])
	sign_uy = np.sign(uy[:,:-1,:] + uy[:,1:,:])
        # define upstream for particulate phase (contains sinking vel.)
	p_upz_p = sign_uz_S * (sign_uz_S - 1)/2
	n_upz_p = sign_uz_S * (sign_uz_S + 1)/2
        # define upstream for dissolved phase
	p_upz_d = sign_uz * (sign_uz - 1)/2
	n_upz_d = sign_uz * (sign_uz + 1)/2
	# define upstream in x
	p_upx = sign_ux * (sign_ux - 1)/2
	n_upx = sign_ux * (sign_ux + 1)/2
	# define upstream in y
	p_upy = sign_uy * (sign_uy - 1)/2
	n_upy = sign_uy * (sign_uy + 1)/2


        while (t < T):

                # dissolved:
                g.a += ( Q - k_ad * g.a + k_de * h.a + adscheme(g, u, p_upz_d, n_upz_d, p_upy, n_upy, p_upx, n_upx, sinkrate = 0) ) * g.dt

                # particulate:
                h.a += ( k_ad * g.a - k_de * h.a + adscheme(h, u, p_upz_p, n_upz_p, p_upy, n_upy, p_upx, n_upx, sinkrate = S) ) * h.dt
                
                g.fillBCs_d(k_ad, Q)
                h.fillBCs_p()

                t += g.dt

        return g, h

def flux(conc, u, p_upz, n_upz, p_upy, n_upy, p_upx, n_upx, sinkrate):
        """Flux based advection scheme
        """
	nz, ny, nx = conc.nz, conc.ny, conc.nx
	
	vert_flux = conc.a * (u[0,:,:,:] + sinkrate)
	horx_flux = conc.a * u[1,:,:,:]
	hory_flux = conc.a * u[2,:,:,:]
	
	leftx_flux =  n_upx[1:nz-1, 1:ny-1, 0:nx-2] * ( horx_flux[1:nz-1, 1:ny-1, 0:nx-2] - horx_flux[1:nz-1, 1:ny-1, 1:nx-1] )
	rightx_flux = p_upx[1:nz-1, 1:ny-1, 1:nx-1] * ( horx_flux[1:nz-1, 1:ny-1, 1:nx-1] - horx_flux[1:nz-1, 1:ny-1, 2:nx] )
	lefty_flux =  n_upx[1:nz-1, 0:ny-2, 1:nx-1] * ( hory_flux[1:nz-1, 0:ny-2, 1:nx-1] - hory_flux[1:nz-1, 1:ny-1, 1:nx-1] )
	righty_flux = p_upx[1:nz-1, 1:ny-1, 1:nx-1] * ( hory_flux[1:nz-1, 1:ny-1, 1:nx-1] - hory_flux[1:nz-1, 2:ny, 1:nx-1] )
	up_flux = n_upz[0:nz-2, 1:ny-1, 1:nx-1] * ( vert_flux[0:nz-2, 1:ny-1, 1:nx-1] - vert_flux[1:nz-1, 1:ny-1, 1:nx-1] )
        down_flux = p_upz[1:nz-1, 1:ny-1, 1:nx-1] * ( vert_flux[1:nz-1, 1:ny-1, 1:nx-1] - vert_flux[2:nz, 1:ny-1, 1:nx-1] )
		    
	adv = np.empty_like(conc.a)
        adv[1:nz-1, 1:ny-1, 1:nx-1] = (leftx_flux + rightx_flux) * conc.dx_i + (lefty_flux + righty_flux) * conc.dy_i + (up_flux + down_flux) * conc.dz_i
                 
        return adv





def k_sorp3(string, zmin, zmax, nx, ny, nz):
	""" Computes adsorption,desorption, & production constants for either Th or Pa
	:arg string: a string, either 'Th' or 'Pa'
	:arg xmin: minimum x on the grid (latitude)
	:arg xmax: maximum x on the grid
	:arg ymin: minimum y on the grid (langitude)
	:arg ymax: maximum y on the grid
	:arg zmin: minimum z on the grid (depth)
	:arg zmax: maximum z on the grid
	:arg nx: number of points in x dimension
	:arg nz: number of points in z dimension
	"""
        # spatial coordinates
	z = np.linspace(zmin, zmax, nz)

        if string == 'Pa':
                # define number of points per region
                idx = (np.round((70 - 47.5)*nx/140), np.round((47.5 - 45)*nx/140), np.round((45 - 42.5)*nx/140), np.round((42.5 + 70)*nx/140))
                # define indices based on number of points:
                idx = (idx[0], idx[0] + idx[1], idx[0] + idx[1] + idx[2], idx[0] + idx[1] + idx[2] + idx[3])
                k_ad = np.ones((nz, ny, nx))
                # 70S to 47.5S
                k_ad[:, :, :idx[0]] = 0.44
                k_ad[251 <= z, :, :idx[0]] = 0.33 
                k_ad[500 <= z, :, :idx[0]] = 0.22
                # 47.5S to 45S
                k_ad[:, :, idx[0]:idx[1]] = 0.3
                k_ad[251 <= z, :, idx[0]:idx[1]] = 0.225 
                k_ad[500 <= z, :, idx[0]:idx[1]] = 0.15
                # 45S to 42.5S
                k_ad[:, :, idx[1]:idx[2]] = 0.2
                k_ad[251 <= z, :, idx[1]:idx[2]] = 0.15
                k_ad[500 <= z, :, idx[1]:idx[2]] = 0.1
                #42.5S to 70N
                k_ad[:, :, idx[2]:] = 0.08
                k_ad[251 <= z, :, idx[2]:] = 0.06
                k_ad[500 <= z, :, idx[2]:] = 0.04
                # desorption: constant in lat, lon, and depth
                k_de = np.zeros((nz, ny, nx))
                k_de[:] = 1.6
                # production: constant in lat, lon, and depth
                Q = 0.00246

        if string == 'Th':
                # define number of points in each region
                idx = (np.round((70 - 50)*nx/140), np.round((50 + 70)*nx/140))
                # define indices based on number of points:
                idx = (idx[0], idx[0] + idx[1])
                k_ad = np.zeros((nz, ny, nx))
                # 70S to 50S
                k_ad[:, :, :idx[0]] = 0.6
                k_ad[251 <= z, :, :idx[0]] = 0.45 
                k_ad[500 <= z, :, :idx[0]] = 0.3
                # 50S to 70N
                k_ad[:, :, idx[0]:] = 1.
                k_ad[251 <= z, :, idx[0]: ]= 0.75 
                k_ad[500 <= z, :, idx[0]:] = 0.5
                # desorption: constant in lat, lon, and depth
                k_de = np.ones((nz, ny, nx))
                # production: constant in lat, lon, and depth
                Q = 0.0267
	return k_ad, k_de, Q	
	
