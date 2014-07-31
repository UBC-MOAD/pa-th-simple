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

	def fillBCs_d(self, k_ad, Q, dt):    
		self.a[self.ilo, :] = self.a[self.ilo, :] + (Q - k_ad[0, :]*self.a[self.ilo, :] ) * dt     
		#self.a[self.ilo, :] = 2*self.a[self.ilo + 1, :] - self.a[self.ilo + 2, :]
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
	uz = u[0, :, :]
	ux = u[1, :, :]

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
                g.a += ( Q - k_ad * g.a + k_de * h.a + adscheme(g, u, p_upz_d, n_upz_d, p_upx, n_upx, sinkrate = 0) ) * dt

                # particulate:
                h.a += ( k_ad * g.a - k_de * h.a + adscheme(h, u, p_upz_p, n_upz_p, p_upx, n_upx, sinkrate = S) ) * dt
                
                g.fillBCs_d(k_ad, Q, dt)
                h.fillBCs_p()

                t += dt

        return g, h


def flux(conc, u, p_upz, n_upz, p_upx, n_upx, sinkrate):
        """Flux based advection scheme
        """
	nz, nx = conc.nz, conc.nx
	
	vert_flux = conc.a * (u[0, :, :] + sinkrate)
	horz_flux = conc.a * u[1, :, :]
	
	left_flux =  n_upx[1:nz-1, 0:nx-2] * ( horz_flux[1:nz-1, 0:nx-2] - horz_flux[1:nz-1, 1:nx-1] )
	right_flux = p_upx[1:nz-1, 1:nx-1] * ( horz_flux[1:nz-1, 1:nx-1] - horz_flux[1:nz-1, 2:nx] )
	up_flux = n_upz[0:nz-2, 1:nx-1] * ( vert_flux[0:nz-2, 1:nx-1] - vert_flux[1:nz-1, 1:nx-1] )
        down_flux = p_upz[1:nz-1, 1:nx-1] * ( vert_flux[1:nz-1, 1:nx-1] - vert_flux[2:nz, 1:nx-1] )
		    
	adv = np.zeros((conc.nz, conc.nx))
        adv[1:nz-1, 1:nx-1] = (left_flux + right_flux) * conc.dx_i + (up_flux + down_flux) * conc.dz_i
                 
        return adv


def upstream(conc, u, p_upz, n_upz, p_upx, n_upx, sinkrate):

        # extract indices
        nz = conc.nz
        nx = conc.nx

        # extract the velocities
        uz = u[0, :, :] + sinkrate
        ux = u[1, :, :]

        # define fluxes
        adv = np.empty_like(conc.a)

        left_grad = n_upx[1:nz-1, 0:nx-2]*(conc.a[1:nz-1, 0:nx-2] - conc.a[1:nz-1, 1:nx-1])
        right_grad = p_upx[1:nz-1, 1:nx-1]*(conc.a[1:nz-1, 1:nx-1] - conc.a[1:nz-1, 2:nx])
        up_grad = n_upz[0:nz-2, 1:nx-1]*(conc.a[0:nz-2, 1:nx-1] - conc.a[1:nz-1, 1:nx-1])
        down_grad = p_upz[1:nz-1, 1:nx-1]*(conc.a[1:nz-1, 1:nx-1] - conc.a[2:nz, 1:nx-1])

        # dissolved advective term:
        adv[1:nz-1, 1:nx-1] = sinkrate * ( up_grad + down_grad ) * conc.dz_i + ux[1:nz-1, 1:nx-1] * ( left_grad + right_grad ) * conc.dx_i + uz[1:nz-1, 1:nx-1] * ( up_grad + down_grad ) * conc.dz_i 

        return adv


def TVD(conc, u, p_upz, n_upz, p_upx, n_upx, sinkrate):
        """total variance diminishing advection scheme
        """
        """total variance diminishing advection scheme
        """
        # grid
        nz = conc.nz
        nx = conc.nx
        dt = 0.001
        # extract velocity
        uz = u[0,:,:]
        ux = u[1,:,:]

        # upstream flux: C m/s
        fluxx_up = np.zeros((nz, nx));         fluxz_up = np.zeros((nz, nx))
        # define entries 0:nx-2 inclusive using n_upz etc.
        fluxx_up[:, 0:nx-1] = ux[:, 0:nx-1]*conc.a[:, 0:nx-1] * n_upx[:, 0:nx-1] + ux[:, 1:nx]*conc.a[:, 1:nx] * p_upx[:, 0:nx-1]     # C m/s
        fluxz_up[0:nz-1, :] = (sinkrate + uz[0:nz-1, :])*conc.a[0:nz-1, :] * n_upz[0:nz-1, :] + (sinkrate + uz[1:nz, :])*conc.a[1:nz, :] * p_upz[0:nz-1, :]    # C m/s
        # simlulate settling in the bottom boundary (should be >0)
        fluxz_up[nz-1, :] = 2*fluxz_up[nz-2, :] - fluxz_up[nz-3, :]

        # d(conc)/dt according to upstream scheme (on the grid points)
        dtau_up_dt = np.zeros((nz, nx))
        dtau_up_dt[1:nz, 1:nx] = (fluxx_up[1:nz, 0:nx-1] - fluxx_up[1:nz, 1:nx]) * conc.dx_i + (fluxz_up[0:nz-1, 1:nx] - fluxz_up[1:nz, 1:nx])  * conc.dz_i 
        dtau_up_dt[0, 1:nx] = (fluxx_up[0, 0:nx-1] - fluxx_up[0, 1:nx]) * conc.dx_i - fluxz_up[0, 1:nx]  * conc.dz_i
        dtau_up_dt[1:nz, 0] = -fluxx_up[1:nz, 0] * conc.dx_i + (fluxz_up[0:nz-1, 0] - fluxz_up[1:nz, 0])  * conc.dz_i
        dtau_up_dt[0, 0] = - fluxx_up[0, 0] * conc.dx_i - fluxz_up[0, 0]  * conc.dz_i
        # new concentration based on upstream scheme
        tau_up = conc.a + dtau_up_dt * dt

        # centred flux
        fluxx_cen = np.zeros((nz, nx));         fluxz_cen = np.zeros((nz, nx))
        fluxx_cen[:, 0:nx-1] = 0.5 * ( conc.a[:, 0:nx-1]*ux[:, 0:nx-1] + conc.a[:, 1:nx]*ux[:, 1:nx] ) 
        fluxz_cen[0:nz-1, :] = 0.5 * ( conc.a[0:nz-1, :]*(sinkrate + uz[0:nz-1, :]) + conc.a[1:nz, :]*(sinkrate + uz[1:nz, :] ))
        # flux at bottom boundary
        fluxz_cen[nz-1, :] = 2*fluxz_cen[nz-2, :] - fluxz_cen[nz-3, :]
        # anti-diffusive flux
        adfx = fluxx_cen - fluxx_up                     # conc*velocity
        adfz = fluxz_cen - fluxz_up

        # max and min concentrations in region
        conc_up = np.zeros((nz, nx)); xdo = np.zeros((nz, nx))
        conc_do = np.zeros((nz, nx)); zdo = np.zeros((nz, nx))

        # max and min concentrations in region
        conc_up = np.empty_like(conc.a)
        xdo = np.empty_like(conc.a)
        conc_do = np.empty_like(conc.a)
        zdo = np.empty_like(conc.a)
        # max/min in middle domain
        j = 1
        i = 1
        while j <=nx - 1:
                while i <= nz - 1:
                        conc_up[i, j] = max( np.max(conc.a[i-1:i+2, j-1:j+2]), np.max(tau_up[i-1:i+2, j-1:j+2]) ) 
                        conc_do[i, j] = min( np.min(conc.a[i-1:i+2, j-1:j+2]), np.min(tau_up[i-1:i+2, j-1:j+2]) )
                        i += 1
                j += 1
        # max/min on x-bounds
        for i in range(1, nz - 1):
                conc_up[i, 0] = max( np.max(conc.a[i-1:i+2, 0:2]), np.max(tau_up[i-1:i+2, 0:2]))
                conc_up[i, nx-1] = max( np.max(conc.a[i-1:i+2, nx-2:nx]), np.max(tau_up[i-1:i+2, nx-2:nx]))
                conc_do[i, 0] = min( np.min(conc.a[i-1:i+2, 0:2]), np.min(tau_up[i-1:i+2, 0:2]))
                conc_do[i, nx-1] = min( np.min(conc.a[i-1:i+2, nx-2:nx]), np.min(tau_up[i-1:i+2, nx-2:nx]))
            
        for j in range(1, nx - 1):
                conc_up[0, j] = max( np.max(conc.a[0:2, j-1:j+2]), np.max(tau_up[0:2, j-1:j+2]))
                conc_up[nz-1, j] = max( np.max(conc.a[nz-2:nz, j-1:j+2]), np.max(tau_up[nz-2:nz, j-1:j+2]))
                conc_do[0, j] = min( np.min(conc.a[0:2, j-1:j+2]), np.min(tau_up[0:2, j-1:j+2]))
                conc_do[nz-1, j] = min( np.min(conc.a[nz-2:nz, j-1:j+2]), np.min(tau_up[nz-2:nz, j-1:j+2]))
            
        conc_up[0, 0] = max( np.max(conc.a[0:2, 0:2]), np.max(tau_up[0:2, 0:2]))
        conc_up[0, nx-1] = max( np.max(conc.a[0:2, nx-2:nx]), np.max(tau_up[0:2, nx-2:nx]))
        conc_up[nz-1, 0] = max( np.max(conc.a[nz-2:nz, 0:2]), np.max(tau_up[nz-2:nz, 0:2]))
        conc_up[nz-1, nx-1] = max( np.max(conc.a[nz-2:nz, nx-2:nx]), np.max(tau_up[nz-2:nz, nx-2:nx]))

        conc_do[0, 0] = min( np.min(conc.a[0:2, 0:2]), np.min(tau_up[0:2, 0:2]))
        conc_do[0, nx-1] = min( np.min(conc.a[0:2, nx-2:nx]), np.min(tau_up[0:2, nx-2:nx]))
        conc_do[nz-1, 0] = min( np.min(conc.a[nz-2:nz, 0:2]), np.min(tau_up[nz-2:nz, 0:2]))
        conc_do[nz-1, nx-1] = min( np.min(conc.a[nz-2:nz, nx-2:nx]), np.min(tau_up[nz-2:nz, nx-2:nx]))

        # +/- adfx
        xpos = np.empty_like(conc.a)
        xneg = np.empty_like(conc.a) 
        nfluxx = np.sign(adfx)*0.5*(np.sign(adfx) - 1)                                          # dimensionless
        pfluxx = np.sign(adfx)*0.5*(np.sign(adfx) + 1)
        xpos[:, 1:nx] = pfluxx[:, 0:nx-1] * adfx[:, 0:nx-1] - nfluxx[:, 1:] * adfx[:, 1:]   # conc*velocity
        xpos[:, 0] = - nfluxx[:, 0] * adfx[:, 0]
        xneg[:, 1:] = pfluxx[:, 1:] * adfx[:, 1:] - nfluxx[:, 0:nx-1] * adfx[:, 0:nx-1]
        xneg[:, 0] = pfluxx[:, 0] * adfx[:, 0] 
        # +/- adfz
        zpos = np.empty_like(conc.a) 
        zneg = np.empty_like(conc.a)
        nfluxz = np.sign(adfz)*0.5*(np.sign(adfz) - 1)
        pfluxz = np.sign(adfz)*0.5*(np.sign(adfz) + 1)
        zpos[1:, :] = pfluxz[0:nz-1, :] * adfz[0:nz-1, :] - nfluxz[1:, :] * adfz[1:, :]   # conc*velocity
        zpos[0, :] = - nfluxz[0, :] * adfz[0, :]
        zneg[1:, :] = pfluxz[1:, :] * adfz[1:, :] - nfluxz[0:nz-1, :] * adfz[0:nz-1, :]
        zneg[0, :] = pfluxz[0, :] * adfz[0, :]
        # +/- adf
        fpos = xpos/conc.dx + zpos/conc.dz                                                      # units: concentration/time
        fneg = xneg/conc.dx + zneg/conc.dz
        # total influx/outflux

        # non dimensional Zalesak parameter 
        vsmall = 1e-12
        # = (max_conc - upstream_conc) / influx
        fpos_i = fpos*dt + vsmall
        betaup = (conc_up - tau_up) * fpos_i
        # = (upstream_conc - min_conc) / outflux
        fneg_i = fneg*dt + vsmall
        betado = (tau_up - conc_do) * fneg_i
        # flux limiters
        zaux = np.zeros((nz,nx))
        zaux[:,0:nx-1] = np.minimum(np.ones(nx-1), np.minimum(betado[:,:nx-1], betaup[:,1:]))
        zbux = np.zeros((nz,nx))
        zbux[:,0:nx-1] = np.minimum(np.ones(nx-1), np.minimum(betaup[:,:nx-1], betado[:,1:]))
        zcux = (0.5 + 0.5*np.sign(adfx))
        zauz = np.zeros((nz,nx))
        zauz[0:nz-1,:] = np.minimum(np.ones((nz-1, nx)),np.minimum(betado[0:nz-1,:], betaup[1:,:]))
        zbuz = np.zeros((nz,nx))
        zbuz[0:nz-1,:] = np.minimum(np.ones((nz-1, nx)),np.minimum(betaup[0:nz-1,:], betado[1:,:]))
        zcuz = (0.5 + 0.5*np.sign(adfz))

        # calculate TVD flux in x and z
        aaz = adfz * (zcuz * zauz + (1-zcuz)*zbuz)                                                   # C m/s on flux points
        aax = adfx * (zcux * zaux + (1-zcux)*zbux)

        # final sol.
        adv = np.zeros((nz, nx))
        adv[1:nz, 1:nx] = dtau_up_dt[1:nz, 1:nx] +  (aax[1:nz, 0:nx-1] - aax[1:nz, 1:nx]) * conc.dx_i + (aaz[0:nz-1, 1:nx] - aaz[1:nz, 1:nx]) * conc.dz_i
               
        return adv


def k_sorp(string, zmin, zmax, nx, nz):
	""" Computes adsorption,desorption, & production constants for either Th or Pa
	:arg string: a string, either 'Th' or 'Pa'
	:arg xmin: minimum x on the grid
	:arg xmax: maximum x on the grid
	:arg zmin: minimum z on the grid
	:arg zmax: maximum z on the grid
	:arg nx: number of points in x dimension
	:arg nz: number of points in z dimension
	"""
        # spatial coordinates
        dz = (zmax - zmin) / (nz - 1)
        z = zmin + (np.arange(nz) - 1) * dz

        if string == 'Pa':
                # define number of points per region
                idx = (np.round((70 - 47.5)*nx/140), np.round((47.5 - 45)*nx/140), np.round((45 - 42.5)*nx/140), np.round((42.5 + 70)*nx/140))
                # define indices based on number of points:
                idx = (idx[0], idx[0] + idx[1], idx[0] + idx[1] + idx[2], idx[0] + idx[1] + idx[2] + idx[3])
                k_ad = np.ones((nz, nx))
                # 70S to 47.5S
                k_ad[:, :idx[0]] = 0.44
                k_ad[251 <= z, :idx[0]] = 0.33 
                k_ad[500 <= z, :idx[0]] = 0.22
                # 47.5S to 45S
                k_ad[:, idx[0]:idx[1]] = 0.3
                k_ad[251 <= z, idx[0]:idx[1]] = 0.225 
                k_ad[500 <= z, idx[0]:idx[1]] = 0.15
                # 45S to 42.5S
                k_ad[:, idx[1]:idx[2]] = 0.2
                k_ad[251 <= z, idx[1]:idx[2]] = 0.15
                k_ad[500 <= z, idx[1]:idx[2]] = 0.1
                #42.5S to 70N
                k_ad[:, idx[2]:] = 0.08
                k_ad[251 <= z, idx[2]:] = 0.06
                k_ad[500 <= z, idx[2]:] = 0.04
                # desorption: constant in latitude and depth
                k_de = np.zeros((nz, nx))
                k_de[:] = 1.6
                # production: constant in latitude and depth
                Q = 0.00246

        if string == 'Th':
                # define number of points in each region
                idx = (np.round((70 - 50)*nx/140), np.round((50 + 70)*nx/140))
                # define indices based on number of points:
                idx = (idx[0], idx[0] + idx[1])
                k_ad = np.zeros((nz, nx))
                # 70S to 50S
                k_ad[:, :idx[0]] = 0.6
                k_ad[251 <= z, :idx[0]] = 0.45 
                k_ad[500 <= z, :idx[0]] = 0.3
                # 50S to 70N
                k_ad[:, idx[0]:] = 1.
                k_ad[251 <= z, idx[0]: ]= 0.75 
                k_ad[500 <= z, idx[0]:] = 0.5

                k_de = np.ones((nz, nx))

                Q = 0.0267
	return k_ad, k_de, Q	



