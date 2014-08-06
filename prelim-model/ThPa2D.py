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

	def __init__(self, nx, nz, ng, dt, xmin = 0, xmax = 1e6, zmin = 0, 
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
                self.dt = dt
		# storage for the solution 
		self.a = np.zeros((nz, nx), dtype=np.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return np.zeros((self.nz, self.nx), dtype=np.float64)

	def fillBCs_d(self, k_ad, Q):    
		self.a[self.ilo, :] = self.a[self.ilo, :] + (Q - k_ad[0, :]*self.a[self.ilo, :] ) * self.dt
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
                g.a += ( Q - k_ad * g.a + k_de * h.a + adscheme(g, u, p_upz_d, n_upz_d, p_upx, n_upx, sinkrate = 0) ) * g.dt

                # particulate:
                h.a += ( k_ad * g.a - k_de * h.a + adscheme(h, u, p_upz_p, n_upz_p, p_upx, n_upx, sinkrate = S) ) * h.dt
                
                g.fillBCs_d(k_ad, Q)
                h.fillBCs_p()

                t += g.dt

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
        # extract velocity
        uz = u[0,:,:]
        ux = u[1,:,:]
        # upstream flux: C m/s
        fluxx_up = np.zeros((nz, nx));         fluxz_up = np.zeros((nz, nx))
        # define entries 0:nx-2 inclusive using n_upz etc.
        fluxx_up[:, 0:nx-1] = ux[:, 0:nx-1]*conc.a[:, 0:nx-1] * n_upx[:, 0:nx-1] + ux[:, 1:nx]*conc.a[:, 1:nx] * p_upx[:, 0:nx-1]                               # C m/s
        fluxz_up[0:nz-1, :] = (sinkrate + uz[0:nz-1, :])*conc.a[0:nz-1, :] * n_upz[0:nz-1, :] + (sinkrate + uz[1:nz, :])*conc.a[1:nz, :] * p_upz[0:nz-1, :]     # C m/s
        # simlulate settling in the bottom boundary (should be >0)
        fluxz_up[nz-1, :] = 2*fluxz_up[nz-2, :] - fluxz_up[nz-3, :]
        # d(conc)/dt according to upstream scheme (on the grid points)
        dtau_up_dt = np.zeros((nz, nx))
        dtau_up_dt[1:nz, 1:nx] = (fluxx_up[1:nz, 0:nx-1] - fluxx_up[1:nz, 1:nx]) * conc.dx_i + (fluxz_up[0:nz-1, 1:nx] - fluxz_up[1:nz, 1:nx])  * conc.dz_i 
        dtau_up_dt[0, 1:nx] = (fluxx_up[0, 0:nx-1] - fluxx_up[0, 1:nx]) * conc.dx_i - fluxz_up[0, 1:nx]  * conc.dz_i
        dtau_up_dt[1:nz, 0] = -fluxx_up[1:nz, 0] * conc.dx_i + (fluxz_up[0:nz-1, 0] - fluxz_up[1:nz, 0])  * conc.dz_i
        dtau_up_dt[0, 0] = - fluxx_up[0, 0] * conc.dx_i - fluxz_up[0, 0]  * conc.dz_i
        # new concentration based on upstream scheme
        tau_up = conc.a + dtau_up_dt * conc.dt
        # centred flux
        fluxx_cen = np.zeros((nz, nx));         fluxz_cen = np.zeros((nz, nx))
        fluxx_cen[:, 0:nx-1] = 0.5 * ( conc.a[:, 0:nx-1]*ux[:, 0:nx-1] + conc.a[:, 1:nx]*ux[:, 1:nx] ) 
        fluxz_cen[0:nz-1, :] = 0.5 * ( conc.a[0:nz-1, :]*(sinkrate + uz[0:nz-1, :]) + conc.a[1:nz, :]*(sinkrate + uz[1:nz, :] ))
        # flux at bottom boundary
        fluxz_cen[nz-1, :] = 2*fluxz_cen[nz-2, :] - fluxz_cen[nz-3, :]
        # anti-diffusive flux
        adfx = fluxx_cen - fluxx_up                     # C*velocity
        adfz = fluxz_cen - fluxz_up
        # calculate max/min values in neighbourhood 
        conc3=np.zeros((5,nz,nx)); tau3=np.zeros((5,nz,nx))
        # open domain
        conc3[0,1:nz-1,1:nx-1] = conc.a[1:nz-1,0:nx-2]
        conc3[1,1:nz-1,1:nx-1] = conc.a[1:nz-1,1:nx-1]
        conc3[2,1:nz-1,1:nx-1] = conc.a[1:nz-1,2:nx]
        conc3[3,1:nz-1,1:nx-1] = conc.a[0:nz-2,1:nx-1]
        conc3[4,1:nz-1,1:nx-1] = conc.a[2:nz,1:nx-1]
        tau3[0,1:nz-1,1:nx-1] = tau_up[1:nz-1,0:nx-2]
        tau3[1,1:nz-1,1:nx-1] = tau_up[1:nz-1,1:nx-1]
        tau3[2,1:nz-1,1:nx-1] = tau_up[1:nz-1,2:nx]
        tau3[3,1:nz-1,1:nx-1] = tau_up[0:nz-2,1:nx-1]
        tau3[4,1:nz-1,1:nx-1] = tau_up[2:nz,1:nx-1]   # C
        # on bounds
        # x = 0
        conc3[0,1:nz-1, 0] = conc.a[0:nz-2, 0]
        conc3[1,1:nz-1, 0] = conc.a[1:nz-1, 0]
        conc3[2,1:nz-1, 0] = conc.a[2:nz, 0]
        conc3[3,1:nz-1, 0] = conc.a[1:nz-1, 1]
        conc3[4,1:nz-1, 0] = conc.a[1:nz-1, 1]
        tau3[0,1:nz-1, 0] = tau_up[0:nz-2, 0]
        tau3[1,1:nz-1, 0] = tau_up[1:nz-1, 0]
        tau3[2,1:nz-1, 0] = tau_up[2:nz, 0]
        tau3[3,1:nz-1, 0] = tau_up[1:nz-1, 1]
        tau3[4,1:nz-1, 0] = tau_up[1:nz-1, 1]
        # x = xmax
        conc3[0,1:nz-1, nx-1] = conc.a[0:nz-2, nx-1]
        conc3[1,1:nz-1, nx-1] = conc.a[1:nz-1, nx-1]
        conc3[2,1:nz-1, nx-1] = conc.a[2:nz, nx-1]
        conc3[3,1:nz-1, nx-1] = conc.a[1:nz-1, nx-2]
        conc3[4,1:nz-1, nx-1] = conc.a[1:nz-1, nx-2]
        tau3[0,1:nz-1, nx-1] = tau_up[0:nz-2, nx-1]
        tau3[1,1:nz-1, nx-1] = tau_up[1:nz-1, nx-1]
        tau3[2,1:nz-1, nx-1] = tau_up[2:nz, nx-1]
        tau3[3,1:nz-1, nx-1] = tau_up[1:nz-1, nx-2]
        tau3[4,1:nz-1, nx-1] = tau_up[1:nz-1, nx-2]
        # z = 0
        conc3[0,0,1:nx-1] = conc.a[0,0:nx-2]
        conc3[1,0,1:nx-1] = conc.a[0,1:nx-1]
        conc3[2,0,1:nx-1] = conc.a[0,2:nx]
        conc3[3,0,1:nx-1] = conc.a[1,1:nx-1]
        conc3[4,0,1:nx-1] = conc.a[1,1:nx-1]
        tau3[0,0,1:nx-1] = tau_up[0,0:nx-2]
        tau3[1,0,1:nx-1] = tau_up[0,1:nx-1]
        tau3[2,0,1:nx-1] = tau_up[0,2:nx]
        tau3[3,0,1:nx-1] = tau_up[1,1:nx-1]
        tau3[4,0,1:nx-1] = tau_up[1,1:nx-1]
        # z = zmax
        conc3[0,nz-1,1:nx-1] = conc.a[nz-1,0:nx-2]
        conc3[1,nz-1,1:nx-1] = conc.a[nz-1,1:nx-1]
        conc3[2,nz-1,1:nx-1] = conc.a[nz-1,2:nx]
        conc3[3,nz-1,1:nx-1] = conc.a[nz-2,1:nx-1]
        conc3[4,nz-1,1:nx-1] = conc.a[nz-2,1:nx-1]
        tau3[0,nz-1,1:nx-1] = tau_up[nz-1,0:nx-2]
        tau3[1,nz-1,1:nx-1] = tau_up[nz-1,1:nx-1]
        tau3[2,nz-1,1:nx-1] = tau_up[nz-1,2:nx]
        tau3[3,nz-1,1:nx-1] = tau_up[nz-2,1:nx-1]
        tau3[4,nz-1,1:nx-1] = tau_up[nz-2,1:nx-1]
        # top left
        conc3[0,0,0] = conc.a[0,0]
        conc3[1,0,0] = conc.a[0,1]
        conc3[2,0,0] = conc.a[1,0]
        conc3[3,0,0] = conc.a[1,0]
        conc3[4,0,0] = conc.a[1,0]
        tau3[0,0,0] = tau_up[0,0]
        tau3[1,0,0] = tau_up[0,1]
        tau3[2,0,0] = tau_up[1,0]
        tau3[3,0,0] = tau_up[1,0]
        tau3[4,0,0] = tau_up[1,0]
        # top right
        conc3[0,0,nx-1] = conc.a[0,nx-1]
        conc3[1,0,nx-1] = conc.a[0,nx-2]
        conc3[2,0,nx-1] = conc.a[1,nx-1]
        conc3[3,0,nx-1] = conc.a[1,nx-1]
        conc3[4,0,nx-1] = conc.a[1,nx-1]
        tau3[0,0,nx-1] = tau_up[0,nx-1]
        tau3[1,0,nx-1] = tau_up[0,nx-2]
        tau3[2,0,nx-1] = tau_up[1,nx-1]
        tau3[3,0,nx-1] = tau_up[1,nx-1]
        tau3[4,0,nx-1] = tau_up[1,nx-1]
        # bottom left
        conc3[0,nz-1,0] = conc.a[nz-1,0]
        conc3[1,nz-1,0] = conc.a[nz-1,1]
        conc3[2,nz-1,0] = conc.a[nz-2,0]
        conc3[3,nz-1,0] = conc.a[nz-2,0]
        conc3[4,nz-1,0] = conc.a[nz-2,0]
        tau3[0,nz-1,0] = tau_up[nz-1,0]
        tau3[1,nz-1,0] = tau_up[nz-1,1]
        tau3[2,nz-1,0] = tau_up[nz-2,0]
        tau3[3,nz-1,0] = tau_up[nz-2,0]
        tau3[4,nz-1,0] = tau_up[nz-2,0]
        # bottom right
        conc3[0,nz-1,nx-1] = conc.a[nz-1,nx-1]
        conc3[1,nz-1,nx-1] = conc.a[nz-1,nx-2]
        conc3[2,nz-1,nx-1] = conc.a[nz-2,nx-1]
        conc3[3,nz-1,nx-1] = conc.a[nz-2,nx-1]
        conc3[4,nz-1,nx-1] = conc.a[nz-2,nx-1]
        tau3[0,nz-1,nx-1] = tau_up[nz-1,nx-1]
        tau3[1,nz-1,nx-1] = tau_up[nz-1,nx-2]
        tau3[2,nz-1,nx-1] = tau_up[nz-2,nx-1]
        tau3[3,nz-1,nx-1] = tau_up[nz-2,nx-1]
        tau3[4,nz-1,nx-1] = tau_up[nz-2,nx-1]
        # take minimum along 0-axis
        conc_up = np.maximum(np.amax(conc3,axis=0),np.amax(tau3,axis=0))   # C
        conc_do = np.minimum(np.amin(conc3,axis=0),np.amin(tau3,axis=0))   # C
        # define anti-diffusive influx and outflux in x
        xpos = np.zeros((nz, nx)); xneg = np.zeros((nz, nx))  
        nfluxx = np.sign(adfx)*0.5*(np.sign(adfx) - 1)                          # dimensionless
        pfluxx = np.sign(adfx)*0.5*(np.sign(adfx) + 1)
        xpos[0:nz, 1:nx] = pfluxx[0:nz, 0:nx-1] * adfx[0:nz, 0:nx-1] - nfluxx[0:nz, 1:nx] * adfx[0:nz, 1:nx]    # conc*velocity
        xpos[0:nz, 0] = - nfluxx[0:nz, 0] * adfx[0:nz, 0]
        xneg[0:nz, 1:nx] = pfluxx[0:nz, 1:nx] * adfx[0:nz, 1:nx] - nfluxx[0:nz, 0:nx-1] * adfx[0:nz, 0:nx-1]
        xneg[0:nz, 0] = pfluxx[0:nz, 0] * adfx[0:nz, 0] 
        # anti-diffusive influx and outflux in z 
        zpos = np.zeros((nz, nx)); zneg = np.zeros((nz, nx))
        nfluxz = np.sign(adfz)*0.5*(np.sign(adfz) - 1)
        pfluxz = np.sign(adfz)*0.5*(np.sign(adfz) + 1)
        zpos[1:nz, 0:nx] = pfluxz[0:nz-1, 0:nx] * adfz[0:nz-1, 0:nx] - nfluxz[1:nz, 0:nx] * adfz[1:nz, 0:nx]    # conc*velocity
        zpos[0, 0:nx] = - nfluxz[0, 0:nx] * adfz[0, 0:nx]
        zneg[1:nz, 0:nx] = pfluxz[1:nz, 0:nx] * adfz[1:nz, 0:nx] - nfluxz[0:nz-1, 0:nx] * adfz[0:nz-1, 0:nx]
        zneg[0, 0:nx] = pfluxz[0, 0:nx] * adfz[0, 0:nx]
        # total influx/outflux
        fpos = xpos*conc.dx_i + zpos*conc.dz_i              # units: concentration/time
        fneg = xneg*conc.dx_i + zneg*conc.dz_i
        # non dimensional Zalesak parameter 
        vsmall = 1e-12
        # = (max_conc - upstream_conc) / influx
        betaup = (conc_up - tau_up) / (fpos*conc.dt + vsmall)
        # = (upstream_conc - min_conc) / outflux
        betado = (tau_up - conc_do) / (fneg*conc.dt + vsmall)
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



