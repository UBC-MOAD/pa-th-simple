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
                g.a += ( Q - k_ad * g.a + k_de * h.a 
                             + adscheme(g, u, p_upz_d, n_upz_d, p_upx, n_upx, sinkrate = 0) ) * dt

                # particulate:
                h.a += ( k_ad * g.a - k_de * h.a 
                             + adscheme(h, u, p_upz_p, n_upz_p, p_upx, n_upx, sinkrate = S) ) * dt


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
		    
	adv = np.empty_like(conc.a)
        adv[1:nz-1, 1:nx-1] = (left_flux + right_flux) * conc.dx_i + (up_flux + down_flux) * conc.dz_i
                 
        return adv


def upstream_d(g, u, p_upz, n_upz, p_upx, n_upx, gdx_i, gdz_i):

        # extract indices
        nz = g.nz
        nx = g.nx

        # extract the velocities
        uz = u[:, :, 0]
        ux = u[:, :, 1]

        # define fluxes
        d_adv = np.zeros(np.shape(g.a))

        # dissolved advective term:
        d_adv[1:nz-1, 1:nx-1] = ux[1:nz-1, 1:nx-1] * ( n_upx[1:nz-1, 0:nx-2]*(g.a[1:nz-1, 0:nx-2] - g.a[1:nz-1, 1:nx-1]) + p_upx[1:nz-1, 1:nx-1]*(g.a[1:nz-1, 1:nx-1] - g.a[1:nz-1, 2:nx]) ) * g.dx_i + uz[1:nz-1, 1:nx-1] * ( n_upz[0:nz-2, 1:nx-1]*(g.a[0:nz-2, 1:nx-1] - g.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(g.a[1:nz-1, 1:nx-1] - g.a[2:nz, 1:nx-1]) ) * g.dz_i 

        return d_adv




def upstream_p(h, u, p_upz, n_upz, p_upx, n_upx, hdx_i, hdz_i, S):

        # extract indices
        nz = h.nz
        nx = h.nx

        # extract the velocities
        uz = u[:, :, 0] + S
        ux = u[:, :, 1]

        # define fluxes
        p_adv = np.zeros(np.shape(h.a))

        p_adv[1:nz-1, 1:nx-1] = S * ( n_upz[1:nz-1, 1:nx-1]*(h.a[0:nz-2, 1:nx-1] - h.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[2:nz, 1:nx-1]) ) * h.dz_i + ux[1:nz-1, 1:nx-1] * ( n_upx[1:nz-1, 0:nx-2]*(h.a[1:nz-1, 0:nx-2] - h.a[1:nz-1, 1:nx-1]) + p_upx[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[1:nz-1, 2:nx]) ) * h.dx_i + uz[1:nz-1, 1:nx-1] * ( n_upz[0:nz-2, 1:nx-1]*(h.a[0:nz-2, 1:nx-1] - h.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[2:nz, 1:nx-1]) ) * h.dz_i 

        return p_adv





def TVD_d(g, u, p_upz, n_upz, p_upx, n_upx, gdx_i, gdz_i):
        """total variance diminishing advection scheme
        """
        # define grid
        nz = g.nz
        nx = g.nx

        # extract velocity
        uz = u[:,:,0]
        ux = u[:,:,1]

        # define upstream flux (in multi-directional flux field)
        fluxx_up = np.zeros((nz, nx))
        fluxx_up[1:nz-2, 1:nx-2] = ux[1:nz-2, 0:nx-3]*g.a[1:nz-2, 0:nx-3] * n_upx[1:nz-2, 1:nx-2] + ux[1:nz-2, 2:nx-1]*g.a[1:nz-2, 2:nx-1] * p_upx[1:nz-2, 1:nx-2]
        fluxz_up = np.zeros((nz, nx))
        fluxz_up[1:nz-2, 1:nx-2] = uz[0:nz-3, 1:nx-2]*g.a[0:nz-3, 1:nx-2] * n_upz[1:nz-2, 1:nx-2] + uz[2:nz-1, 1:nx-2]*g.a[2:nz-1, 1:nx-2] * p_upz[1:nz-2, 1:nx-2]

        # d(conc)/dt based on upstream-flux method
        tau_up = np.zeros((nz, nx))
        tau_up[1:nz-2, 1:nx-2] = ((fluxx_up[1:nz-2, 0:nx-3] - fluxx_up[1:nz-2, 1:nx-2]) * n_upx[1:nz-2, 1:nx-2] + (fluxx_up[1:nz-2, 1:nx-2] - fluxx_up[1:nz-2, 2:nx - 1]) * p_upx[1:nz-2, 1:nx-2]) * gdx_i + ((fluxz_up[0:nz-3, 1:nx-2] - fluxz_up[1:nz-2, 1:nx-2]) * n_upz[1:nz-2, 1:nx-2] + (fluxz_up[1:nz-2, 1:nx-2] - fluxz_up[2:nz - 1, 1:nx-2]) * p_upz[1:nz-2, 1:nx-2]) * gdz_i 

        # define centred flux
        fluxx_cen = np.zeros((nz, nx))
        fluxx_cen[1:nz-2, 1:nx-2] = 0.5 * ( g.a[1:nz-2, 0:nx-3]*ux[1:nz-2, 0:nx-3] - g.a[1:nz-2, 2:nx-1]*ux[1:nz-2, 2:nx-1] ) 
        fluxz_cen = np.zeros((nz, nx))
        fluxz_cen[1:nz-2, 1:nx-2] = 0.5 * ( g.a[0:nz-3, 1:nx-2]*uz[0:nz-3, 1:nx-2] - g.a[2:nz-1, 1:nx-2]*uz[2:nz-1, 1:nx-2] )

        # define anti-diffusive flux; shape =  [nz, nx]
        adfx = fluxx_cen - fluxx_up
        adfz = fluxz_cen - fluxz_up

        # max and min concentrations in region
        xup = np.zeros((nz, nx)); xdo = np.zeros((nz, nx))
        zup = np.zeros((nz, nx)); zdo = np.zeros((nz, nx))

        for j in range(1, nx - 1):
            xup[1:nz-1, j] = max( np.max(g.a[1:nz-1, j-1:j+1]), np.max(tau_up[1:nz-1, j-1:j+1]) )    # C
            xdo[1:nz-1, j] = min( np.min(g.a[1:nz-1, j-1:j+1]), np.min(tau_up[1:nz-1, j-1:j+1]) )    # C
        for i in range(1, nz - 1):
            zup[i, 1:nx-1] = max( np.max(g.a[i-1:i+1, 1:nx-1]), np.max(tau_up[i-1:i+1, 1:nx-1]) )    # C
            zdo[i, 1:nx-1] = min( np.min(g.a[i-1:i+1, 1:nx-1]), np.min(tau_up[i-1:i+1, 1:nx-1]) )    # C

        # define pos/neg fluxes in x  
        xpos = np.zeros((nz, nx)); xneg = np.zeros((nz, nx))  
        nfluxx = np.sign(adfx)*0.5*(np.sign(adfx) - 1)
        pfluxx = np.sign(adfx)*0.5*(np.sign(adfx) + 1)
        xpos[0:nz - 1, 1:nx - 1] = pfluxx[0:nz - 1, 0:nx - 2] * adfx[0:nz - 1, 0:nx - 2] - nfluxx[0:nz - 1, 1:nx - 1] * adfx[0:nz - 1, 1:nx - 1]
        xneg[0:nz - 1, 1:nx - 1] = pfluxx[0:nz - 1, 1:nx - 1] * adfx[0:nz - 1, 1:nx - 1] - nfluxx[0:nz - 1, 0:nx - 2] * adfx[0:nz - 1, 0:nx - 2]
        # define pos/neg fluxes in z 
        zpos = np.zeros((nz, nx)); zneg = np.zeros((nz, nx))
        nfluxz = np.sign(adfz)*0.5*(np.sign(adfz) - 1)
        pfluxz = np.sign(adfz)*0.5*(np.sign(adfz) + 1)
        zpos[1:nz - 1, 0:nx - 1] = pfluxz[0:nz - 2, 0:nx - 1] * adfz[0:nz - 2, 0:nx - 1] - nfluxz[1:nz - 1, 0:nx - 1] * adfz[1:nz - 1, 0:nx - 1]
        zneg[1:nz - 1, 0:nx - 1] = pfluxz[1:nz - 1, 0:nx - 1] * adfz[1:nz - 1, 0:nx - 1] - nfluxz[0:nz - 2, 0:nx - 1] * adfz[0:nz - 2, 0:nx - 1]

        # calculate the Zalesak parameter
        zbetaup = (zup - tau_up) / zpos #* dx/dt                                                 # C / (C m/s) * m/s = non dimensional
        zbetado = (tau_up - zdo) / zneg #* dx/dt
        xbetaup = (xup - tau_up) / xpos #* dz/dt                                                 # C / (C m/s) * m/s = non dimensional
        xbetado = (tau_up - xdo) / xneg #* dz/dt
        # remove nans
        zeros = np.zeros(np.shape(zbetaup))
        idx = np.isnan(zbetaup)
        zbetaup[idx] = 0
        idx = np.isnan(zbetado)
        zbetado[idx] = 0
        idx = np.isnan(xbetaup)
        xbetaup[idx] = 0
        idx = np.isnan(xbetado)
        xbetado[idx] = 0

        # calculate zau & xau
        # =one by default
        zau = np.ones((nz, nx))
        xau = np.ones((nz, nx))
        # =zbetado if zbetado < 1
        zau[zbetado < 1] = zbetado[zbetado < 1]                           # non dim and on the flux points
        xau[xbetado < 1] = xbetado[xbetado < 1]                           # non dim and on the flux points
        # shift zbetaup by one index
        zbetaup[0:nz - 1, 0:nx - 2] = zbetaup[0:nz - 1, 1:nx - 1]
        xbetaup[0:nz - 2, 0:nx - 1] = xbetaup[1:nz - 1, 0:nx - 1]
        # =zbetaup if zbetaup[:, j + 1] < zbetado[:, j]
        zau[zbetaup < zbetado] = zbetaup[zbetaup < zbetado]
        xau[xbetaup < xbetado] = xbetaup[xbetaup < xbetado]
        # set last column to zero since it's out of range
        zau[:, nx - 1] = 0
        xau[nz - 1, :] = 0

        # calculate zbu & xbu
        # =one by default
        zbu = np.ones((nz, nx))
        xbu = np.ones((nz, nx))
        # =zbetaup if zbetaup < 1
        zbu[zbetaup < 1] = zbetaup[zbetaup < 1]                           # non dim and on the flux points
        xbu[xbetaup < 1] = xbetaup[xbetaup < 1]                           # non dim and on the flux points
        # shift zbetado by one index
        zbetado[0:nz - 1, 0:nx - 2] = zbetado[0:nz - 1, 1:nx - 1]
        xbetado[0:nz - 2, 0:nx - 1] = xbetado[1:nz - 1, 0:nx - 1]
        # =zbetaup if zbetaup[:, j + 1] < zbetado[:, j]
        zbu[zbetado < zbetaup] = zbetaup[zbetado < zbetaup]
        xbu[xbetado < xbetaup] = xbetaup[xbetado < xbetaup]
        # set last column to zero since it's out of range
        zbu[:, nx - 1] = 0
        xbu[nz - 1, :] = 0

        # calculate zcu & xcu
        zcu = (0.5 + 0.5*np.sign(adfz))  
        xcu = (0.5 + 0.5*np.sign(adfx))  

        # calculate TVD flux in x and z
        daaz = adfz * (zcu * zau + (1-zcu)*zbu)                                                   # C m/s on flux points
        daax = adfx * (xcu * xau + (1-xcu)*xbu)

        # final sol.
        d_adv = np.zeros((nz, nx))
        d_adv[1:nz-1, 1:nx-1] = tau_up[1:nz-1, 1:nx-1] +  (daax[1:nz-1, 0:nx-2] - daax[1:nz-1, 1:nx-1]) * gdx_i + (daaz[0:nz-2, 1:nx-1] - daaz[1:nz-1, 1:nx-1]) * gdz_i 
                


        return d_adv

def TVD_p(h, u, p_upz, n_upz, p_upx, n_upx, hdx_i, hdz_i, S):
        """total variance diminishing advection scheme
        """
        # define grid
        nz = h.nz
        nx = h.nx

        # extract velocity
        uz = u[:,:,0]
        ux = u[:,:,1]

        # define upstream flux (in multi-directional flux field)
        fluxx_up = np.zeros(np.shape(h.a))
        fluxx_up[1:nz-2, 1:nx-2] = ux[1:nz-2, 0:nx-3]*h.a[1:nz-2, 0:nx-3] * n_upx[1:nz-2, 1:nx-2] + ux[1:nz-2, 2:nx-1]*h.a[1:nz-2, 2:nx-1] * p_upx[1:nz-2, 1:nx-2]
        fluxz_up = np.zeros(np.shape(h.a))
        fluxz_up[1:nz-2, 1:nx-2] = uz[0:nz-3, 1:nx-2]*h.a[0:nz-3, 1:nx-2] * n_upz[1:nz-2, 1:nx-2] + uz[2:nz-1, 1:nx-2]*h.a[2:nz-1, 1:nx-2] * p_upz[1:nz-2, 1:nx-2]

        # d(conc)/dt based on upstream-flux method
        tau_up = np.zeros(np.shape(h.a))
        tau_up[1:nz-2, 1:nx-2] = ((fluxx_up[1:nz-2, 0:nx-3] - fluxx_up[1:nz-2, 1:nx-2]) * n_upx[1:nz-2, 1:nx-2] + (fluxx_up[1:nz-2, 1:nx-2] - fluxx_up[1:nz-2, 2:nx - 1]) * p_upx[1:nz-2, 1:nx-2]) * hdx_i + ((fluxz_up[0:nz-3, 1:nx-2] - fluxz_up[1:nz-2, 1:nx-2]) * n_upz[1:nz-2, 1:nx-2] + (fluxz_up[1:nz-2, 1:nx-2] - fluxz_up[2:nz - 1, 1:nx-2]) * p_upz[1:nz-2, 1:nx-2]) * hdz_i 

        # define centred flux
        fluxx_cen = np.zeros((nz, nx))
        fluxx_cen[1:nz-2, 1:nx-2] = 0.5 * ( h.a[1:nz-2, 0:nx-3]*ux[1:nz-2, 0:nx-3] - h.a[1:nz-2, 2:nx-1]*ux[1:nz-2, 2:nx-1] ) 
        fluxz_cen = np.zeros((nz, nx))
        fluxz_cen[1:nz-2, 1:nx-2] = 0.5 * ( h.a[0:nz-3, 1:nx-2]*uz[0:nz-3, 1:nx-2] - h.a[2:nz-1, 1:nx-2]*uz[2:nz-1, 1:nx-2] )

        # define anti-diffusive flux; shape =  [nz, nx]
        adfx = fluxx_cen - fluxx_up
        adfz = fluxz_cen - fluxz_up

        # max and min concentrations in region
        xup = np.zeros((nz, nx)); xdo = np.zeros((nz, nx))
        zup = np.zeros((nz, nx)); zdo = np.zeros((nz, nx))

        for j in range(1, nx - 1):
            xup[1:nz-1, j] = max( np.max(h.a[1:nz-1, j-1:j+1]), np.max(tau_up[1:nz-1, j-1:j+1]) )    # C
            xdo[1:nz-1, j] = min( np.min(h.a[1:nz-1, j-1:j+1]), np.min(tau_up[1:nz-1, j-1:j+1]) )    # C
        for i in range(1, nz - 1):
            zup[i, 1:nx-1] = max( np.max(h.a[i-1:i+1, 1:nx-1]), np.max(tau_up[i-1:i+1, 1:nx-1]) )    # C
            zdo[i, 1:nx-1] = min( np.min(h.a[i-1:i+1, 1:nx-1]), np.min(tau_up[i-1:i+1, 1:nx-1]) )    # C

        # define pos/neg fluxes in x  
        xpos = np.zeros((nz, nx)); xneg = np.zeros((nz, nx))  
        nfluxx = np.sign(adfx)*0.5*(np.sign(adfx) - 1)
        pfluxx = np.sign(adfx)*0.5*(np.sign(adfx) + 1)
        xpos[0:nz - 1, 1:nx - 1] = pfluxx[0:nz - 1, 0:nx - 2] * adfx[0:nz - 1, 0:nx - 2] - nfluxx[0:nz - 1, 1:nx - 1] * adfx[0:nz - 1, 1:nx - 1]
        xneg[0:nz - 1, 1:nx - 1] = pfluxx[0:nz - 1, 1:nx - 1] * adfx[0:nz - 1, 1:nx - 1] - nfluxx[0:nz - 1, 0:nx - 2] * adfx[0:nz - 1, 0:nx - 2]
        # define pos/neg fluxes in z 
        zpos = np.zeros((nz, nx)); zneg = np.zeros((nz, nx))
        nfluxz = np.sign(adfz)*0.5*(np.sign(adfz) - 1)
        pfluxz = np.sign(adfz)*0.5*(np.sign(adfz) + 1)
        zpos[1:nz - 1, 0:nx - 1] = pfluxz[0:nz - 2, 0:nx - 1] * adfz[0:nz - 2, 0:nx - 1] - nfluxz[1:nz - 1, 0:nx - 1] * adfz[1:nz - 1, 0:nx - 1]
        zneg[1:nz - 1, 0:nx - 1] = pfluxz[1:nz - 1, 0:nx - 1] * adfz[1:nz - 1, 0:nx - 1] - nfluxz[0:nz - 2, 0:nx - 1] * adfz[0:nz - 2, 0:nx - 1]

        # calculate the Zalesak parameter
        zbetaup = (zup - tau_up) / zpos #* dx/dt                                                 # C / (C m/s) * m/s = non dimensional
        zbetado = (tau_up - zdo) / zneg #* dx/dt
        xbetaup = (xup - tau_up) / xpos #* dz/dt                                                 # C / (C m/s) * m/s = non dimensional
        xbetado = (tau_up - xdo) / xneg #* dz/dt
        # remove nans
        zeros = np.zeros(np.shape(zbetaup))
        idx = np.isnan(zbetaup)
        zbetaup[idx] = 0
        idx = np.isnan(zbetado)
        zbetado[idx] = 0
        idx = np.isnan(xbetaup)
        xbetaup[idx] = 0
        idx = np.isnan(xbetado)
        xbetado[idx] = 0

        # calculate zau & xau
        # =one by default
        zau = np.ones((nz, nx))
        xau = np.ones((nz, nx))
        # =zbetado if zbetado < 1
        zau[zbetado < 1] = zbetado[zbetado < 1]                           # non dim and on the flux points
        xau[xbetado < 1] = xbetado[xbetado < 1]                           # non dim and on the flux points
        # shift zbetaup by one index
        zbetaup[0:nz - 1, 0:nx - 2] = zbetaup[0:nz - 1, 1:nx - 1]
        xbetaup[0:nz - 2, 0:nx - 1] = xbetaup[1:nz - 1, 0:nx - 1]
        # =zbetaup if zbetaup[:, j + 1] < zbetado[:, j]
        zau[zbetaup < zbetado] = zbetaup[zbetaup < zbetado]
        xau[xbetaup < xbetado] = xbetaup[xbetaup < xbetado]
        # set last column to zero since it's out of range
        zau[:, nx - 1] = 0
        xau[nz - 1, :] = 0

        # calculate zbu & xbu
        # =one by default
        zbu = np.ones((nz, nx))
        xbu = np.ones((nz, nx))
        # =zbetaup if zbetaup < 1
        zbu[zbetaup < 1] = zbetaup[zbetaup < 1]                           # non dim and on the flux points
        xbu[xbetaup < 1] = xbetaup[xbetaup < 1]                           # non dim and on the flux points
        # shift zbetado by one index
        zbetado[0:nz - 1, 0:nx - 2] = zbetado[0:nz - 1, 1:nx - 1]
        xbetado[0:nz - 2, 0:nx - 1] = xbetado[1:nz - 1, 0:nx - 1]
        # =zbetaup if zbetaup[:, j + 1] < zbetado[:, j]
        zbu[zbetado < zbetaup] = zbetaup[zbetado < zbetaup]
        xbu[xbetado < xbetaup] = xbetaup[xbetado < xbetaup]
        # set last column to zero since it's out of range
        zbu[:, nx - 1] = 0
        xbu[nz - 1, :] = 0

        # calculate zcu & xcu
        zcu = (0.5 + 0.5*np.sign(adfz))  
        xcu = (0.5 + 0.5*np.sign(adfx))  

        # calculate TVD flux in x and z
        paaz = adfz * (zcu * zau + (1-zcu)*zbu)                                                   # C m/s on flux points
        paax = adfx * (xcu * xau + (1-xcu)*xbu)


        # final sol.
        p_adv = np.zeros((nz, nx))
        # try doing the subtraction without upstream discrimination first:
        p_adv[1:nz-1, 1:nx-1] = tau_up[1:nz-1, 1:nx-1] +  (paax[1:nz-1, 0:nx-2] - paax[1:nz-1, 1:nx-1]) * hdx_i + (paaz[0:nz-2, 1:nx-1] - paaz[1:nz-1, 1:nx-1]) * hdz_i 
                
        return p_adv   

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



