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

	def fillBCs(self, k_ad, Q, dt):          
		self.a[self.ilo, :] = self.a[self.ilo, :] + (Q - k_ad[0, :]*self.a[self.ilo, :] ) * dt                        # PDE		
                #self.a[self.ilo, :] = 2*self.a[self.ilo + 1, :] - self.a[self.ilo + 2, :]                              # interpolated
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

	def fillBCs(self, k_ad, Q, dt):  
		self.a[self.ilo, :] = self.a[self.ilo, :] + ( Q  - k_ad[0, :]*self.a[self.ilo, :] ) * dt                      # PDE 		
                #self.a[self.ilo, :] = 2*self.a[self.ilo + 1, :] - self.a[self.ilo + 2, :]                              # interpolated		
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
		

def adflow(g, h, t, T, u, k_ad, k_de, Q, adscheme_d, adscheme_p):
        """
        Compute and store the dissolved and particulate [Th] profiles, 
        write them to a file, plot the results.

        :arg t: scale for time at which code is initiated
        :type t: int

        :arg T: scale for time at which code is terminated
        :typeT: int

        :arg V: scale for ux, uz, which are originally order 1.
        :type V: int

        :arg u: 3D tensor of shape (nz, nx, 2), z component of velocity in (:, :, 0), x component of velocity in (:, :, 1)
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
        S_i = 1/S

         # time info (yr)
        dt = 0.001
        t = t * (g.zmax - g.zmin)*S_i
        T = T * (g.zmax - g.zmin)*S_i

        # evolution loop
        anew = g.a
        bnew = h.a

        # extract the velocities
        uz = u[:, :, 0]
        ux = u[:, :, 1]

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
        gdx_i = 1/g.dx
        gdz_i = 1/g.dz
        hdx_i = 1/h.dx
        hdz_i = 1/h.dz

	# vectorize spatial indices 
        i = np.arange(g.ilo + 1, g.ihi, 1, dtype = int)
        j = np.arange(g.jlo + 1, g.jhi, 1, dtype = int)

        while (t < T):

                # dissolved:
                anew = g.a + ( Q - k_ad * g.a + k_de * h.a + adscheme_d(g, u, p_upz_d, n_upz_d, p_upx, n_upx, gdx_i, gdz_i) ) * dt

                # particulate:
                bnew = h.a + ( k_ad * g.a - k_de * h.a + adscheme_p(h, u, p_upz_p, n_upz_p, p_upx, n_upx, hdx_i, hdz_i, S) ) * dt

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]

                # fill boundary conditions
                g.fillBCs(k_ad, Q, dt)
                h.fillBCs()

                t += dt

        return g, h




def flux_d(g, u, p_upz, n_upz, p_upx, n_upx, gdx_i, gdz_i):
        """Flux based advection scheme
        """
        # extract indices
        nz = g.nz
        nx = g.nx

        # extract the velocities
        uz = u[:, :, 0]
        ux = u[:, :, 1]

        # define fluxes
        d_adv = np.zeros(np.shape(g.a))
        fluxx = g.a * ux
        fluxz = g.a * uz

        # dissolved advective term:
        d_adv[1:nz-1,1:nx-1] = ( (n_upx[1:nz-1, 0:nx-2]*(fluxx[1:nz-1, 0:nx-2] - fluxx[1:nz-1, 1:nx-1]) + p_upx[1:nz-1, 1:nx-1]*(fluxx[1:nz-1, 1:nx-1] - fluxx[1:nz-1, 2:nx]) ) * gdx_i + (n_upz[0:nz-2, 1:nx-1]*(fluxz[0:nz-2, 1:nx-1] - fluxz[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(fluxz[1:nz-1, 1:nx-1] - fluxz[2:nz, 1:nx-1]) ) * gdz_i )

        return d_adv

def flux_p(h, u, p_upz, n_upz, p_upx, n_upx, hdx_i, hdz_i, S):
        """Flux based advection scheme
        """
        # extract indices
        nz = h.nz
        nx = h.nx

        # extract the velocities
        uz = u[:, :, 0] + S
        ux = u[:, :, 1]

        # define fluxes
        p_adv = np.zeros(np.shape(h.a))
        fluxx = h.a * ux
        fluxz = h.a * uz

        # particulate:
        p_adv[1:nz-1, 1:nx-1] = (  ( n_upx[1:nz-1, 0:nx-2]*( fluxx[1:nz-1, 0:nx-2] - fluxx[1:nz-1, 1:nx-1] ) + p_upx[1:nz-1, 1:nx-1]*( fluxx[1:nz-1, 1:nx-1] - fluxx[1:nz-1, 2:nx]) ) * hdx_i + ( n_upz[0:nz-2, 1:nx-1]*( fluxz[0:nz-2, 1:nx-1] - fluxz[1:nz-1, 1:nx-1] ) +  p_upz[1:nz-1, 1:nx-1]*( fluxz[1:nz-1, 1:nx-1] - fluxz[2:nz, 1:nx-1] ) ) * hdz_i )

        return p_adv


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
        d_adv[1:nz-1, 1:nx-1] = ux[1:nz-1, 1:nx-1] * ( n_upx[1:nz-1, 0:nx-2]*(g.a[1:nz-1, 0:nx-2] - g.a[1:nz-1, 1:nx-1]) + p_upx[1:nz-1, 1:nx-1]*(g.a[1:nz-1, 1:nx-1] - g.a[1:nz-1, 2:nx]) ) * gdx_i + uz[1:nz-1, 1:nx-1] * ( n_upz[0:nz-2, 1:nx-1]*(g.a[0:nz-2, 1:nx-1] - g.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(g.a[1:nz-1, 1:nx-1] - g.a[2:nz, 1:nx-1]) ) * gdz_i 

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

        # particulate advective term:
        p_adv[1:nz-1, 1:nx-1] = S * ( n_upz[1:nz-1, 1:nx-1]*(h.a[0:nz-2, 1:nx-1] - h.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[2:nz, 1:nx-1]) ) * hdz_i + ux[1:nz-1, 1:nx-1] * ( n_upx[1:nz-1, 0:nx-2]*(h.a[1:nz-1, 0:nx-2] - h.a[1:nz-1, 1:nx-1]) + p_upx[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[1:nz-1, 2:nx]) ) * hdx_i + uz[1:nz-1, 1:nx-1] * ( n_upz[0:nz-2, 1:nx-1]*(h.a[0:nz-2, 1:nx-1] - h.a[1:nz-1, 1:nx-1]) + p_upz[1:nz-1, 1:nx-1]*(h.a[1:nz-1, 1:nx-1] - h.a[2:nz, 1:nx-1]) ) * hdz_i 

        return p_adv





def TVD_d(g, u, p_upz, n_upz, p_upx, n_upx, hdx_i, hdz_i, S):
        """total variance diminishing advection scheme
        """


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

        # define upstream flux
        fluxx_up = np.zeros(np.shape(h.a))
        fluxz_up = np.zeros(np.shape(h.a))
        fluxx_up[1:nz-1, 1:nx-1] = ux[1:nz-1, 0:nx-2]*h.a[1:nz-1, 0:nx-2] * n_upx[1:nz-1, 1:nx-1] + ux[1:nz-1, 2:nx]*h.a[1:nz-1, 2:nx] * p_upx[1:nz-1, 1:nx-1]
        fluxz_up[1:nz-1, 1:nx-1] = uz[0:nz-2, 1:nx-1]*h.a[0:nz-2, 1:nx-1] * n_upz[1:nz-1, 1:nx-1] + uz[2:nz, 1:nx-1]*h.a[2:nz, 1:nx-1] * p_upz[1:nz-1, 1:nx-1]

        # new concentration based on upstream-flux method
        tau_up = np.zeros(np.shape(h.a))
        tau_up[1:nz-1, 1:nx-1] = h.a[1:nz-1, 1:nx-1] + ((fluxx_up[1:nz-1, 0:nx-2] - fluxx_up[1:nz-1, 1:nx-1]) * n_upx[1:nz-1, 1:nx-1] + (fluxx_up[1:nz-1, 1:nx-1] - fluxx_up[1:nz-1, 2:nx]) * p_upx[1:nz-1, 1:nx-1]) * hdx_i + ((fluxz_up[0:nz-2, 1:nx-1] - fluxz_up[1:nz-1, 1:nx-1]) * n_upz[1:nz-1, 1:nx-1] + (fluxz_up[1:nz-1, 1:nx-1] - fluxz_up[2:nz, 1:nx-1]) * p_upz[1:nz-1, 1:nx-1]) * hdz_i 
        
        # define centred flux
        fluxx_cen = np.zeros((nz, nx))
        fluxz_cen = np.zeros((nz, nx))
        fluxx_cen[1:nz-1, 1:nx-1] = 0.5 * ( h.a[1:nz-1, 0:nx-2]*ux[1:nz-1, 0:nx-2] - h.a[1:nz-1, 2:nx]*ux[1:nz-1, 2:nx] ) 
        fluxz_cen[1:nz-1, 1:nx-1] = 0.5 * ( h.a[0:nz-2, 1:nx-1]*uz[0:nz-2, 1:nx-1] - h.a[2:nz, 1:nx-1]*uz[2:nz, 1:nx-1] )

        # define anti-diffusive flux
        adfx = fluxx_cen - fluxx_up
        adfz = fluxz_cen - fluxz_up  



        # max and min concentrations in region
        for j in range(1, nx - 1):
                xup = np.zeros((nz, nx)); xdo = np.zeros((nz, nx))
                xup[1:nz-1, j] = max( np.max(h.a[1:nz-1, j-1:j+1]), np.max(tau_up[1:nz-1, j-1:j+1]) )    # C
                xdo[1:nz-1, j] = min( np.min(h.a[1:nz-1, j-1:j+1]), np.min(tau_up[1:nz-1, j-1:j+1]) )    # C
        for i in range(1, nz - 1):
                zup = np.zeros((nz, nx)); zdo = np.zeros((nz, nx))
                zup[i, 1:nx-1] = max( np.max(h.a[i-1:i+1, 1:nx-1]), np.max(tau_up[i-1:i+1, 1:nx-1]) )    # C
                zdo[i, 1:nx-1] = min( np.min(h.a[i-1:i+1, 1:nx-1]), np.min(tau_up[i-1:i+1, 1:nx-1]) )    # C



        # calculate positive and negative adf fluxes in z and x direction
        xpos = np.zeros((nz, nx)); xneg = np.zeros((nz, nx))
        zpos = np.zeros((nz, nx)); zneg = np.zeros((nz, nx))
        for j in range(1, nx - 1):
                for i in range(1, nz - 1):
                        xpos[i, j] = max(adfx[i, j - 1], 0) - min(adfx[i, j], 0)                      # C m/s
                        xneg[i, j] = max(adfx[i, j], 0) - min(adfx[i, j - 1], 0) 
                        zpos[i, j] = max(adfz[i - 1, j], 0) - min(adfz[i, j], 0)                      # C m/s
                        zneg[i, j] = max(adfz[i, j], 0) - min(adfz[i - 1, j], 0)                      # C m/s

        # calculate the 
        zbetup = (zup - tau_up) / zpos #* dx/dt                                                 # C / (C m/s) * m/s = non dimensional
        zbetdo = (tau_up - zdo) / zneg #* dx/dt

        xbetup = (xup - tau_up) / xpos #* dz/dt                                                 # C / (C m/s) * m/s = non dimensional
        xbetdo = (tau_up - xdo) / xneg #* dz/dt

        # remove nans
        zeros = np.zeros(np.shape(zbetup))
        idx = np.isnan(zbetup)
        zbetup[idx] = zeros[idx]
        idx = np.isnan(zbetdo)
        zbetdo[idx] = zeros[idx]
        idx = np.isnan(xbetup)
        xbetup[idx] = zeros[idx]
        idx = np.isnan(xbetdo)
        xbetdo[idx] = zeros[idx]

        # calculate zau, zbu and zcu
        zau = np.zeros((nz, nx)); zbu = np.zeros((nz, nx)); zcu = np.zeros((nz, nx))
        xau = np.zeros((nz, nx)); xbu = np.zeros((nz, nx)); xcu = np.zeros((nz, nx))

        for i in range(1, nz - 1):
                for j in range(1, nx - 1):
                        zau[i, 1:nx - 1] = min((1, zbetdo[i, j], zbetup[i + 1, j]))  # non dim and on the flux points
                        zbu[i, 1:nx - 1] = min(1, zbetup[i, j], zbetdo[i + 1, j])    # non dim and on the flux points
                        zcu[i, 1:nx - 1] = (0.5 + 0.5*np.sign(adfz[i, j]))  

                        xau[i, 1:nx - 1] = min(1, xbetdo[i, j], xbetup[i, j + 1])    # non dim and on the flux points
                        xbu[i, 1:nx - 1] = min(1, xbetup[i, j], xbetdo[i, j + 1])    # non dim and on the flux points
                        xcu[i, 1:nx - 1] = (0.5 + 0.5*np.sign(adfx[i, j]))  

        # calculate final flux
        paaz = adfz * (zcu * zau + (1-zcu)*zbu)                                                   # C m/s on flux points
        paax = adfx * (xcu * xau + (1-xcu)*xbu)                                                   # C m/s on flux points

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



