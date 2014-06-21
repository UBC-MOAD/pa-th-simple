""" 
Finite-difference implementation of upwind sol. for coupled linear advection.

We are solving  a_t = Q - k_ad*a + k_de*b + <u> * <del>a

                b_t = k_ad*a - k_de*b + S*b_z + <u> * <del>b
                
The FTCS discretization is: a_new = a_old + (C/2) * (aold_{i+1} - aold_{i-1})
The upwind discretization is: a_new = a_old + C *(aold_{i+1} - aold_{i-1})
 
"""
from __future__ import division
import numpy
import pylab
import math
import ThPa2D
import matplotlib.pyplot as plt
from math import pi

class FDgrid:

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
		self.x = xmin + (numpy.arange(nx) - ng) * self.dx
		self.dz = (zmax - zmin) / (nz - 1)
		self.z = zmin + (numpy.arange(nz) - ng) * self.dz
		[self.xx, self.zz] = numpy.meshgrid(self.x, self.z)

		# storage for the solution 
		self.a = numpy.zeros((nz, nx), dtype=numpy.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return numpy.zeros((self.nz, self.nx), dtype=numpy.float64)

	def fillBCs(self):               # BC at x = 0 and x = xmax?
		self.a[self.ilo, :] = 0
		self.a[self.ihi, :] = self.a[self.ihi - 1, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
                #self.a[:, self.jhi - 1] = self.a[:, self.jhi - 2]
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
	S = 500        #m/yr

	# time info
	dt = 0.001          #yr
        t = t * (g.zmax - g.zmin)/S
	T = T * (g.zmax - g.zmin)/S            

        g, h = adscheme(g, h, t, T, u, k_ad, k_de, Q, S, dt)

        return g, h

def upwind(g, h, t, T, u, k_ad, k_de, Q, S, dt):

	# extract the velocities
	uz = u[:, :, 0]
	ux = u[:, :, 1]

	# evolution loop
	anew = g.a
	bnew = h.a

	# define upwind for x, z OUTSIDE loop ONLY while du/dt = 0
	p_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) - 1)
	n_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) + 1)
	p_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) - 1)
	n_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) + 1)

	while (t < T):

		# fill the boundary conditions
		g.fillBCs()
		h.fillBCs()

		# loop over zones: note since we are periodic and both endpoints
		# are on the computational domain boundary, we don't have to
		# update both g.ilo and g.ihi -- we could set them equal instead.
		# But this is more general


                i = numpy.arange(g.ilo + 1, g.ihi, 1, dtype = int)
                j = numpy.arange(g.jlo + 1, g.jhi, 1, dtype = int)
                [i , j] = numpy.meshgrid(i,j)
                # upwind numerical solution

                # dissolved:
                anew[i, j] = g.a[i, j] + ( Q - k_ad[i, j] * g.a[i, j] + k_de[i, j] * h.a[i, j] +
                    ux[i, j] * ( n_upx[i, j]*g.a[i, j - 1] - g.a[i, j] + p_upx[i, j]*g.a[i, j + 1] ) / g.dx + 
                    uz[i, j] * ( n_upz[i, j]*g.a[i - 1, j] - g.a[i, j] + p_upz[i, j]*g.a[i + 1, j] ) / g.dz ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j]) / h.dz + 
                          k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                    ux[i, j] * ( n_upx[i, j]*h.a[i, j - 1] - h.a[i, j] + p_upx[i, j]*h.a[i, j + 1] ) / h.dx +
                    uz[i, j] * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j] ) / h.dz ) * dt

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt
        return g, h






def u_zero(g, h, xmin, xmax, zmin, zmax, nx, nz, V, string):
	""" Produce a matrix of zeros on the input grid to simulate a zero velocity feild
        :arg g_a: the dissolved [] final distribution

        :arg h_a: the particulate [] final distribution

        :arg ainit: the dissolved [] initial distribution 

        :arg binit: the particulate [] initial distribution
 
	:arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""
	# define grid
	a = xmax
	b = zmax
	x = numpy.linspace(a/2, -a/2, nx)
	z = numpy.linspace(b/2, -b/2, nz)
	[xx, zz] = numpy.meshgrid(-x, z)
	rr = numpy.sqrt(xx**2 + zz**2)
	theta = numpy.arctan(zz/xx)
	ux = numpy.zeros([nz, nx])
	uz = numpy.zeros([nz, nx])

	# store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz
	u[:, :, 1] = ux
	
	# plot result
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
	flowfig = pylab.figure(figsize = (49, 5))	
	pylab.subplot(131)
	pylab.quiver(xx_plt/1e3, zz_plt, ux[:], uz[:])
	pylab.gca().invert_yaxis()
	plt.title('Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        # plot initial dist.
        init = pylab.subplots(1, 2, figsize = (23, 5))
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, g.a)
        if string == 'Th': 
	        pylab.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        pylab.title('Initial Dissolved [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

        pylab.subplot(133) 
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, h.a)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])


	return u, flowfig, init

def u_simple(g, h, xmin, xmax, zmin, zmax, nx, nz, V, string):
	""" u_simple computes a simple rotational, divergenceless flow field on a specified grid

	:arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension	

	"""
        # define velocity on square grid, then scale simulate rectangular grid. 
	a = zmax
        b = zmax
        x = numpy.linspace(-a/2, a/2, nx)
        z = numpy.linspace(-b/2, b/2, nz)
        [xx, zz] = numpy.meshgrid(x, z)
        rr = numpy.sqrt(xx**2 + zz**2)
        theta = numpy.arctan(zz/xx)
        ux = numpy.zeros([nz, nx])
        uz = numpy.zeros([nz, nx])
        idx = rr < a*b/( 4 * numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)) )

        ux[idx] = numpy.sin(2*pi*rr[idx] / (a*b / numpy.sqrt((a*numpy.sin(theta[idx])) ** 2 + 
                                        (b*numpy.cos(theta[idx])) ** 2)))/rr[idx] * zz[idx]

        uz[idx] = numpy.sin(2*pi*rr[idx] / (a*b / numpy.sqrt((a*numpy.sin(theta[idx])) ** 2 + 
                                        (b*numpy.cos(theta[idx])) ** 2)))/rr[idx] * xx[idx]

        # scale & store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz / numpy.max(uz) * V * zmax/xmax
	u[:, :, 1] = ux / numpy.max(ux) * V 

	# plot the velocity field you are actually using (so you can be sure you got it right) on rectangular grid.         
        a = xmax
        x = numpy.linspace(-a/2, a/2, nx)
	flowfig = pylab.figure(figsize = (48, 5))	
	pylab.subplot(131)
	pylab.quiver(1e-3*(x[::2]+a/2), z[::2]+b/2, ux[::2,::2], uz[::2,::2])
	pylab.gca().invert_yaxis()
	plt.title('Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        # plot initial dist.
        init = pylab.subplots(1, 2, figsize = (23, 5))	
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, g.a)
        if string == 'Th': 
	        pylab.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        pylab.title('Initial Dissolved [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax/1e3])
        pylab.ylim([zmax, zmin])

        pylab.subplot(133) 
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, h.a)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax/1e3])
        pylab.ylim([zmax, zmin])

	return u, flowfig, init

def u_complex(g, h, xmin, xmax, zmin, zmax, nx, nz, V, string):
	""" u_complex complex computes a rotational, downwelling velocity field

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
	x = numpy.empty(nx)
	z = numpy.empty(nz)
	x[:round(nx/4)] = numpy.linspace(-a/2, 0, len(x[:round(nx/4)]))
	x[round(nx/4) : round(nx/2)] = numpy.linspace(0, a/2, len(x[round(nx/4) : round(nx/2)]))
	x[round(nx/2) : round(3*nx/4)] = numpy.linspace(a/2, 0, len(x[round(nx/2) : round(3*nx/4)]))
	x[round(3*nx/4) : nx] = numpy.linspace(0, -a/2, len(x[round(3*nx/4) : nx]))
	z[:round(nz/2)] = numpy.linspace(-b/2, 0, len(z[:round(nz/2)]))
	z[round(nz/2) : nz] = numpy.linspace(0, b/2, len(z[round(nz/2) : nz]))
	[xx, zz] = numpy.meshgrid(x, z)
	zz[0:, nx/2:] = - zz[0:, nx/2:]  
	rr = numpy.sqrt(xx**2 + zz**2)
	ux = numpy.zeros([nz, nx])
	uz = numpy.zeros([nz, nx])

	# use logical indexing to define points of non-zero velocity
	theta = numpy.arctan(zz/xx)
	idx = rr < a*b/ ( 4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)) )

        ux[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx])) ** 2))/rr[idx] * -zz[idx]

        uz[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx])) ** 2))/rr[idx] * -xx[idx]

	# scale & store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz / numpy.max(uz) * V * zmax/xmax
	u[:, :, 1] = ux / numpy.max(ux) * V

	# plot the velocity field you are actually using (so you can be sure you got it right) 
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
	flowfig = pylab.figure(figsize = (49, 5))
	pylab.subplot(131)
	pylab.quiver(1e-3*xx_plt, zz_plt, ux, uz)
	pylab.gca().invert_yaxis()
	pylab.title('Downwelling Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        # plot initial dist.
        init = pylab.subplots(1, 2, figsize = (23, 5))	
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, g.a)
        if string == 'Th': 
	        pylab.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        pylab.title('Initial Dissolved [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax/1e3])
        pylab.ylim([zmax, zmin])

        pylab.subplot(133) 
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, h.a)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax/1e3])
        pylab.ylim([zmax, zmin])


	return u, flowfig, init

###################################################CHEMISTRY##########################################################################

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
	x = xmin + (numpy.arange(nx) - 1) * dx
	dz = (zmax - zmin) / (nz - 1)
	z = zmin + (numpy.arange(nz) - 1) * dz
	[xx, zz] = numpy.meshgrid(x, z)

	if string == 'Pa':

		k_ad = numpy.ones(numpy.shape(zz))
		k_ad[:, :] = 0.08
		k_ad[251 <= z, :] = 0.06
		k_ad[500 <= z, :] = 0.04

		k_de = numpy.zeros((numpy.shape(zz)))
		k_de[:] = 1.6
		
		Q = 0.00246

	if string == 'Th':
		k_ad = numpy.ones(numpy.shape(zz))
		k_ad[251 <= z, :] = 0.75
		k_ad[500 <= z, :] = 0.5

		k_de = numpy.ones(numpy.shape(zz))

		Q = 0.0267
	
	return k_ad, k_de, Q	
#######################################################PLOTTING#######################################################################
def plotratio(DTh, DPa, PTh, PPa, xmin, xmax, zmin, zmax, nx, nz, T):
	""" Plots the ratio T/P and outputs to notebook

	:arg DTh: 2D profile of dissolved Th

	:arg PTh: 2D profile of particulate Th

	:arg DPa: 2D profile of dissolved Pa	

	:arg PPa: 2D profile of particulate Pa

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""

	# define grid
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)

	# remove NaNs
	Dratio = DTh/DPa
	idx = numpy.isnan(Dratio)
	clean_Dratio = numpy.zeros([nz, nx])
	clean_Dratio[~idx] = Dratio[~idx]

	Pratio = PTh/PPa
	idx = numpy.isnan(Pratio)
	clean_Pratio = numpy.zeros([nz, nx])
	clean_Pratio[~idx] = Pratio[~idx]

	# plot 
	TPratio = pylab.subplots(1, 2, figsize = (16, 5))
	
	pylab.subplot(121)
	D = pylab.pcolormesh(xx_plt*1e-3, zz_plt, clean_Dratio)
	pylab.gca().invert_yaxis()
	plt.title('Dissolved [Th]/[Pa], tmax = ' + str(10*T) + 'yrs')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.colorbar(D)
	cmin = 0
	cmax = numpy.max((numpy.max(clean_Dratio), numpy.max(clean_Pratio)))
	plt.clim(cmin, cmax)

	pylab.subplot(122)
	P = pylab.pcolormesh(xx_plt*1e-3, zz_plt, clean_Pratio)
	pylab.gca().invert_yaxis()
	plt.title('Particulate [Th]/[Pa], tmax = ' + str(10*T) + 'yrs')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.colorbar(P)
	plt.clim(cmin, cmax)

	return TPratio
	
def plotprof(g, h, xmin, xmax, zmin, zmax, nx, nz, T, string):
        """Plot the dissolved and particulate profile of either [Th] or [Pa]

        :arg slowfig: figure with 3 subplots. 1 - vel. field; 2 - dissolved initial distribution; 3 - particulate initial distribution

        :arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	:arg string: string, either 'Th' or 'Pa' which determines which title to use on figures
        """

        # define grid
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 1)*dx
        zmax_plt = (nz - 1)*dz
        tmax = 10*T


        meshTh = pylab.subplots(1, 2, figsize = (16.5, 5)) 
        pylab.subplot(121) 
        mesh3 = pylab.pcolormesh(xx_plt/1e3, zz_plt, g.a)
        if string == 'Th':
	        pylab.title('Final Dissolved [Th], tmax = ' + str(tmax) + 'yrs')
        if string == 'Pa':
	        pylab.title('Final Dissolved [Pa], tmax = ' + str(tmax) + 'yrs')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.colorbar(mesh3)
        plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

        pylab.subplot(122) 
        mesh4 = pylab.pcolormesh(xx_plt/1e3, zz_plt, h.a)
        if string == 'Th':
	        pylab.title('Final Particulate [Th], tmax = ' + str(tmax) + 'yrs')
        if string == 'Pa':
	        pylab.title('Final Particulate [Pa], tmax = ' + str(tmax) + 'yrs')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.colorbar(mesh4)
        plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

	return meshTh
