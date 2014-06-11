###################################################TIME LOOP##########################################################################
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
		self.a = numpy.ones((nz, nx), dtype=numpy.float64)

	def scratchArray(self):
		""" return a scratch array dimensioned for our grid """
		return numpy.zeros((self.nz, self.nx), dtype=numpy.float64)

	def fillBCs(self):               # BC at x = 0 and x = xmax?
		self.a[self.ilo, :] = 0
		self.a[self.ihi - 1, :] = self.a[self.ihi - 2, :]
		self.a[:, self.jlo] = self.a[:, self.jlo + 1]
		self.a[:, self.jhi - 1] = self.a[:, self.jhi - 2]


def adflow(g, h, t, T, V, u, nz, nx, k_ad, k_de, Q):
	"""
	Compute and store the dissolved and particulate [Th] profiles, write them to a file, plot the results.

        :arg t: scale for time at which code is initialized
        :type t: int

	:arg T: scale for time at which code is terminated
	:typeT: int

	:arg V: scale for ux, uz, which are originally order 1.
	:type V: int

	:arg u: 3D tensor of shape (nz, nx, 2), z component of velocity in (:, :, 1), x component of velocity in (:, :, 2) 
	:type u: float

	:arg nz: number of grid points in z dimension
	:type nz: int

	:arg nx: number of grid points in x dimension
	:type nx: int

	:arg k_ad: nz x nx adsorption rate matrix
	:type k_ad: float

	:arg k_de: nz x nx adsorption rate matrix
	:type k_de: float

	"""

	# extract and scale the velocities
	uz = g.zmax / g.xmax * V * u[:, :, 0]
	ux = V * u[:, :, 1]

	# define the CFL, sink velocity, and reaction constant
	S = 500        #m/yr

	# time info
	dt = 0.001          #yr
	tmax = T*(g.zmax - g.zmin)/S            

	# evolution loop
	anew = g.scratchArray()
	bnew = h.scratchArray()

	# define upwind for x, z OUTSIDE loop ONLY while du/dt = 0
	p_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) - 1)
	n_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) + 1)
	p_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) - 1)
	n_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) + 1)

	while (t < tmax):

		# fill the boundary conditions
		g.fillBCs()
		h.fillBCs()

		# loop over zones: note since we are periodic and both endpoints
		# are on the computational domain boundary, we don't have to
		# update both g.ilo and g.ihi -- we could set them equal instead.
		# But this is more general

		i = g.ilo + 1

		while (i <= g.ihi - 1):

			j = g.jlo + 1

			while (j <= g.jhi - 1):

				# upwind numerical solution

				# dissolved:
				anew[i, j] = g.a[i, j] + ( Q - k_ad[i, j] * g.a[i, j] + k_de[i, j] * h.a[i, j] +
					    ux[i, j] * ( n_upx[i, j]*g.a[i, j - 1] - g.a[i, j] + p_upx[i, j]*g.a[i, j + 1] ) / g.dx + 
					    uz[i, j] * ( n_upz[i, j]*g.a[i - 1, j] - g.a[i, j] + p_upz[i, j]*g.a[i + 1, j] ) / g.dz ) 							* dt

				# particulate:
				bnew[i, j] = h.a[i, j] + ( S * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j]) / 							h.dz + 
							  k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
					    ux[i, j] * ( n_upx[i, j]*h.a[i, j - 1] - h.a[i, j] + p_upx[i, j]*h.a[i, j + 1] ) / h.dx +
					    uz[i, j] * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j] ) / h.dz ) 							* dt
				j += 1
			i += 1

		# store the (time) updated solution
		g.a[:] = anew[:]
		h.a[:] = bnew[:]
		t += dt

        return g.a, h.a, T


def plotprof(flowfig, g_a, h_a, xmin, xmax, zmin, zmax, nx, nz, string, T):
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
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz


        meshTh = pylab.subplots(1, 2, figsize = (16.5, 5)) 
        pylab.subplot(121) 
        mesh3 = pylab.pcolormesh(xx_plt/1e3, zz_plt, g_a)
        if string == 'Th':
	        pylab.title('Final Dissolved [Th], tmax = ' + str(10*T) + 'yrs')
        if string == 'Pa':
	        pylab.title('Final Dissolved [Pa], tmax = ' + str(10*t) + 'yrs')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.colorbar(mesh3)
        plt.clim(numpy.min(g_a[:]), numpy.max(g_a[:]))
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

        pylab.subplot(122) 
        mesh4 = pylab.pcolormesh(xx_plt/1e3, zz_plt, h_a)
        if string == 'Th':
	        pylab.title('Final Particulate [Th], tmax = ' + str(10*T) + 'yrs')
        if string == 'Pa':
	        pylab.title('Final Particulate [Pa], tmax = ' + str(10*T) + 'yrs')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.colorbar(mesh4)
        plt.clim(numpy.min(g_a[:]), numpy.max(g_a[:]))
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

	return meshTh



#############################################VELOCITY#################################################################################

def u_zero(ainit, binit, xmin, xmax, zmin, zmax, nx, nz):
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
	flowfig = pylab.subplots(1, 3, figsize = (16, 5))	
	pylab.subplot(131)
	pylab.quiver(xx_plt/1e3, zz_plt, ux[:], -uz[:])
	pylab.gca().invert_yaxis()
	plt.title('Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        # plot initial dist. to flowfig	
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, ainit)
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
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, binit)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])


	return u, flowfig

def u_simple(ainit, binit, xmin, xmax, zmin, zmax, nx, nz, string):
	""" u_simple computes a simple rotational, divergenceless flow field on a specified grid

	:arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension	

	"""

	a = xmax
	b = zmax
	x = numpy.linspace(a/2, -a/2, nx)
	z = numpy.linspace(b/2, -b/2, nz)
	[xx, zz] = numpy.meshgrid(-x, z)
	rr = numpy.sqrt(xx**2 + zz**2)
	theta = numpy.arctan(zz/xx)
	ux = numpy.zeros([nz, nx])
	uz = numpy.zeros([nz, nx])
	idx = rr < a*b/(4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)))
	ux[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * -zz[idx]

	uz[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * xx[idx]
	# store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz
	u[:, :, 1] = ux


	# plot the velocity field

	# define the velocity field with fewer points for plotting, &
	# change sign of uz, because python doesn't understand that
	# down is the positive direction in this case
	N = 10
	x = numpy.linspace(a/2, -a/2, N)
	z = numpy.linspace(b/2, -b/2, N)
	[xx, zz] = numpy.meshgrid(x, z)
	ux_plt = numpy.zeros([N, N])
	uz_plt = numpy.zeros([N, N])
	rr = numpy.sqrt(xx**2 + zz**2)
	theta = numpy.arctan(zz/xx)
	idx = rr < a*b/(4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)))
	ux_plt[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                           (b*numpy.sin(theta[idx]))**2))/rr[idx] * -zz[idx]
	uz_plt[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * xx[idx]

	x_plt = numpy.linspace(xmin, xmax, N)
	z_plt = numpy.linspace(zmin, zmax, N)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
	flowfig = pylab.subplots(1, 3, figsize = (16, 5))	
	pylab.subplot(131)
	pylab.quiver(xx_plt/1e3, zz_plt, ux_plt[:], -zmax / xmax * uz_plt[:])
	pylab.gca().invert_yaxis()
	plt.title('Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

        # plot initial dist. to flowfig	
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, ainit)
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
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, binit)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

	return u, flowfig

def u_complex(ainit, binit, xmin, xmax, zmin, zmax, nx, nz):
	""" u_complex complex computes a rotational, downwelling velocity field

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension

	"""

	# define a grid that will produce downwelling
	a = xmax
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
	idx = rr < a*b/(4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)))
	ux[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * -zz[idx]
	uz[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
		                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * xx[idx]

	# store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz
	u[:, :, 1] = ux 

	# plot the solution
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
	flowfig = pylab.subplots(1, 3, figsize = (16, 5))
	pylab.subplot(131)
	pylab.quiver(xx_plt/1e3, zz_plt, ux, -uz)
	pylab.title('Downwelling Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.gca().invert_yaxis()

        # plot initial dist. to flowfig	
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz
        pylab.subplot(132) 
        mesh1 = pylab.pcolormesh(xx_plt/1e3, zz_plt, ainit)
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
        mesh2 = pylab.pcolormesh(xx_plt/1e3, zz_plt, binit)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])


	return u, flowfig

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
	tmax = 10*T 
	TPratio = pylab.subplots(1, 2, figsize = (16, 5))	
	pylab.subplot(121)
	D = pylab.pcolormesh(xx_plt/1e3, zz_plt, clean_Dratio)
	pylab.gca().invert_yaxis()
	plt.title('Dissolved [Th]/[Pa], tmax = ' + str(tmax) + 'yrs')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.colorbar(D)
	cmin = 0
	cmax = numpy.max((numpy.max(clean_Dratio), numpy.max(clean_Pratio)))
	plt.clim(cmin, cmax)

	pylab.subplot(122)
	ratio = PPa/PTh
	idx = numpy.isnan(ratio)
	ratio = ratio[~idx]
	P = pylab.pcolormesh(xx_plt/1e3, zz_plt, clean_Pratio)
	pylab.gca().invert_yaxis()
	plt.title('Particulate [Th]/[Pa], tmax = ' + str(tmax) + 'yrs')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.colorbar(P)
	plt.clim(cmin, cmax)

	return TPratio

