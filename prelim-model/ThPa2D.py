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

        # save inverses for speed
        g.dx_i = 1/g.dx
        g.dz_i = 1/g.dz
        h.dx_i = 1/h.dx
        h.dz_i = 1/h.dz

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
                    ux[i, j] * ( n_upx[i, j]*g.a[i, j - 1] - g.a[i, j] + p_upx[i, j]*g.a[i, j + 1] ) * g.dx_i + 
                    uz[i, j] * ( n_upz[i, j]*g.a[i - 1, j] - g.a[i, j] + p_upz[i, j]*g.a[i + 1, j] ) * g.dz_i ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j]) * h.dz_i + 
                          k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                    ux[i, j] * ( n_upx[i, j]*h.a[i, j - 1] - h.a[i, j] + p_upx[i, j]*h.a[i, j + 1] ) * h.dx_i +
                    uz[i, j] * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j] ) * h.dz_i ) * dt

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

	# define upwind for x, z OUTSIDE loop ONLY while du/dt = 0
	p_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) - 1)
	n_upx = numpy.sign(ux)*0.5*( numpy.sign(ux) + 1)
	p_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) - 1)
	n_upz = numpy.sign(uz + S)*0.5*( numpy.sign(uz + S) + 1)

        # save inverses for speed
        g.dx_i = 1/g.dx
        g.dz_i = 1/g.dz
        h.dx_i = 1/h.dx
        h.dz_i = 1/h.dz

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
                                          
                      ( n_upx[i, j]*(g.a[i, j - 1]*ux[i, j - 1] - g.a[i, j]*ux[i, j]) + 
                        p_upx[i, j]*(g.a[i, j]*ux[i, j] - g.a[i, j + 1]*ux[i, j + 1]) ) * g.dx_i + 
                      ( n_upz[i, j]*(g.a[i - 1, j]*uz[i - 1, j] - g.a[i, j]*uz[i, j]) + 
                        p_upz[i, j]*(g.a[i, j]*ux[i, j] - g.a[i + 1, j]*uz[i + 1, j]) ) * g.dz_i ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + 
                ( S *( n_upz[i, j]*(h.a[i - 1, j] - h.a[i, j]) + p_upz[i, j]*(h.a[i, j] - h.a[i + 1, j]) )* h.dz_i + 
                          k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                          
                    ( n_upx[i, j]*(h.a[i, j - 1]*ux[i, j - 1] - h.a[i, j]*ux[i, j]) + 
                      p_upx[i, j]*(h.a[i, j]*ux[i, j] - h.a[i, j + 1]*ux[i, j + 1]) ) * h.dx_i +
                    ( n_upz[i, j]*(h.a[i - 1, j]*uz[i - 1, j] - h.a[i, j]*uz[i, j]) + 
                      p_upz[i, j]*(h.a[i, j]*uz[i, j] - h.a[i + 1, j]*uz[i + 1, j]) ) * h.dz_i ) * dt

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt
        return g, h
    
    
    
def TVD(g, h, t, T, u, k_ad, k_de, Q, S, dt):

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
                                          
                    ux[i, j] * ( n_upx[i, j]*(g.a[i, j - 1] - g.a[i, j + 1]) + p_upx[i, j]*(g.a[i, j + 1] - g.a[i, j - 1]) ) / g.dx + 
                    uz[i, j] * ( n_upz[i, j]*(g.a[i - 1, j] - g.a[i + 1, j]) + p_upz[i, j]*(g.a[i + 1, j] - g.a[i - 1, j]) ) / g.dz ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j]) / h.dz + 
                          k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                    ux[i, j] * ( n_upx[i, j]*(h.a[i, j - 1] - h.a[i, j + 1]) + p_upx[i, j]*(h.a[i, j + 1] - h.a[i, j - 1] ) ) / h.dx +
                    uz[i, j] * ( n_upz[i, j]*(h.a[i - 1, j] - h.a[i + 1, j]) + p_upz[i, j]*(h.a[i + 1, j] - h.a[i - 1, j] ) ) / h.dz ) * dt


                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt
        return g, h


def u_zero(xmin, xmax, zmin, zmax, nx, nz, V):
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
	flowfig = pylab.figure(figsize = (25, 5))	
	pylab.quiver(1e-3 * xx_plt, zz_plt, ux[:], uz[:])
	pylab.gca().invert_yaxis()
	plt.title('Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')

	return u, flowfig

def u_simple(xmin, xmax, zmin, zmax, nx, nz, V):
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

	# calculate size of half a grid cell in the vertical
	hdz = 0.5*b/nz
        [xx, zz] = numpy.meshgrid(x, z)
        rr = numpy.sqrt(xx**2 + zz**2)

	# calculate the radius half a grid cell up and half a grid cell down
	rrup = numpy.sqrt(xx**2 + (zz+hdz)**2)
	rrdw = numpy.sqrt(xx**2 + (zz-hdz)**2)
	
        ux = numpy.zeros([nz, nx])
        uz = numpy.zeros([nz, nx])

	# left side of domain, where flow is upward, go 1/2 step down
        idx = numpy.logical_and(rr < a/2, xx <= 0)

        ux[idx] = -numpy.sin(2*pi*rrdw[idx] / a) / rrdw[idx] * (zz[idx]-hdz)
        uz[idx] = numpy.sin(2*pi*rrdw[idx] / a) / rrdw[idx] * xx[idx]

	# right side of domain where flow is downward, go 1/2 step up
	jdx = numpy.logical_and(rr < a/2, xx > 0)

        ux[jdx] = -numpy.sin(2*pi*rrup[jdx] / a) / rrup[jdx] * (zz[jdx]+hdz)
        uz[jdx] = numpy.sin(2*pi*rrup[jdx] / a) / rrup[jdx] * xx[jdx]

        # remove nans
        nanfill = numpy.zeros((nz, nx))
        id_nan = numpy.isnan(ux)
        ux[id_nan] = nanfill[id_nan]
        id_nan = numpy.isnan(uz)
        uz[id_nan] = nanfill[id_nan]

        # scale & store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz / numpy.max(uz) * V * zmax/xmax
	u[:, :, 1] = ux / numpy.max(ux) * V 

	# plot the velocity field you are actually using (so you can be sure you got it right) on rectangular grid.         
        a = xmax
        x = numpy.linspace(-a/2, a/2, nx)
	flowfig = pylab.figure(figsize = (25, 5))	
	pylab.quiver(1e-3*(x+a/2), z+b/2, ux, -uz, pivot = 'mid')
	pylab.gca().invert_yaxis()
	plt.title('Velocity Field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')


	return u, flowfig

def u_simple_c(u, xmin, xmax, zmin, zmax, nx, nz):
        """Correct the analytical solution to conserve mass discretely
        """
        # extract velocity components
        uz = u[:, :, 0]
        ux = u[:, :, 1]
        
        # define upstream 
        p_upz = numpy.sign(uz)*0.5*( numpy.sign(uz) - 1)
        n_upz = numpy.sign(uz)*0.5*( numpy.sign(uz) + 1)

        # set up vectorized correction 
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        ux = ux 

        # vectorize region where z > 0, ux > 0
        i = numpy.arange(1, nz/2, 1, dtype = int)
        j = 1
        while j <= nx-2:

            ux[i, j] = ux[i, j - 1] + dx/dz* (( uz[i,j] - uz[i + 1, j])*p_upz[i, j] + (uz[i - 1, j] - uz[i,j])*n_upz[i, j])
            j += 1

        # vectorize region z < 0, ux < 0
        i = numpy.arange(nz/2, nz - 1, 1, dtype = int)
        j = nx - 2
        while j >= 1:
            
            ux[i, j] = ux[i, j + 1] - dx/dz* ((uz[i,j] - uz[i + 1, j]) *p_upz[i, j] + (uz[i - 1, j] - uz[i,j])*n_upz[i, j])
            j -= 1

        # store result
        u[:, :, 1] = ux
        # plot result        
        a = xmax
        b = zmax
        x = numpy.linspace(-a/2, a/2, nx)
        z = numpy.linspace(-b/2, b/2, nz)
        flowfig = pylab.figure(figsize = (25, 5))
        # scale uz by 10 for visual effect
        pylab.quiver(1e-3*(x+a/2), z+b/2, ux, -dx/dz*uz)
        pylab.gca().invert_yaxis()
        plt.title('Corrected Velocity Field')
        plt.xlabel('x [km]')
        plt.ylabel('depth [m]') 

        return u, flowfig

def u_complex(xmin, xmax, zmin, zmax, nx, nz, V):
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
	x = numpy.zeros(nx)

        #nx should always be odd
        x[0:(nx+1)/2] = numpy.linspace(-a/2, a/2, (nx+1)/2)
        x[(nx-1)/2:] = numpy.linspace(a/2, -a/2, (nx+1)/2)
        z = numpy.linspace(-b/2, b/2, nz)
	[xx, zz] = numpy.meshgrid(x, z)
	zz[0:, nx/2:] = - zz[0:, nx/2:]  
	rr = numpy.sqrt(xx**2 + zz**2)
	ux = numpy.zeros((nz, nx))
	uz = numpy.zeros((nz, nx))

	# use logical indexing to define points of non-zero velocity
	idx = rr < a/2

        ux[idx] = numpy.sin(2*pi*rr[idx] / a) / rr[idx] * -zz[idx]

        uz[idx] = -numpy.sin(2*pi*rr[idx] / a) / rr[idx] * -xx[idx]

        # remove nans
        nanfill = numpy.zeros((nz, nx))
        id_nan = numpy.isnan(ux)
        ux[id_nan] = nanfill[id_nan]
        id_nan = numpy.isnan(uz)
        uz[id_nan] = nanfill[id_nan]

	# scale & store the solution in a matrix
	u = numpy.zeros([nz, nx, 2])
	u[:, :, 0] = uz / numpy.max(uz) * V * zmax/xmax
	u[:, :, 1] = ux / numpy.max(ux) * V

	# plot the velocity field you are actually using (so you can be sure you got it right) 
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
	flowfig = pylab.figure(figsize = (25, 5))
	pylab.quiver(1e-3*xx_plt, zz_plt, ux, -uz)
	pylab.gca().invert_yaxis()
	pylab.title('Downwelling Velocity field')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')


	return u, flowfig

def u_complex_c(u, xmin, xmax, zmin, zmax, nx, nz):
        """Correct the complex velocity field to conserve mass on grid-by-grid basis
        """
        ux = u[:,:,1]
        uz = u[:,:,0]
        p_upz = numpy.sign(uz)*0.5*( numpy.sign(uz) - 1)
        n_upz = numpy.sign(uz)*0.5*( numpy.sign(uz) + 1)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        # vectorize region z > 0
        i = numpy.arange(1, nz/2, 1, dtype = int)
        j = 1

        while j <= nx/2 - 1:
            ux[i, j] = ux[i, j - 1]+ dx/dz * ((uz[i - 1, j] - uz[i, j])*p_upz[i, j] + (uz[i, j] - uz[i + 1, j])*n_upz[i, j])
            j += 1
             
        while j <= nx - 2:
            ux[i, j] = ux[i, j + 1] - dx/dz * ((uz[i - 1, j] - uz[i, j])*n_upz[i, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j])
            j += 1
               
        # vectorize region z < 0
        i = numpy.arange(nz/2, nz - 1, 1, dtype = int)
        j = 1
        while j <= nx/2 - 1:
            ux[i, j] = ux[i, j + 1] - dx/dz * ( (uz[i, j] - uz[i + 1, j])*p_upz[i, j] + (uz[i - 1, j] - uz[i, j])*n_upz[i, j] )
            j += 1
            
        while j <= nx - 2:
            ux[i, j] = ux[i, j - 1] + dx/dz * ( (uz[i - 1, j] - uz[i, j])*n_upz[i, j] + (uz[i, j] - uz[i + 1, j])*p_upz[i, j] )
            j += 1
            
            
        # plot result        
        a = xmax
        b = zmax
        x = numpy.linspace(-a/2, a/2, nx)
        z = numpy.linspace(-b/2, b/2, nz)
        flowfig = pylab.figure(figsize = (25, 5))
        pylab.quiver(1e-3 * (x+a/2), z+b/2, ux, -dx/dz * uz)
        pylab.gca().invert_yaxis()
        plt.title('Corrected Velocity Field')
        plt.xlabel('x [km]')
        plt.ylabel('depth [m]') 
        
        return u, flowfig

def plot_init(g, h, xmin, xmax, zmin, zmax, nx, nz, string):

        # plot initial dist.
	x_plt = numpy.linspace(xmin, xmax, nx)
	z_plt = numpy.linspace(zmin, zmax, nz)
	[xx_plt, zz_plt] = numpy.meshgrid(x_plt, z_plt)
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1)
        xmax_plt = (nx - 2)*dx
        zmax_plt = (nz - 2)*dz

        init = pylab.subplots(1, 2, figsize = (25, 5))

        pylab.subplot(121) 
        mesh1 = pylab.pcolormesh(1e-3 * xx_plt, zz_plt, g.a)
        if string == 'Th': 
	        pylab.title('Initial Dissolved [Th]')
        if string == 'Pa':
	        pylab.title('Initial Dissolved [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([1e-3 * xmin, 1e-3 * xmax_plt])
        pylab.ylim([zmax_plt, zmin])

        pylab.subplot(122) 
        mesh2 = pylab.pcolormesh(1e-3 * xx_plt, zz_plt, h.a)
        if string == 'Th':
	        pylab.title('Initial Particulate [Th]')
        if string == 'Pa':
	        pylab.title('Initial Particulate [Pa]')
        pylab.gca().invert_yaxis()
        pylab.ylabel('depth [m]')
        pylab.xlabel('x [km]')
        pylab.xlim([1e-3 * xmin, 1e-3 * xmax_plt])
        pylab.ylim([zmax_plt, zmin])

        return init

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
	#cmin = 0
	#cmax = numpy.max((numpy.max(clean_Dratio), numpy.max(clean_Pratio)))
	#plt.clim(cmin, cmax)

	pylab.subplot(122)
	P = pylab.pcolormesh(xx_plt*1e-3, zz_plt, clean_Pratio)
	pylab.gca().invert_yaxis()
	plt.title('Particulate [Th]/[Pa], tmax = ' + str(10*T) + 'yrs')
	plt.xlabel('x [km]')
	plt.ylabel('depth [m]')
	pylab.colorbar(P)
	#plt.clim(cmin, cmax)

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


        meshTh = pylab.subplots(1, 2, figsize = (25, 5)) 
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
        #plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
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
        #plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
        pylab.xlim([xmin/1e3, xmax_plt/1e3])
        pylab.ylim([zmax_plt, zmin])

	return meshTh

def divtest(u, xmax, xmin, zmax, zmin, nx, nz):

        ux = u[:,:,1]
        uz = u[:,:,0]

        # set up vectorized correction \n",
        dx = (xmax - xmin) / (nx - 1)
        dz = (zmax - zmin) / (nz - 1) 

        # QUAD 1
        i = numpy.arange(0, nz/2, 1, dtype = int)
        j = 1
        div = numpy.zeros((nz, nx))
        while j <= (nx - 1)/2:

            div[i,j] = dz * (ux[i, j - 1] - ux[i, j]) + dx * (uz[i,j] - uz[i + 1, j])
            j += 1    

        # QUAD 2
        i = numpy.arange(1, nz/2, 1, dtype = int)
        while j < nx - 2:

            div[i, j] = dz * (ux[i, j - 1] - ux[i, j]) + dx * (uz[i - 1, j] - uz[i,j])
            j += 1    

        # QUAD 3
        i = numpy.arange(nz/2, nz, 1, dtype = int)
        while j >= (nx - 1)/2:

            div[i, j] = dz * (ux[i, j] - ux[i, j + 1]) + dx * (uz[i - 1, j] - uz[i,j])
            j -= 1    

        # QUAD 4
        i = numpy.arange(nz/2, nz - 1, 1, dtype = int)
        while j >= 0:

            div[i, j] = dz * (ux[i, j] - ux[i, j + 1]) + dx * (uz[i,j] - uz[i + 1, j])
            j -= 1    
                
        # plot the results
        pylab.figure(figsize = (25, 5))
        divplot = pylab.pcolormesh(div)
        pylab.colorbar(divplot)
        pylab.gca().invert_yaxis()

        return divplot
