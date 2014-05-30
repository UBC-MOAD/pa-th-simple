# finite-difference implementation of upwind sol. for linear advection.
#
# We are solving a_t = Q + S*a_z
#
# The FTCS discretization is: anew = aold + (C/2) (aold_{i+1} - aold_{i-1})
# 
# where S is the sinking velocity
#
# M. Zingale (2013-03-12)
from __future__ import division
import numpy
import pylab
import math
import matplotlib.pyplot as plt
from math import pi

class FDgrid:

    def __init__(self, nx, nz, ng, xmin = 1, xmax = 10000, zmin = 0, 
                 zmax = 5000):

        self.xmin = xmin
        self.xmax = xmax
        self.zmin = zmin
        self.zmax = zmax
        self.ng = ng
        self.nx = nx
        self.nz = nz

        # python is zero-based!
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
        self.a[self.ihi - 1, :] = self.a[self.ihi - 2, :]
        self.a[:, self.jlo] = self.a[:, self.jlo + 1]
        self.a[:, self.jhi - 1] = self.a[:, self.jhi - 2]


def adflow(T, V):
    """Calculate the Th concentration, particular and dissolved.

    :arg T: Scale for tmax such that tmax = T*(g.zmax - g.zmin)/S 
    :type int:

    :arg V: Scale for ux, uz, which are originally order 1.
    :type V: int
    """
    # create the grid
    nz = 10
    nx = 10
    ng = 1
    g = FDgrid(nx, nz, ng)
    h = FDgrid(nx, nz, ng)

    # create the grid
    nz = 10
    nx = 10
    ng = 1
    g = FDgrid(nx, nz, ng)
    h = FDgrid(nx, nz, ng)

    # define the velocity field u = <ux, uz>
    a = g.xmax
    b = g.zmax
    x = numpy.linspace(a/2, -a/2, nx)
    z = numpy.linspace(b/2, -b/2, nz)
    [xx, zz] = numpy.meshgrid(-x, z)
    rr = numpy.sqrt(xx**2 + zz**2)
    ux = numpy.zeros([nz, nx])
    uz = numpy.zeros([nz, nx])
    theta = numpy.arctan(zz/xx)
    idx = rr < a*b/(4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)))
    ux[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * -zz[idx]

    uz[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * xx[idx]

    ux = V * ux
    uz = V * uz

    # define the CFL, sink velocity, and reaction constant
    S = 500        #m/yr
    Q = 0.0267     #dpm/m^3

    # time info
    dt = 0.001          #yr
    t = 0.0
    tmax = T*(g.zmax - g.zmin)/S            # time interval to reach bottom from surface

    # adsorption/desorption constants
    k_ad = numpy.ones(numpy.shape(g.zz))
    k_ad[251 <= g.z, :] = 0.75
    k_ad[500 <= g.z, :] = 0.5

    k_de = numpy.zeros((numpy.shape(g.zz)))
    k_de[:] = 1.6

    # set initial conditions
    g.a[:, :] = 0.0
    h.a[:, :] = 0.0

    # save initial conditions 
    ainit = g.a.copy()
    binit = h.a.copy()

    # evolution loop
    anew = g.scratchArray()
    bnew = h.scratchArray()

    # time derivative storage
    dgdt = g.scratchArray()
    dhdt = h.scratchArray()

    # depth derivative storage
    dgdz = g.scratchArray()
    dhdz = h.scratchArray()

    # stability parameter storage
    p_upx = g.scratchArray
    n_upx = g.scratchArray
    p_upz = g.scratchArray
    n_upz = g.scratchArray

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
                            uz[i, j] * ( n_upz[i, j]*g.a[i - 1, j] - g.a[i, j] + p_upz[i, j]*g.a[i + 1, j] ) / g.dz ) * dt

                # particulate:
                bnew[i, j] = h.a[i, j] + ( S * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j]) / h.dz + 
                                          k_ad[i, j] * g.a[i, j] - k_de[i, j] * h.a[i, j] + 
                            ux[i, j] * ( n_upx[i, j]*h.a[i, j - 1] - h.a[i, j] + p_upx[i, j]*h.a[i, j + 1] ) / h.dx +
                            uz[i, j] * ( n_upz[i, j]*h.a[i - 1, j] - h.a[i, j] + p_upz[i, j]*h.a[i + 1, j] ) / h.dz ) * dt
                j += 1
            i += 1

        # store the (time) updated solution
        g.a[:] = anew[:]
        h.a[:] = bnew[:]
        t += dt
        
    
    # plot the velocity field

    # define the velocity field with fewer points for plotting
    # change sign of uz, because python doesn't understand that
    # 'down' is the positive direction
    N = 10
    a = g.xmax
    b = g.zmax
    x = numpy.linspace(a/2, -a/2, N)
    z = numpy.linspace(b/2, -b/2, N)
    [xx, zz] = numpy.meshgrid(x, z)
    rr = numpy.sqrt(xx**2 + zz**2)
    ux_plt = numpy.zeros([N, N])
    uz_plt = numpy.zeros([N, N])
    theta = numpy.arctan(zz/xx)
    idx = rr < a*b/(4*numpy.sqrt(1/4 * ((b*numpy.cos(theta))**2 + (a*numpy.sin(theta))**2)))
    ux_plt[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * -zz[idx]
    uz_plt[idx] = numpy.sin(2*pi*rr[idx] / numpy.sqrt((a*numpy.cos(theta[idx])) ** 2 + 
                                            (b*numpy.sin(theta[idx]))**2))/rr[idx] * xx[idx]
    x = numpy.linspace(g.xmin, g.xmax, N)
    z = numpy.linspace(g.zmin, g.zmax, N)
    [xx, zz] = numpy.meshgrid(x, z)
    
    meshinit = pylab.subplots(1, 3, figsize = (17, 5)) 
    pylab.subplot(131)
    uplot = pylab.quiver(xx, zz, ux_plt[:], -uz_plt[:])
    pylab.gca().invert_yaxis()
    plt.title('Velocity field')
    plt.xlabel('x [m]')
    plt.ylabel('depth [m]')


    # plot the Th profiles

    # define the x and zlim maxima
    xmax_plt = (nx - 2)*g.dx
    zmax_plt = (nz - 2)*g.dz
 
    pylab.subplot(132) 
    mesh1 = pylab.pcolormesh(g.xx, g.zz, ainit)
    pylab.title('Initial Dissolved [Th]')
    pylab.gca().invert_yaxis()
    pylab.ylabel('depth [m]')
    pylab.xlabel('x [m]')
    pylab.colorbar(mesh1)
    plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
    pylab.xlim([g.xmin, xmax_plt])
    pylab.ylim([zmax_plt, g.zmin])

    pylab.subplot(133) 
    mesh2 = pylab.pcolormesh(g.xx, g.zz, binit)
    pylab.title('Initial Particulate [Th]')
    pylab.gca().invert_yaxis()
    pylab.ylabel('depth [m]')
    pylab.xlabel('x [m]')
    pylab.colorbar(mesh2)
    plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
    pylab.xlim([g.xmin, xmax_plt])
    pylab.ylim([zmax_plt, g.zmin])

    meshTh = pylab.subplots(1, 2, figsize = (17., 5)) 
    pylab.subplot(121) 
    mesh3 = pylab.pcolormesh(g.xx, g.zz, g.a)
    pylab.title('Final Dissolved [Th], tmax = ' + str(tmax) + 'yrs')
    pylab.gca().invert_yaxis()
    pylab.ylabel('depth [m]')
    pylab.xlabel('x [m]')
    pylab.colorbar(mesh3)
    plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
    pylab.xlim([g.xmin, xmax_plt])
    pylab.ylim([zmax_plt, g.zmin])

    pylab.subplot(122) 
    mesh4 = pylab.pcolormesh(g.xx, g.zz, h.a)
    pylab.title('Final Particulate [Th], tmax = ' + str(tmax) + 'yrs')
    pylab.gca().invert_yaxis()
    pylab.ylabel('depth [m]')
    pylab.xlabel('x [m]')
    pylab.colorbar(mesh4)
    plt.clim(numpy.min(g.a[:]), numpy.max(g.a[:]))
    pylab.xlim([g.xmin, xmax_plt])
    pylab.ylim([zmax_plt, g.zmin])
    
    #save [Th] profiles
    log_ga = open('ga_'+str(T)+'*tmax_'+str(V)+'*U.log', 'w')
    log_ha = open('ha_'+str(T)+'*tmax_'+str(V)+'*U.log', 'w')
    print>>log_ga, g.a
    print>>log_ha, h.a
    
    return meshinit, meshTh
