*****************
Preliminary Model
*****************

The preliminary model is composed to two major working parts: the Th2D module, which contains all the functions used to produce the results, and the Coupled 2D Model notebook, which displays the results. 


================
Coupled 1D Model
================

- Produce a 1D profile of [Th] in which dissolved and particulate phases are coupled by constant adsorption and desorption rates. 


- Code takes ~5 tmax to reach a linear steady state, where tmax = (g.zmax - g.zmin)/S, the time for a particle to sink from surface to floor (10 years with current parameters).


- The grid is 100 x 1.


- Before reaching a steady state, Th is uniformly distributed in depth, which is partly due to the production term Q, and partly due to the initial distribution. 


- It takes a time interval t = [0:tmax] for a particle to sink from an initial depth z = [0:zmax], which means after a time interval tmax, the initial distribution has fallen out. 


- Unintuitively, it takes longer than this to reach a steady state. The adsorption and desorption rates are significant enough that they extend the time elapsed before steady state. Since only the particulate phase of Th is subject to sinking, there could be a series of "stalls" in the sinking trajectory of a Th atom when is desorbs into the dissolved phase. This theory could be tested by changing the ad/de-sorption rates and comparing time to reach steady state.


- The steady state propogates from the surface to the floor. This is predominantly due to the initial distribution falling out at t = h/S, where h is the height of water above a given point, and S is sinking velocity, and partially due to a higher rate of adsorption near the surface.


=========================
Coupled 2D Model: Th2D.py
=========================

.. function:: Th2D.adflow(T, V, u, nz, nx, k_ad, k_de, Q, flowfig)
	
	Compute and store the dissolved and particulate [Th] profiles, 
	write them to a file, plot the results.

	:arg T: scale for tmax such that tmax = T*(g.zmax - g.zmin)/S 
	:type T: int

	:arg V: scale for ux, uz, which are originally order 1.
	:type V: int

	:arg u: 3D tensor of shape [nz, nx, 2]. Stores z component of velocity in [:, :, 1], x component of velocity in [:, :, 2] 
	:type u: float

	:arg nz: number of grid points in z dimension
	:type nz: int

	:arg nx: number of grid points in x dimension
	:type nx: int

	:arg k_ad: nz x nx adsorption rate matrix
	:type k_ad: float

	:arg k_de: nz x nx adsorption rate matrix
	:type k_de: float

	:arg adscheme: function to implement the desired advection scheme 
	:type adscheme: function

.. function:: Th2D.u_simple(xmin, xmax, zmin, zmax, nx, nz)

	Compute a simple rotational, divergenceless flow field 
	on a specified grid.

	:arg xmin: minimum x on the grid
	
	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension	


.. function:: Th2D.u_complex(xmin, xmax, zmin, zmax, nx, nz)

	Compute a rotational, downwelling velocity field.

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension



.. function:: Th2D.k_sorp(string, xmin, xmax, zmin, zmax, nx, nz)

	Compute adsorption,desorption, & production constants for 
	Th or Pa.

	:arg string: a string, either 'Th' or 'Pa'

	:arg xmin: minimum x on the grid

	:arg xmax: maximum x on the grid

	:arg zmin: minimum z on the grid

	:arg zmax: maximum z on the grid

	:arg nx: number of points in x dimension

	:arg nz: number of points in z dimension


.. function:: Th2D.plotratio(DTh, DPa, PTh, PPa, xmin, xmax, zmin, zmax, nx, nz, T)

	Plot the ratio T/P and outputs to notebook.

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

	:arg T: scale for tmax such that tmax = T*(g.zmax - g.zmin)/S
	:type T: int



