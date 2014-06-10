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

====================================
Coupled 2D Model: Underlying Physics
====================================
- The results show that the steady state is reached after 30 years. The steady state is different from steady state in the static case: the downwelling region exhibits higher [Th] (left), and the upwellling exhibits lower [Th] (right), as opposed to a horizontally uniform distribution. 

- The velocity scheme is superposed on the background sinking velocity of particulate Th, S = 500 m/yr. The overall sinking rate is therefore minimal in upwelling region, and maximal in downwelling region. The downwelling region should reach a steady state quickly, because the faster the sinking rate, the faster the initial distribution falls out. 

- Why is the steady state different from the zero velocity steady state? 

- Increasing the sinking velocity in the 1D model increases the maximum [Th] at z = zmax. This is congruent to the horizontally varying effect observed here. 

- Why does downwelling and/or a faster sinking rate increase the magnitude of the steady state[Th]?

- A non-zero velocity field can move dissolved Th, which is otherwise stationary. 

- When the sinking velocity is faster, the [Th] has to be decreased near the surface because the production and ad/de-sorption rates have not changed, but the particulate phase falls out at a faster rate. This lesser [Th] at the surface has to be balanced by an increased [Th] at the floor. This is why a faster sinking rate and/or a downwelling region increases the steady state [Th] at depth.

==============================
Coupled 2D Model: Steady State
==============================

**[Th]** 	
		- zero velocity: The magnitude and distribution of [Th] change very little between 40 and 100 years. After 50 years, 				the solution is 90% similar to the 100 year solution.
		- single cell velocity: also takes 50 years
		- two cell velocity: also takes 50 years

**[Th] / [Pa]**	- zero velocity: takes longer than 80 to reach steady state. Still have to run more code.
		- single cell velocity:
		- two cell velocity:



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

	Plot the ratio T/P and output to notebook.

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



