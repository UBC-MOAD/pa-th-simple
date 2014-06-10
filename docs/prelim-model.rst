*****************
Preliminary Model
*****************

The preliminary model is composed to two major working parts: the Th2D module, which contains all the functions used to produce the results, and the Coupled 2D Model notebook, which displays the results. 


================
Coupled 1D Model
================

- The steady state is reached after 50 years.

- The grid is 100 x 1.

- The solution propogates from surface to seafloor, and increases linearly with depth. 

- The [Th] distribution is constant *before* the steady state is established because the production, adsorption, and desorption rates are constant.

- It takes a time interval t = [0:tmax] for a particle to sink from an initial depth z = [0:zmax], which means after a time interval tmax = zmax/S, where S is the sinking velocity, the initial distribution has fallen out. 

- Unintuitively, it takes longer than this to reach a steady state. Since only the particulate phase of Th is subject to sinking, there could be a series of "stalls" in the sinking trajectory of an atom when is desorbs into the dissolved phase.

.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/1Dinit[Th].png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/1Dfinal[Th].png
  
====================================
Coupled 2D Model: Underlying Physics
====================================

- The steady state is reached after 50 years, and propogates from surface to seafloor.

- The downwelling region approaches steady state at a faster rate the the upwelling region because particles in this region traverse the full ocean depth more quickly, so the steady state propagates more quickly also.

- The downwelling region exhibits higher s.s. [], and the upwellling exhibits lower s.s. [].

- When the sinking velocity is faster, the [] has to be decreased near the surface because the production and ad/de-sorption rates have not changed, but the particulate phase falls out at a faster rate. This lesser [] at the surface has to be balanced by an increased [] at the floor. This is why a faster sinking rate increases the steady state [] at depth, as well as the slope along an x-isoline.

.. figure:: Pa-Th/images/2D0vel.png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/2D50yr[Th]0vel.png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/2D1vel.png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/2D50yr[Th]1vel.png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/2D2vel.png
.. figure:: /home/abellas/Documents/GEOTRACES/Pa-Th/images/2D50yr[Th]2vel.png


==============================
Coupled 2D Model: Steady State
==============================

**[Th]** 	
		- zero velocity: The magnitude and distribution of [Th] change very little between 40 and 100 years. After 50 years, 				the solution is 90% similar to the 100 year solution. The solution shows a constant [Th] over depths 4250 - 				5000m, but this is just a relic of the boundary condition that was not considered in the analytical solution.

		- single cell velocity: also takes 50 years

		- two cell velocity: also takes 50 years

**[Th] / [Pa]**	
		- zero velocity: takes longer than 80 years to reach steady state. Still have to run more code.

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



