*****************
Preliminary Model
*****************

The preliminary model is composed to two major working parts: the Th2D module, which contains all the functions used to produce the results, and the Coupled 2D Model notebook, which displays the results. 

`Th2D module <http://ccar-modeling-pa-th.readthedocs.org/en/latest/prelim-model/Th2Dmodule/.html>`_

.. function:: adflow(T, V, u, nz, nx, k_ad, k_de, Q, flowfig)
	"""
	Compute and store the dissolved and particulate [Th] profiles, write them to a file, plot the results.

	:arg T: scale for tmax such that tmax = T*(g.zmax - g.zmin)/S 
	:typeT: int

	:arg V: scale for ux, uz, which are originally order 1.
	:type V: int

	:arg u: 3D tensor of shape (nz, nx, 2), z component of velocity in (:, :, 1), x component of velocity in (:, :, 2) 
	:type u: float

	:arg nz: number of grid points in z dimension
	:type: int

	:arg nx: number of grid points in x dimension
	:type: int

	:arg k_ad: nz x nx adsorption rate matrix
	:type k_ad: float

	:arg k_de: nz x nx adsorption rate matrix
	:type k_de: float

	:arg adscheme: function to implement the desired advection scheme 

	"""

   :module: Th2D.py

   Return a line of text input from the user.


