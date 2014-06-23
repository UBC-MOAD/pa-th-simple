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

		i = g.ilo + 1

		while (i <= g.ihi - 1):

                        j = numpy.arange(g.jlo + 1, g.jhi, 1, dtype = int)

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
	                i += 1

                # store the (time) updated solution
                g.a[:] = anew[:]
                h.a[:] = bnew[:]
                t += dt
        return g, h


