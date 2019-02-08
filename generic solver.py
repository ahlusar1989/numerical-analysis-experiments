import matlab.engine
import os
import math
import numpy as np
import numpy.linalg as npla
import scipy
from scipy import sparse
from scipy import linalg
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource


def generate_A(k):
	"""Create the matrix for the problem on a k-by-k grid.
	Parameters: 
	  k: number of grid points in each dimension.
	Outputs:
	  A: the sparse k**2-by-k**2 matrix representing the finite difference approximation to Poisson's equation.
	"""
	# First make a list with one triple (row, column, value) for each nonzero element of A
	triples = []
	for i in range(k):
		for j in range(k):
			# what row of the matrix is grid point (i,j)?
			row = j + i * k
			# the diagonal element in this row
			triples.append((row, row, 4.0))
			# connect to left grid neighbor
			if j > 0:
				triples.append((row, row - 1, -1.0))
			# ... right neighbor
			if j < k - 1:
				triples.append((row, row + 1, -1.0))
			# ... neighbor above
			if i > 0:
				triples.append((row, row - k, -1.0))
			# ... neighbor below
			if i < k - 1:
				triples.append((row, row + k, -1.0))
	
	# Finally convert the list of triples to a scipy sparse matrix
	ndim = k * k
	rownum = [t[0] for t in triples]
	colnum = [t[1] for t in triples]
	values = [t[2] for t in triples]

	A = sparse.csr_matrix((values, (rownum, colnum)), shape = (ndim, ndim))
	
	return A

if __name__ == "__main__":

	eng = matlab.engine.start_matlab('nojvm')
	k = 7
	
	A = generate_A(k)
	A_dense = A.todense()
	A_matlab = matlab.double(A_dense.tolist())
	
	X, Y = np.meshgrid(range(1, k), range(1, k))

	X = matlab.double(X.tolist())
	Y = matlab.double(Y.tolist())

	X, Y, Z = eng.centralDifferencePoisson(0.5, A_matlab, nargout=3)


	X_p = np.asarray(X)
	Y_p = np.asarray(Y)
	Z_p = np.asarray(Z)

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	region = np.s_[5:49, 5:49]
	x, y, z = X_p[region], Y_p[region], Z_p[region]

	fig, ax = plt.subplots(subplot_kw = dict(projection='3d'))

	ls = LightSource(270, 45)
	# To use a custom hillshading mode, override the built-in shading and pass
	# in the rgb colors of the shaded surface calculated from "shade".
	rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
	surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
	                       linewidth=0, antialiased=False, shade=False)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	fig.savefig("surface3d_frontpage.png", dpi=25)  # results in 160x120 px image

	# #----- Static image
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	# plt.tight_layout()
	# ax.view_init(20, -106)

	# fig.savefig("equation.png", dpi=25)  # results in 160x120 px image
	eng.quit()



