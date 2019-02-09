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
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def matprint(mat, fmt="g"):
	col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
	for x in mat:
		for i, y in enumerate(x):
			print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
		print("")

def rhs(x, y):
	# Element-wise multiplication
	return np.multiply(x, (x-y)**3)

def bc_dirichlet(x, y, m):

	bc = np.zeros((m+1, m+1))
	bc[:, 0] = y[:, 0]**2
	bc[:, m] = np.ones((m+1, 1)).ravel()
	bc[0, :] = x[0, :]**3
	bc[m, :] = np.ones((1, m+1)).ravel()
	return bc


def generate_sparse_matrix(m):

	main_diag = 2 * np.ones((m - 1, 1)).ravel()
	off_diag = -1 * np.ones((m - 2, 1)).ravel()

	diagonals = [main_diag, off_diag, off_diag]
	b1 = sparse.diags(diagonals, [0, -1, 1], shape=(m - 1, m - 1)).toarray()
	sB = sparse.csc_matrix(b1)
	
	I = sparse.eye(m - 1, format="csr").toarray()
	sI = sparse.csc_matrix(I)

	a1 = sparse.kron(sI, sB).toarray()
	a2 = sparse.kron(sB, sI).toarray()
	mat = sparse.csc_matrix(a1 + a2)
	return mat

	
def hwk_1_part_1():
	M = 7
	a = 0.0
	b = 1.0

	h = (b - a) / M

	x1 = np.linspace(a, b, M + 1)

	X, Y = np.meshgrid(x1, x1)

	#----- Right hand side 
	f = rhs(X, Y)
	f = np.array(f.T)[1 : M, 1 : M].reshape(((M - 1) * (M - 1), 1))

	#----- Boundary conditions
	G = bc_dirichlet(X, Y, M)

	#----- Rearranges matrix G into an array
	g = np.zeros(((M-1)**2, 1))
	g[0:M-1, 0] = G[1:M, 0]
	g[(M-1)**2-M+1:M**2, 0] = G[1:M, M]
	g[0:M**2:M-1, 0] = g[0:M**2:M-1, 0] + G[0, 1:M]
	g[M-2:M**2:M-1, 0] = g[M-2:M**2:M-1, 0] + G[M, 1:M]

	A = generate_sparse_matrix(M)

	#----- Solve A*x=b => x=A\b
	U = spsolve(A, f*(h**2) + g)
	U = U.reshape((M-1, M-1)).T	

	#----- Solve A*x=b => x=A\b
	U = spsolve(A, f*(h**2)+g)
	U = U.reshape((M-1, M-1)).T

	G[1:M, 1:M] = U

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	surf = ax.plot_surface(X, Y, G, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)

	#----- Static image
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('G')
	plt.tight_layout()
	ax.view_init(20, -106)


def make_A(k):
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


def make_b(k, top = 0, bottom = 0, left = 0, right = 0):
	"""Create the right-hand side for the problem on a k-by-k grid.
	Parameters: 
	  k: number of grid points in each dimension.
	  top: list of k values for top boundary (optional, defaults to 0)
	  bottom: list of k values for bottom boundary (optional, defaults to 0)
	  left: list of k values for top boundary (optional, defaults to 0)
	  right: list of k values for top boundary (optional, defaults to 0)
	Outputs:
	  b: the k**2 element vector (as a numpy array) for the rhs of the Poisson equation with given boundary conditions
	"""
	# Start with a vector of zeros
	ndim = k * k
	b = np.zeros(shape = ndim)
	
	# Fill in the four boundaries as appropriate
	b[0        : k       ] += top
	b[ndim - k : ndim    ] += bottom
	b[0        : ndim : k] += left
	b[k - 1      : ndim : k] += right
	
	return b

if __name__ == "__main__":
	# print(hwk_1_part_1())
	print(hwk_1_part_1())
