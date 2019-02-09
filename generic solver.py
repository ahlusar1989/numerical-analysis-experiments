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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.io import savemat

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
from IPython.core.display import display, HTML
import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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
    bc[:, m] = np.ones((m + 1, 1)).ravel()
    bc[0, :] = x[0, :]**3
    bc[m, :] = np.ones((1, m+1)).ravel()
    return bc


def generate_A(k):
    """Create the matrix for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**2-by-k**2 matrix representing the finite difference approximation to Poisson's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []
    for i in range(k):
        for j in range(k):
            # what row of the matrix is grid point (i, j)?
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
    ndim = k*k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = sparse.csr_matrix((values, (rownum, colnum)), shape = (ndim, ndim))
    matprint(A.toarray())
    return A  
  
  
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
    matprint(mat.toarray())
    return mat

def plot3D(X, Y, Z, height=600, xlabel = "X", ylabel = "Y", zlabel = "Z", initialCamera = None):

    options = {
        "width": "100%",
        "style": "surface",
        "showPerspective": True,
        "showGrid": True,
        "showShadow": False,
        "keepAspectRatio": True,
        "height": str(height) + "px"
    }

    if initialCamera:
        options["cameraPosition"] = initialCamera
        
    data = [ {"x": X[y,x], "y": Y[y,x], "z": Z[y,x]} for y in range(X.shape[0]) for x in range(X.shape[1]) ]
    visCode = r"""
       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />
       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>
       <div id="visualization"></div>
       <script type="text/javascript">
        var data = new vis.DataSet();
        data.add(""" + json.dumps(data) + """);
        var options = """ + json.dumps(options) + """;
        var container = document.getElementById("visualization");
        var graph3d = new vis.Graph3d(container, data, options);
        graph3d.on("cameraPositionChange", function(evt)
        {
            elem = document.getElementById("pos");
            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;
        });
       </script>
    """
    htmlCode = "<iframe srcdoc='"+visCode+"' width='100%' height='" + str(height) + "px' style='border:0;' scrolling='no'> </iframe>"
    display(HTML(htmlCode))


if __name__ == "__main__":


	M = 5
	a = 0.0
	b = 4

	h = (b - a)/M
	x1 = np.linspace(a, b, M + 1)

	X, Y = np.meshgrid(x1, x1)

	#----- Right hand side 
	f = rhs(X, Y)
	f = np.array(f.T)[1:M, 1:M].reshape(((M - 1)*(M - 1), 1))

	#----- Boundary conditions
	G = bc_dirichlet(X, Y, M)

	#----- Rearranges matrix G into an array
	g = np.zeros(((M - 1)**2, 1))
	g[0: M - 1, 0] = G[1 : M, 0]
	g[(M - 1)**2 - M + 1: M**2, 0] = G[1 : M, M]
	g[0 : M**2: M - 1, 0] = g[0 : M**2: M - 1, 0] + G[0, 1: M]
	g[M - 2:M**2: M - 1, 0] = g[M - 2:M**2:M - 1, 0] + G[M, 1: M]
	print("Iterative version is above ==========>")
	A_iterative = generate_A(4)
	print()
	A = generate_sparse_matrix(M)

	print("Non-iterative method ==========>")
	A_to_dense = A.todense() # for debugging

	#----- Solve A*x=b => x=A\b
	U = spsolve(A, f*(h**2)+g)
	U = U.reshape((M - 1, M - 1)).T

	G[1:M, 1:M] = U

	# eng = matlab.engine.start_matlab('nojvm')
	# k = 7
	# A = generate_A(k)
	# savemat('A_temp', {'A' : A})
	# A_dense = A.todense()
	# A_matlab = matlab.double(A_dense.tolist())
	# # X, Y = np.meshgrid(range(1, k), range(1, k))

	# # X = matlab.double(X.tolist())
	# # Y = matlab.double(Y.tolist())

	# X, Y, Z = eng.matrixSolver(0.5, nargout=3)


	# X_p = np.nan_to_num(np.asarray(X))
	# # print(X_p)
	# # print()
	# Y_p = np.nan_to_num(np.asarray(Y))
	# # print(Y_p)
	# Z_p = np.nan_to_num(np.asarray(Z))
	# # print("=======")
	# # print(Z)
	# # fig = plt.figure()
	# fig = plt.figure(figsize=(20,20))

	# ax = fig.gca(projection='3d')

	# # # region = np.s_[5:49, 5:49]
	# # # x, y, z = X_p[region], Y_p[region], Z_p[region]

	# # # fig, ax = plt.subplots(subplot_kw = dict(projection='3d'))

	# surf = ax.plot_surface(Y_p,	X_p, Z_p,
	#                        linewidth=0, antialiased=False)

	# # #----- Static image
	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_zlabel('Z')
	# # ax.view_init(20, -106)


	# fig.savefig("equation.png")  # results in 160x120 px image
	# eng.quit()



