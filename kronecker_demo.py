import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng

ny, nx = 5, 6

y = np.arange(0, ny)
x = np.arange(0, nx)

# x-component of covariance
Cx = np.zeros((nx, nx))
for i in range(nx):
    for j in range(nx):
        Cx[i, j] = np.exp(-0.5*((x[i] - x[j])/1.0)**2)
    Cx[i, i] += 1E-6

# y-component of covariance
Cy = np.zeros((ny, ny))
for i in range(ny):
    for j in range(ny):
        Cy[i, j] = np.exp(-0.5*((y[i] - y[j])/1.0)**2)
    Cy[i, i] += 1E-6

# Eigen decompositions
Ex, Vx = np.linalg.eig(Cx)
Ey, Vy = np.linalg.eig(Cy)

# Explicit full covariance matrix
n = nx*ny
C = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dx = x[i % nx] - x[j % nx]
        dy = y[i // nx] - y[j // nx]
        rsq = dx**2 + dy**2
        C[i, j] = np.exp(-0.5*rsq/1.0**2)
    C[i, i] += 1E-6
# C would equal np.kron(Cy, Cx) if it weren't for the extra diagonal bit.

# Some data
L = np.linalg.cholesky(C)
ns = rng.randn(n)
ys = C @ ns
img = ys.reshape((ny, nx))
plt.imshow(img)
plt.show()

# Explicit solve
logl0 = -n*0.5*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(C)) \
                - 0.5*np.dot(ys, np.linalg.solve(C, ys))

# Cholesky solve
logl1 = -n*0.5*np.log(2*np.pi) - np.sum(np.log(np.diag(L))) \
                - 0.5*np.dot(ys, ns)

# Kronecker solve
E = np.kron(Ey, Ex) # All eigenvalues of C

# Dot product of the data against eigenvectors of C
# = the data represented in that basis
coeffs = np.zeros(n)
k = 0
for i in range(ny):
    for j in range(nx):
        V = np.kron(Vy[:,i], Vx[:,j])
        coeffs[k] = np.dot(ys, V)
        k += 1

logl2 = -n*0.5*np.log(2*np.pi) - 0.5*np.sum(np.log(E)) \
                - 0.5*np.dot(coeffs, coeffs/E)

# Eigendecomposition: C = V @ diag(E) @ V.T since C is symmetric V.T = Vinv

# First two should be equal, third should be close.
print(logl0, logl1, logl2)
plt.plot(coeffs)
plt.show()

