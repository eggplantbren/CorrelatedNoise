import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.random as rng

# Domain size
nx, ny = 30, 30
n = nx*ny

# Parameter
alpha = 0.1

@numba.jit
def scalar(image):
    f = 0.0

    zs = np.zeros(n)
    k = 0
    for x in range(nx):
        for y in range(ny):
            zs[k] = image[x, y]

            if x > 0:
                zs[k] += -alpha*image[x-1, y]
            if x < nx - 1:
                zs[k] += -alpha*image[x+1, y]

            if y > 0:
                zs[k] += -alpha*image[x, y-1]
            if y < ny - 1:
                zs[k] += -alpha*image[x, y+1]

            k += 1

    return -0.5*np.sum(zs**2)

def matrix(image):
    M = np.zeros((n, n))
    for x in range(nx):
        for y in range(ny):

            # Diagonal terms
            k1 = x*ny + y
            M[k1, k1] += 1.0            

            # Neighbourly terms
            if x > 0:
                k2 = (x-1)*ny + y
                M[k1, k2] += -alpha
            if x < nx - 1:
                k2 = (x+1)*ny + y
                M[k1, k2] += -alpha

            if y > 0:
                k2 = x*ny + y-1
                M[k1, k2] += -alpha
            if y < ny - 1:
                k2 = x*ny + y+1
                M[k1, k2] += -alpha

    return M


## Initial conditions
image = np.zeros((nx, ny))
f = scalar(image)

#print(np.log(np.linalg.det(matrix(image))))
#plt.imshow(matrix(image))
#print(matrix(image))
#plt.show()

for i in range(100000):
    proposal = image.copy()
    x, y = rng.randint(nx), rng.randint(ny)
    proposal[x, y] += np.exp(3.0*rng.randn())*rng.randn()
    f_proposal = scalar(proposal)

    if rng.rand() <= np.exp(f_proposal - f):
        image = proposal
        f = f_proposal

    if (i+1) % 10000 == 0:
        print(i+1, f, image.std())
        plt.imshow(image, origin="lower")
        plt.show()

np.savetxt("data2.txt", image)
