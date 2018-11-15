import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import numpy.random as rng
import scipy.linalg

# Seed RNG
rng.seed(0)

# Image dimensions
ni, nj = 100, 101

@jit
def make_grid():
    """
    Create a grid of indices.
    """
    ii = np.empty((ni, nj))
    jj = np.empty((ni, nj))
    for i in range(ni):
        for j in range(nj):
            ii[i, j] = i
            jj[i, j] = j
    return (ii, jj)

@jit
def unitary_fft2(y):
    """
    A unitary version of the fft2.
    """
    return np.fft.fft2(y)/np.sqrt(ni*nj)

@jit
def unitary_ifft2(y):
    """
    A unitary version of the ifft2.
    """
    return np.fft.ifft2(y)*np.sqrt(ni*nj)


# Create a grid
ii, jj = make_grid()


@jit
def make_psf(width):
    rsq = (ii - ni/2)**2 + (jj - nj/2)**2
    blur = np.exp(-rsq/width**2) #1.0 / (1.0 + rsq/width**2)**2
    blur = blur/np.sqrt(np.sum(blur**2))*np.sqrt(blur.size)
    return blur

@jit
def log_likelihood(width, data_fourier):
    psf = make_psf(width)
    psf_fourier = unitary_fft2(psf)
    dot_prod = np.real(data_fourier/psf_fourier\
                        *np.conj(data_fourier/psf_fourier))
    c = -0.5*data_fourier.size*np.log(2*np.pi)
    return c - 0.5*np.sum(np.log(np.real(psf_fourier*np.conj(psf_fourier))))\
                    - 0.5*np.sum(dot_prod)

# Some white noise
ns = rng.randn(ni, nj)
ns_fourier = unitary_fft2(ns)
#print(np.sum(ns**2))

# A kernel to blur the noise with to produce some data
psf = make_psf(1.3)
psf = np.fft.fftshift(psf)
psf_fourier = unitary_fft2(psf)

# Create the data
data_fourier = ns_fourier*psf_fourier
data = 5.7*unitary_ifft2(data_fourier).real

# Compare fourier/non-fourier calcs for independent data
#print(log_likelihood(1E-4, data_fourier))
#print(-data.size*0.5*np.log(2*np.pi) -0.5*np.sum(data**2))

np.savetxt("data.txt", data)
plt.imshow(data)
plt.title("Data")
plt.show()

logw = np.log(5.0) + np.linspace(-0.5, 0.5, 5001)
logl = np.empty(len(logw))
for i in range(len(logw)):
    logl[i] = log_likelihood(np.exp(logw[i]), data_fourier)
    print(i+1)

plt.plot(np.exp(logw), np.exp(logl - logl.max()), "o-")
plt.xlabel("Width")
plt.ylabel("Relative log likelihood")
print("max(logl) =", np.max(logl))
plt.show()

