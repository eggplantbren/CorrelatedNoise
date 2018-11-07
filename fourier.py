import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import numpy.random as rng
import scipy.linalg

# Seed RNG
rng.seed(0)

# Image dimensions
ni, nj = 200, 300

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
    blur = np.exp(-0.5*(ii - ni/2)**2/width**2 - 0.5*(jj - nj/2)**2/width**2)
    blur[ni//2, nj//2] += 1E-3
    blur = blur/np.sqrt(np.sum(blur**2))*np.sqrt(blur.size)
    return blur

@jit
def log_likelihood(width, data_fourier):
    psf = make_psf(width)
    psf_fourier = unitary_fft2(psf)
    sds = np.abs(psf_fourier.real)/np.sqrt(2)
    ratio = data_fourier.real/sds
    return -np.sum(np.log(sds)) \
                - 0.5*np.sum(ratio**2)


# Some white noise
ns = rng.randn(ni, nj)
ns_fourier = unitary_fft2(ns)
#print(np.sum(ns**2))

# A kernel to blur the noise with to produce some data
psf = make_psf(5.0)
psf = np.fft.fftshift(psf)
psf_fourier = unitary_fft2(psf)

# Create the data
data_fourier = ns_fourier*psf_fourier
data = unitary_ifft2(data_fourier).real
#print(np.sum(data**2))

plt.imshow(data)
plt.title("Data")
plt.show()

logw = np.linspace(0.0, 2.0, 10001)
logl = np.empty(len(logw))
for i in range(len(logw)):
    logl[i] = log_likelihood(np.exp(logw[i]), data_fourier)
    print(i+1)

np.savetxt("data.txt", data)

plt.plot(np.exp(logw), logl, "o-")
plt.xlabel("Width")
plt.ylabel("Error")
plt.show()

