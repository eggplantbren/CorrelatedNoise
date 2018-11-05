import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import scipy.linalg

# Seed RNG
#rng.seed(0)

# Create coordinate grid
n = 100
ii = np.arange(0, n)

# Fourier transform of real
y = rng.randn(n)
Y = np.fft.fft(y)/np.sqrt(n)

# After element n//2, repeats in reverse and conjugated, up to full length n
plt.figure(1)
plt.plot(np.real(Y), "o-") # sd = 1/sqrt(2)
                           # except for first element which has sd=1
plt.plot(np.imag(Y), "o-") # sd = 1/sqrt(2)
                           # except for first element which has sd=0
plt.axhline(0.0, linestyle="--")
plt.axvline(n//2, linestyle="--")

# Generate from FFT
Y = np.zeros(n, dtype="complex128")
for i in range(n):
    if i==0:
        Y[0] = rng.randn()
#        print(i)
    elif i <= n//2:
        Y[i] = (rng.randn() + 1j*rng.randn())/np.sqrt(2.0)
#        print(i)
    else:
        Y[i] = np.conj(Y[n-i])
#        print(i, n-i)

    # Multiply in an envelope
    if i <= n//2:
        Y[i] *= np.exp(-0.5*i**2/20.0**2)

# Inverse FFT to real data
y = np.fft.ifft(Y)/np.sqrt(n)

plt.figure(2)
plt.plot(np.real(Y), "o-")
plt.plot(np.imag(Y), "o-")
plt.axhline(0.0, linestyle="--")
plt.axvline(n//2, linestyle="--")

plt.figure(3)
plt.plot(np.real(y), "o-")
plt.plot(np.imag(y), "o-", alpha=0.2)
plt.show()

