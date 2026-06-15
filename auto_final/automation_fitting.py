"""
This file contains all the functions related to fitting data.
"""

import numpy as np
from math import factorial
from scipy.fft import fft2, fftshift
from scipy.signal import fftconvolve
from lmfit import Model, create_params


# Define basic Zernike polynomials (on the unit disk)
def zernike_radial(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        coeff = (-1)**k * factorial(n - k)
        coeff /= (factorial(k) *
                  factorial((n + abs(m)) // 2 - k) *
                  factorial((n - abs(m)) // 2 - k))
        R += coeff * rho**(n - 2 * k)
    return R


def zernike(n, m, rho, theta):
    if m > 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    elif m < 0:
        return zernike_radial(n, -m, rho) * np.sin(-m * theta)
    else:
        return zernike_radial(n, 0, rho)


# Mapping Noll index to (n, m)
NOLL_MAP = {
    1:  (0,  0),   # Piston (constant phase offset)
    2:  (1, -1),   # Tilt X (Tip)
    3:  (1,  1),   # Tilt Y (Tilt)
    4:  (2, -2),   # Oblique Astigmatism
    5:  (2,  0),   # Defocus
    6:  (2,  2),   # Vertical Astigmatism
    7:  (3, -3),   # Vertical Trefoil
    8:  (3, -1),   # Vertical Coma
    9:  (3,  1),   # Horizontal Coma
    10: (3,  3),   # Horizontal Trefoil
    11: (4, -4),   # Oblique Tetrafoil
    12: (4, -2),   # Oblique Secondary Astigmatism
    13: (4,  0),   # Primary Spherical Aberration
    14: (4,  2),   # Vertical Secondary Astigmatism
    15: (4,  4),   # Vertical Tetrafoil
}


def incoherent_zernike_psf(N=256, defocus = None, xstig = None, ystig = None): # zernike_coeffs=None):
    if defocus is None: # if zernike_coeffs is None:
        zernike_coeffs = {
            5:2,  # defocus
        }

    else:
        zernike_coeffs = {
            5:defocus
        }

    if xstig is not None:
        zernike_coeffs[6] = xstig

    if ystig is not None:
        zernike_coeffs[4] = ystig

    x = np.linspace(- 1, 1, N)
    X, Y = np.meshgrid(x, x)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    aperture = rho <= 1 #boolean array

    wavefront = np.zeros_like(rho)

    for j, coeff in zernike_coeffs.items():
        n, m = NOLL_MAP[j]
        Z = zernike(n, m, rho, theta)
        Z[~aperture] = 0
        wavefront += coeff * Z

    # Pupil function with aberration
    phase = 2 * np.pi * wavefront # this could be scaled by the wavelength.
    # suggest to not to do this in this case that adds to the confusion ==> this is not the same as a coherent point source!

    pupil = np.exp(1j * phase) 
    pupil[~aperture] = 0

    # Coherent PSF
    h = fftshift(fft2(pupil)) #The point spread function (PSF), which is given by the square of the absolute value of the 
    psf_coh = np.abs(h)**2    #Fourier transform of the wavefront, shows what the intensity image of a point source would look like.

    # Incoherent PSF: autocorrelation of coherent intensity
    psf_incoh = fftconvolve(psf_coh, psf_coh, mode='same')
    psf_incoh /= psf_incoh.sum()

    return psf_incoh


def incoh_psf_fit(ac, df, xst, yst):
    model = Model(incoherent_zernike_psf)

    pars = create_params(N = dict(value = 256, vary = False),
                         defocus = dict(value = df, vary = True, min = 0.1, max=5),
                         xstig = dict(value = xst, vary = True, min=-4, max=4),
                         ystig = dict(value = yst, vary = True, min=-4, max=4))

    result = model.fit(ac, params = pars) # uses the above parameters and  model to fit the data

    return result