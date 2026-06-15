"""
This file contains the optimization functions.
"""

import PyPhenom as ppi
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
import automation_location as location
import automation_autocorrelation as autocorrelation
import automation_fitting as fitting
import automation_consts as consts


# connects to SEM 
PhenomID = 'MVE086101-20057-F'
username = 'MVE08610120057F'
password = 'KN68X6HJQMT9'
phenom = ppi.Phenom(PhenomID, username, password)

# determines the blurring kernel used in a location to filter the image
size = autocorrelation.calibrate_kernel_size(phenom, consts.wd_val_min)

# the image counters
counter_wd = [0]
counter_st = [0]


def optimum_wd(wd, phenom = phenom, kernel_size = size, image_counter = counter_wd):
    """
    Returns the negative sum of the autocorrelation values for the four pixels surrounding the 
    center pixel given wd, a list containing a working distance.
    This is the objective function to optimize the working distance.

    :param wd: the working distance to set the SEM to
    :type wd: `tuple` of `int` or `float`
    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param kernel_size: the size of the blurring kernel
    :type kernel_size: an odd `int`
    :param image_counter: the number of pictures taken
    :type image_counter: `list` of a `int`
    :returns: the negative sum of the four center pixels in the autocorrelation
    :rtype: `float`
    """

    phenom.SetSemWD(wd[0]) # sets the working distance to wd[0]

    print(image_counter)
    if image_counter[0] % 10 == 0:
        phenom.SemAutoContrastBrightness()
    img_array, _ = location.take_image(phenom) # takes an image of the current location
    image_counter[0] += 1

    acdata = autocorrelation.filtered_autocorr(img_array, kernel_size) # calculates the filtered autocorrelation
    sums = (acdata[acdata.shape[0] // 2 - 1, acdata.shape[1] // 2] + 
            acdata[acdata.shape[0] // 2 + 1, acdata.shape[1] // 2] + 
            acdata[acdata.shape[0] // 2, acdata.shape[1] // 2 - 1] + 
            acdata[acdata.shape[0] // 2, acdata.shape[1] // 2 + 1]) # sums together the four center pixels
    
    return sums * (-1) # returns the negative of the sum


def auto_focus(phenom, wd_val_min, wd_val_max, initial_guess = None, calls = 50, points = 10):
    """
    Determines the working distance corresponding to a focused image using bayesian optimization. 
    Sets the working distance to the optimal value.
    Returns the optimal working distance.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param wd_val_min: the minimum working distance that is considered when optimizing
    :type wd_val_min: `float`
    :param wd_val_max: the maximum working distance that is considered when optimizing
    :type wd_val_max: `float`
    :param initial_guess: an initial guess for the optimal working distance
    :type initial_guess: `float` | None
    :param calls: the number of calls to optimum_wd
    :type calls: `int`
    :param points: the number of evaluations of optimum_wd with initialization points
    :type points: `int`
    :returns: the optimal working distance
    :rtype: `float`
    """

    res = gp_minimize(optimum_wd, # calls scikit's bayesian optimization 
                      [(wd_val_min, wd_val_max)], 
                      acq_func = "gp_hedge", 
                      x0 = initial_guess,
                      n_calls = calls, 
                      n_initial_points = points, 
                      verbose = False, 
                      random_state = 1234) 

    phenom.SetSemWD(res.x[0]) # sets the working distance to the optimum
    return res.x[0]


def optimum_stig(st, phenom = phenom, image_counter = counter_st):
    """
    Returns the astigmatism coefficient of a Zernike polynomial fitted to the autocorrelation data
    at the given stigmate st.
    This is the objective function to optimize the stigmate.

    :param st: (x, y) representing the x-stigmate and y-stigmate
    :type st: `tuple` of `floats`
    :param wd: the working distance to set the SEM to
    :type wd: `tuple` of `int` or `float`
    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param image_counter: the number of pictures taken
    :type image_counter: `list` of a `int`
    :returns: the square root of the squared sum of the zernike polynomial coefficients 
              oblique astigmatism and vertical astigmatism
    :rtype: `float`
    """

    xst, yst = st
    phenom.SetSemStigmate(ppi.Position(xst, yst))

    print(image_counter)
    if image_counter[0] % 10 == 0:
        phenom.SemAutoContrastBrightness()
    img_array, _ = location.take_image(phenom) # takes an image of the current location
    image_counter[0] += 1

    acdata = autocorrelation.auto_correlation(img_array) # calculates the autocorrelation for the image
    filtered_acdata = autocorrelation.remove_background_autocorr(acdata) # removes the autocorrelation background

    result = fitting.incoh_psf_fit(filtered_acdata, 1, 1, 1) # fits the data

    return np.sqrt(result.best_values['xstig'] ** 2 + result.best_values['ystig'] ** 2)


def auto_stig(phenom, xst_val_min, xst_val_max, yst_val_min, yst_val_max, 
              initial_guess = None, calls = 50, points = 10):
    """
    Determines the stigmate values corresponding to a focused image using bayesian optimization. 
    Sets the stigmate to the optimal values.
    Returns the optimal stigmate.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param xst_val_min: the minimum x-stigmate that is considered when optimizing
    :type xst_val_min: `float`
    :param xst_val_max: the maximum x-stigmate distance that is considered when optimizing
    :type xst_val_max: `float`
    :param yst_val_min: the minimum y-stigmate distance that is considered when optimizing
    :type yst_val_min: `float`
    :param yst_val_max: the maximum y-stigmate distance that is considered when optimizing
    :type yst_val_max: `float`
    :param initial_guess: an initial guess for the optimal working distance
    :type initial_guess: `tuple` of `float` | None
    :param calls: the number of calls to optimum_wd
    :type calls: `int`
    :param points: the number of evaluations of optimum_wd with initialization points
    :type points: `int`
    :returns: the optimal stigmate
    :rtype: `tuple` of `floats`
    """

    res = gp_minimize(optimum_stig, 
                      [(xst_val_min, xst_val_max), (yst_val_min, yst_val_max)], 
                      acq_func = "gp_hedge", 
                      x0 = initial_guess,
                      n_calls = calls, 
                      n_initial_points = points, 
                      verbose = False, 
                      random_state = 1234) 

    phenom.SetSemStigmate(ppi.Position(res.x[0], res.x[1]))

    return res.x[0], res.x[1]
