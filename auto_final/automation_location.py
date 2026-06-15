"""
This file contains all the functions related to the location of the sample and repositioning.
"""

import PyPhenom as ppi
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import automation_autocorrelation


def take_image(phenom, xdim = 256, ydim = 256, stack = 4, HDR = False, scale = 0.5):
    """
    Takes an image of the current location shown on the SEM and returns the image array
    and the image metadata.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param xdim: the number of pixels in the x-direction in the image
    :type xdim: `int`
    :param ydim: the number of pixels in the y-direction in the image
    :type ydim: `int`
    :param stack: the number of pictures to average into one picture
    :type stack: `int`
    :param HDR: adds more contrast to the image
    :type HDR: `boolean`
    :param scale: the zoom in the image
    :type scale: `int` or `float`
    :returns: the image as a numpy array and a tuple of image metadata
    :rtype: `tuple` of `numpy.ndarray` and `tuple`
    """

    # stack can be et to a value < 1 to zoom in on a part of the image - centered

    # phenom.SemAutoContrastBrightness()
    acq = phenom.SemAcquireImage(xdim, ydim, stack, ppi.DetectorMode.All, HDR, scale)

    img_array = np.array(acq.image) # converts image into an array

    return img_array, acq.metadata


def reposition_working_distance(phenom):
    """
    Repositions the SEM for working distance optimization.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    """

    img_array, meta = take_image(phenom) # takes picture of current location
    entropy_filtered = entropy(img_array, disk(5)) # calculates the entropy

    flat_index = np.argmax(entropy_filtered)
    y, x = np.unravel_index(flat_index, entropy_filtered.shape)
    new_center = (x, y)

    # moves to the new center
    center = (img_array.shape[0] // 2, img_array.shape[1] // 2)
    px_distance = [center[0] - new_center[0], center[1] - new_center[1]]
    m_distance = [px_distance[0] * meta.pixelSize.width, px_distance[1] * meta.pixelSize.height]
    phenom.MoveBy(-1 * m_distance[0], m_distance[1])


def pixelposition(img, meta, xix, yix, ratio):
    """
    Obtain position on stage for pixel with index xix, yix in image img.
    Note that ratio > 1 the ratio HFW_old/HFW_new still has to be set.

    :param img: a numpy array representing an image
    :type img: `numpy.ndarray`
    :param meta: the metadata for the image
    :type meta: `tuple`
    :param xix: the x index in img
    :type xix: `int` or `float`
    :param yix: the y index in img
    :type yix: `int` or `float`
    :param ratio: the ratio between the old HFW and the new HFW
    :type ratio: `float`
    :returns: the position to move to
    :rtype: `phenom.Position`

    Afterwards: 
        phenom.MoveTo(pixelposition(img, meta, xix, yix))
        phenom.SetHFW(phenom.GetHFW()/ratio)
    """

    xpix = xix
    ypix = img.shape[1] - yix

    x, y = meta.position.x, meta.position.y

    x_size = img.shape[0]/ratio
    y_size = img.shape[1]/ratio

    xnew = x + (xpix + x_size//2 - img.shape[0] // 2) * meta.pixelSize.width
    ynew = y + (ypix - y_size//2 - img.shape[1] // 2) * meta.pixelSize.height

    return ppi.Position(xnew, ynew)


def find_stig_opt_position(img, meta, x_size = 64, y_size = 64):
    """
    Finds the section in the current location to move to for stigmate optimization.

    :param img: a numpy array representing an image
    :type img: `numpy.ndarray`
    :param meta: the metadata for the image
    :type meta: `tuple`
    :param x_size: the size in pixels of the x-dimension sub-image
    :type x_size: `int`
    :param y_size: the size in pixels of the y-dimension sub-image
    :type y_size: `int`
    :returns: the (x, y) index of the section to move to, the coordinates to move to, 
              the ratio between the old HFW and the new HFW, and the auto-correlation data for every sub-image
    :rtype: `tuple` of `tuple`, `phenom.Position`, `float`, and `numpy.ndarray`
    """

    acdata = np.zeros((img.shape[0]-x_size+1, img.shape[1]-y_size+1, x_size, y_size))

    for i in range(0, img.shape[0]-x_size+1):
        for j in range(0, img.shape[1]-y_size+1):
            window = img[i:i+x_size,j:j+y_size] # each individual window
            result = automation_autocorrelation.auto_correlation(window)
            acdata[i, j] = result

    xix, yix = np.unravel_index(np.argmin(acdata.sum(axis=(2,3))), acdata.shape[:2])
    ratio = img.shape[0] / x_size
    
    return (xix, yix), pixelposition(img, meta, xix, yix, ratio), ratio, acdata


def reposition_stigmate(phenom):
    """
    Repositions the SEM for stigmate optimization.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    """

    img, meta = take_image(phenom)
    (xix, yix), optpos, ratio, acdata = find_stig_opt_position(img, meta)
    phenom.MoveTo(optpos)
    phenom.SetHFW(phenom.GetHFW()/ratio)