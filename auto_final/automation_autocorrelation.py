"""
This contains the functions related to the calculation of the autocorrelation of an image.
"""

import PyPhenom as ppi
import numpy as np
import cv2
from skimage.draw import polygon
from scipy.signal import savgol_filter
import automation_location as location


def gradient_2d(data):
    """
    Returns the 2D gradient of the given data.

    :params data: a 2D numpy array
    :type data: `numpy.ndarray`
    :returns: 2D gradient of the data
    :rtype: `numpy.ndarray`
    """

    grad_y, grad_x = np.gradient(data)
    combined = np.zeros_like(data)

    for y in range(grad_y.shape[0]):
        for x in range(grad_x.shape[0]):
                combined[x][y] = np.sqrt(grad_x[x][y]**2 + grad_y[x][y]**2)

    return combined


def slice_with_slope(data, x_change, y_change):
    """
    Returns a tuple of the slice of the data based off of the slope y_change / x_change and the points of 
    the original data contained in the slice.

    :params data: a 2D numpy array
    :type data: `numpy.ndarray`
    :params x_change: the number of pixels in the x-direction to travel before moving in the y-direction
    :type x_change: `int`
    :params y_change: the number of pixels in the y-direction to travel before moving in the x-direction
    :type y_change: `int`
    :returns: a slice of the data and the sampled points
    :rtype: `tuple` of `lists` 
    """

    data_slice = []
    pts_sampled = []

    # use slicing if in x or y axis
    if x_change == 0:
        if np.sign(y_change) == 1:
            data_slice = data[data.shape[0] // 2, data.shape[1] // 2:] 
            for y in range(data.shape[1] // 2):                            
                pts_sampled.append((data.shape[0] // 2, data.shape[1] // 2 + y))
        else:
            data_slice = data[data.shape[0] // 2, :data.shape[1] // 2]
            data_slice = data_slice[::-1]
            for y in range(data.shape[1] // 2):
                pts_sampled.append((data.shape[0] // 2, data.shape[1] // 2 - y))
    
    elif y_change == 0:
        if np.sign(x_change) == 1:
            data_slice = data[data.shape[0] // 2:, data.shape[1] // 2]
            for x in range(data.shape[0] // 2):
                pts_sampled.append((data.shape[0] // 2 + x, data.shape[1] // 2))
        else:
            data_slice = data[:data.shape[0] // 2, data.shape[1] // 2]
            data_slice = data_slice[::-1]
            for x in range(data.shape[0] // 2):
                pts_sampled.append((data.shape[0] // 2 - x, data.shape[1] // 2))

    # if not in x or y axis, take points along the desired slope
    else:
        center = (data.shape[0] // 2, data.shape[1] // 2)
        x_pos = center[0]
        y_pos = center[1]

        while (x_pos + x_change < data.shape[0] and x_pos + x_change >= 0 and
               y_pos + y_change < data.shape[1] and y_pos + y_change >= 0):
            for x in range(np.abs(x_change)): 
                pts_sampled.append((x_pos, y_pos))
                data_slice.append(data[x_pos, y_pos])
                if x_change > 0:
                    if x < x_change:
                        x_pos += 1
                else:
                    if x < np.abs(x_change):
                        x_pos -= 1

            for y in range(np.abs(y_change)):
                pts_sampled.append((x_pos, y_pos))
                data_slice.append(data[x_pos, y_pos])
                if y_change > 0:
                    if y < y_change:
                        y_pos += 1
                else:
                    if y < np.abs(y_change):
                        y_pos -= 1

    return data_slice, pts_sampled


def locate_center_peak(data):
    """
    Returns a tuple of the x-coordinates and the y-coordinates in the peak.

    :params data: a 2D numpy array
    :type data: `numpy.ndarray`
    :returns: a tuple of lists of x-coordinates and y-coordinates
    :rtype: `tuple` of `lists`
    """

    dir_to_sample = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    regions = []
    pts = []
    for pt in dir_to_sample:
        sliced_region, sliced_pts = slice_with_slope(data, pt[0], pt[1])
        regions.append(sliced_region)
        pts.append(sliced_pts)

    max_pts = []
    for index in range(len(regions)):
        max_arg_index = np.argmax(regions[index])
        max_pts.append(pts[index][max_arg_index])

    max_pts = np.array(max_pts)
    x_coords = max_pts[:, 0]
    y_coords = max_pts[:, 1]
    peak_x, peak_y = polygon(x_coords, y_coords)

    return peak_x, peak_y


def interpolate_center_peak(data, peak_x, peak_y):
    """
    Interpolates the center peak found with locate_center_peak.

    :params data: a 2D numpy array
    :type data: `numpy.ndarray`
    :params peak_x: the x-coordinates in the peak
    :type peak_x: `list` of `floats`
    :params peak_y: the y-coordinates in the peak
    :type peak_y: `list` of `floats`
    """

    for coord in range(peak_x.shape[0]):
        placeholder = [peak_x.min() - 1, peak_y.min() - 1]
        data[peak_x[coord]][peak_y[coord]] = data[placeholder[0], placeholder[1]]


def find_first_min(data):
    """
    Returns the index corresponding to the first local minimum.

    :params data: a list of points
    :type data: `list`
    :returns: the index of the first local minimum
    :rtype: `int`
    """

    for ind in range(data.shape[0] - 1):
        if data[ind] < data[ind + 1]:
            return ind
        
    return data.shape[0] - 1


def locate_boundary(gradient_data, peak_x, peak_y):
    """
    Returns a tuple (x, y) of the location of the first inflection point.

    :params gradient_data: a 2D numpy array
    :type gradient_data: `numpy.ndarray`
    :params peak_x: the x-coordinates in the peak
    :type peak_x: `list` of `floats`
    :params peak_y: the y-coordinates in the peak
    :type peak_y: `list` of `floats`
    :returns: the inflection point location
    :rtype: `tuple` of `ints`
    """

    boundary_coords = []

    x_coord_slopes = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_coord_slopes = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for x_slope in x_coord_slopes:
        for y_slope in y_coord_slopes:
            orig_region, orig_pts = slice_with_slope(gradient_data, x_slope, y_slope)

            indices_to_remove = []
            for pt in range(peak_x.shape[0]):
                if (peak_x[pt], peak_y[pt]) in orig_pts:
                    indices_to_remove.append((peak_x[pt], peak_y[pt]))

            removal_positions = []
            for index, coord in enumerate(orig_pts):
                for removal in indices_to_remove:
                    if coord[0] == removal[0] and coord[1] == removal[1]:
                        removal_positions.append(index)

            no_peak_region = np.delete(orig_region, removal_positions)
            new_region_ind = find_first_min(no_peak_region)
            new_region = no_peak_region[new_region_ind:]

            window_size = int(new_region.shape[0] * 0.67)
            
            denoised_new_region = savgol_filter(new_region, window_size, 7)
            max_pos = orig_pts[np.argmax(denoised_new_region) + len(indices_to_remove) + new_region_ind]
            boundary_coords.append(max_pos)

    return boundary_coords


def get_ellipse_parameters(ellipse_coords):
    """
    Returns the parameters of the ellipse that fits the given coordinates.

    :param ellipse_coords: the coordinates of the center shape
    :type ellipse_coords: `list`
    :returns: ellipse parameters: center, major_semi_axis, minor_semi_axis, angle
    :rtype: `tuple`
    """

    ellipse = np.array(ellipse_coords)
    ellipse_data = cv2.fitEllipse(ellipse) 
    (center, axes_lengths, angle) = ellipse_data

    xc, yc = center
    major_axis_len, minor_axis_len = axes_lengths

    major_semi_axis = major_axis_len / 2
    minor_semi_axis = minor_axis_len / 2

    return center, major_semi_axis, minor_semi_axis, angle


def create_ellipse_array(x_center, y_center, semi_major, semi_minor, angle_rad, num_points=100):
    """
    Returns a numpy array of points representing an ellipse.

    :param x_center: x-coordinate of the ellipse center
    :type x_center: `float`
    :param y_center: t-coordinate of the ellipse center
    :type y_center: `float`
    :param semi-major: length of the semi-major axis
    :type semi_major: `float`
    :param semi-minor: length of the semi-minor axis
    :type semi_minor: `float`
    :param angle_rad: rotation angle of the ellipse in radians
    :type angle_rad: `float`
    :param num_points: number of points to generate for the ellipse
    :type num_points: `int`
    :returns: a 2D array where each row is an (x, y) coordinate of a point on the ellipse
    :rtype: `numpy.ndarray`
    """

    grid = np.linspace(0, 2 * np.pi, num_points)

    # Unrotated ellipse points centered at origin
    x_unrotated = semi_major * np.cos(grid)
    y_unrotated = semi_minor * np.sin(grid)

    # Apply rotation
    x_rotated = x_unrotated * np.cos(angle_rad) - y_unrotated * np.sin(angle_rad)
    y_rotated = x_unrotated * np.sin(angle_rad) + y_unrotated * np.cos(angle_rad)

    # Translate to the specified center
    x_coords = x_rotated + x_center
    y_coords = y_rotated + y_center

    return np.array([x_coords, y_coords]).T


def calibrate_kernel_size(phenom, wd_val_min):
    """
    Returns the kernel size used to apply a gaussian blur.

    :param phenom: a Phenom object
    :type phenom: `Phenom`
    :param wd_val_min: the minimum working distance that the SEM can image at
    :type wd_val_min: `float`
    :returns: the kernel size used in blurring
    :rtype: an odd `int`
    """

    sums = -1
    kernel_size = 5

    phenom.SetSemStigmate(ppi.Position(0, 0)) # sets the stigmate to (0, 0) for sort of optimized stig
    phenom.SetHFW(ppi.MagnificationToFieldWidth(10000, 0.52)) # sets magnification to 10000
    phenom.SetSemWD(wd_val_min) # sets the working distance to wd_val_min

    img_array, _ = location.take_image(phenom) # takes image of current location

    while np.round(sums, 4) != 0:
        blurred_img = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0) # blurs the image
        filtered_data = cv2.Canny(blurred_img, 50, 150) # applies a canny filter to the blurred image

        sums = filtered_data.sum() # sums up the pixel values in the filtered image
        
        if np.round(sums, 4) != 0: # if the sum is not 0, add 2 to the kernel size and repeat
            kernel_size = kernel_size + 2

    return kernel_size


def auto_correlation(frame):
    """
    Returns the computed auto_correlation for a given frame.

    :param frame: a 2D numpy array representing an SEM image
    :type frame: `numpy.ndarray'
    :returns: the auto_correlation of the frame
    :rtype: `numpy.ndarray`
    """

    ft = np.fft.fft2(frame)
    autocorr = np.fft.ifft2(ft * np.conjugate(ft))
    return np.fft.fftshift(autocorr).real


def filtered_autocorr(img_array, kernel_size):
    """
    Returns the auto_correlation array for the given image array after applying a blur 
    (given the blurring kernel kernel_size) and a canny filter to img_array.

    :param img_array: a 2D numpy array representing an image
    :type img_array: `numpy.ndarray`
    :param kernel_size: the size of the blurring kernel
    :type kernel_size: an odd `int`
    :returns: the auto_correlation after filtering the image
    :rtype: `numpy.ndarray`
    """
    
    blurred_img = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0) # blurs the image
    filtered_data = cv2.Canny(blurred_img, 50, 150) # sets the values of edges = 255, everything else = 0
    acdata = auto_correlation(filtered_data) # calculates the auto_correlation data

    return acdata


def remove_background_autocorr(data):
    """
    Returns the auto_correlation with the background removed.

    :params data: a 2D numpy array of an auto_correlation
    :type data: `numpy.ndarray`
    :returns: the auto_correlation with the background removed
    :rtype: `numpy.ndarray`
    """

    ac = data.copy()
    grad1 = gradient_2d(ac) # calculates the amplitudes of first and second derivatives
    grad2 = gradient_2d(grad1)

    peak_x, peak_y = locate_center_peak(grad2)
    interpolate_center_peak(ac, peak_x, peak_y)

    new_grad1 = gradient_2d(ac)

    boundary_coords = locate_boundary(new_grad1, peak_x, peak_y)
    center, major_semi_axis, minor_semi_axis, angle = get_ellipse_parameters(boundary_coords)
    ellipse_points = create_ellipse_array(x_center=center[0], y_center=center[1], semi_major=major_semi_axis, 
                                          semi_minor=minor_semi_axis, angle_rad= angle * np.pi / 180, num_points=200)
    
    ellipse_points = np.array(ellipse_points)
    ellipse_x, ellipse_y = polygon(ellipse_points[:, 0], ellipse_points[:, 1])

    if (ellipse_x.min() >= 0 and ellipse_x.min() < ac.shape[0] and
        ellipse_y.min() >= 0 and ellipse_y.min() < ac.shape[1]):
        threshold = ac[ellipse_x.min()][ellipse_y.min()]
    else:
        threshold = ac[0][0]

    blank_image = np.zeros((ac.shape[0], ac.shape[1], 3), dtype=np.uint8) # generates a blank image
    grayscale_ellipse = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY) # generates a grayscale image
    for pt in range(ellipse_x.shape[0]):
        if (ellipse_x[pt] >= 0 and ellipse_x[pt] < ac.shape[0] and
            ellipse_y[pt] >= 0 and ellipse_y[pt] < ac.shape[1]):
            grayscale_ellipse[ellipse_x[pt], ellipse_y[pt]] = 255

    mask = grayscale_ellipse == 0
    ac[mask] = threshold

    for r in range(ac.shape[0]):
        for c in range(ac.shape[1]): # substracts either the threshold or img_array value, whichever is smaller
            ac[r][c] = ac[r][c] - min(threshold, ac[r][c]) # from every point in img_array

    if ac.sum() != 0:
        ac = ac / ac.sum() # normalizes the data

    return ac