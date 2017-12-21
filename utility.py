# =============================================================
# This file contains helper functions and classes
#
# Mushfiqul Alam, 2017
#
# Report bugs/suggestions:
#   mushfiqulalam@gmail.com
# =============================================================

import png
import numpy as np
import scipy.misc
import math
from scipy import signal        # for convolutions
from scipy import ndimage       # for n-dimensional convolution
from scipy import interpolate

# =============================================================
# function: imsave
#   save image in image formats
#   data:   is the image data
#   output_dtype: output data type
#   input_dtype: input data type
#   is_scale: is scaling needed to go from input data type to output data type
# =============================================================
def imsave(data, output_name, output_dtype="uint8", input_dtype="uint8", is_scale=False):

    dtype_dictionary = {"uint8" : np.uint8(data), "uint16" : np.uint16(data),\
                        "uint32" : np.uint32(data), "uint64" : np.uint64(data),\
                        "int8" : np.int8(data), "int16" : np.int16(data),\
                        "int32" : np.int32(data), "int64" : np.int64(data),\
                        "float16" : np.float16(data), "float32" : np.float32(data),\
                        "float64" : np.float64(data)}

    min_val_dictionary = {"uint8" : 0, "uint16" : 0,\
                          "uint32" : 0, "uint64" : 0,\
                          "int8" : -128, "int16" : -32768,\
                          "int32" : -2147483648, "int64" : -9223372036854775808}

    max_val_dictionary = {"uint8" : 255, "uint16" : 65535,\
                          "uint32" : 4294967295, "uint64" : 18446744073709551615,\
                          "int8" : 127, "int16" : 32767,\
                          "int32" : 2147483647, "int64" : 9223372036854775807}

    # scale the data in case scaling is necessary to go from input_dtype
    # to output_dtype
    if (is_scale):

        # convert data into float32
        data = np.float32(data)

        # Get minimum and maximum value of the input and output data types
        in_min  = min_val_dictionary[input_dtype]
        in_max  = max_val_dictionary[input_dtype]
        out_min = min_val_dictionary[output_dtype]
        out_max = max_val_dictionary[output_dtype]

        # clip the input data in the input_dtype range
        data = np.clip(data, in_min, in_max)

        # scale the data
        data = out_min + (data - in_min) * (out_max - out_min) / (in_max - in_min)

        # clip scaled data in output_dtype range
        data = np.clip(data, out_min, out_max)

    # convert the data into the output_dtype
    data = dtype_dictionary[output_dtype]

    # output image type: raw, png, jpeg
    output_file_type = output_name[-3:]

    # save files depending on output_file_type
    if (output_file_type == "raw"):
        pass # will be added later
        return

    elif (output_file_type == "png"):

        # png will only save uint8 or uint16
        if ((output_dtype == "uint16") or (output_dtype == "uint8")):
            if (output_dtype == "uint16"):
                output_bitdepth = 16
            elif (output_dtype == "uint8"):
                output_bitdepth = 8

            pass
        else:
            print("For png output, output_dtype must be uint8 or uint16")
            return

        with open(output_name, "wb") as f:
            # rgb image
            if (np.ndim(data) == 3):
                # create the png writer
                writer = png.Writer(width=data.shape[1], height=data.shape[0],\
                                    bitdepth = output_bitdepth)
                # convert data to the python lists expected by the png Writer
                data2list = data.reshape(-1, data.shape[1]*data.shape[2]).tolist()
                # write in the file
                writer.write(f, data2list)

            # greyscale image
            elif (np.ndim(data) == 2):
                # create the png writer
                writer = png.Writer(width=data.shape[1], height=data.shape[0],\
                                    bitdepth = output_bitdepth,\
                                    greyscale = True)
                # convert data to the python lists expected by the png Writer
                data2list = data.tolist()
                # write in the file
                writer.write(f, data2list)

    elif (output_file_type == "jpg"):
        pass # will be added later
        return

    else:
        print("output_name should contain extensions of .raw, .png, or .jpg")
        return


# =============================================================
# class: helpers
#   a class of useful helper functions
# =============================================================
class helpers:
    def __init__(self, data=None, name="helper"):
        self.data = np.float32(data)
        self.name = name

    def get_width_height(self):
        #------------------------------------------------------
        # returns width, height
        # We assume data be in height x width x number of channel x frames format
        #------------------------------------------------------
        if (np.ndim(self.data) > 1):
            size = np.shape(self.data)
            width = size[1]
            height = size[0]
            return width, height
        else:
            print("Error! data dimension must be 2 or greater")

    def bayer_channel_separation(self, pattern):
        #------------------------------------------------------
        # function: bayer_channel_separation
        #   Objective: Outputs four channels of the bayer pattern
        #   Input:
        #       data:   the bayer data
        #       pattern:    rggb, grbg, gbrg, or bggr
        #   Output:
        #       R, G1, G2, B (Quarter resolution images)
        #------------------------------------------------------
        if (pattern == "rggb"):
            R = self.data[::2, ::2]
            G1 = self.data[::2, 1::2]
            G2 = self.data[1::2, ::2]
            B = self.data[1::2, 1::2]
        elif (pattern == "grbg"):
            G1 = self.data[::2, ::2]
            R = self.data[::2, 1::2]
            B = self.data[1::2, ::2]
            G2 = self.data[1::2, 1::2]
        elif (pattern == "gbrg"):
            G1 = self.data[::2, ::2]
            B = self.data[::2, 1::2]
            R = self.data[1::2, ::2]
            G2 = self.data[1::2, 1::2]
        elif (pattern == "bggr"):
            B = self.data[::2, ::2]
            G1 = self.data[::2, 1::2]
            G2 = self.data[1::2, ::2]
            R = self.data[1::2, 1::2]
        else:
            print("pattern must be one of these: rggb, grbg, gbrg, bggr")
            return

        return R, G1, G2, B


    def bayer_channel_integration(self, R, G1, G2, B, pattern):
        #------------------------------------------------------
        # function: bayer_channel_integration
        #   Objective: combine data into a raw according to pattern
        #   Input:
        #       R, G1, G2, B:   the four separate channels (Quarter resolution)
        #       pattern:    rggb, grbg, gbrg, or bggr
        #   Output:
        #       data (Full resolution image)
        #------------------------------------------------------
        size = np.shape(R)
        data = np.empty((size[0]*2, size[1]*2), dtype=np.float32)
        if (pattern == "rggb"):
            data[::2, ::2] = R
            data[::2, 1::2] = G1
            data[1::2, ::2] = G2
            data[1::2, 1::2] = B
        elif (pattern == "grbg"):
            data[::2, ::2] = G1
            data[::2, 1::2] = R
            data[1::2, ::2] = B
            data[1::2, 1::2] = G2
        elif (pattern == "gbrg"):
            data[::2, ::2] = G1
            data[::2, 1::2] = B
            data[1::2, ::2] = R
            data[1::2, 1::2] = G2
        elif (pattern == "bggr"):
            data[::2, ::2] = B
            data[::2, 1::2] = G1
            data[1::2, ::2] = G2
            data[1::2, 1::2] = R
        else:
            print("pattern must be one of these: rggb, grbg, gbrg, bggr")
            return

        return data


    def shuffle_bayer_pattern(self, input_pattern, output_pattern):
        #------------------------------------------------------
        # function: shuffle_bayer_pattern
        #   convert from one bayer pattern to another
        #------------------------------------------------------

        # Get separate channels
        R, G1, G2, B = self.bayer_channel_separation(input_pattern)

        # return integrated data
        return self.bayer_channel_integration(R, G1, G2, B, output_pattern)


    def sigma_filter_helper(self, neighborhood_size, sigma):

        if (neighborhood_size % 2) == 0:
            print("Error! neighborhood_size must be odd for example 3, 5, 7")
            return

        # number of pixels to be padded at the borders
        no_of_pixel_pad = math.floor(neighborhood_size / 2.)

        # get width, height
        width, height = self.get_width_height()

        # pad pixels at the borders
        img = np.pad(self.data, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        # allocate memory for output
        output = np.empty((height, width), dtype=np.float32)

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad : i + no_of_pixel_pad+1,\
                                   j - no_of_pixel_pad : j + no_of_pixel_pad+1]

                lower_range = mid_pixel_val - sigma
                upper_range = mid_pixel_val + sigma

                temp = 0.
                ctr = 0
                for ni in range (0, neighborhood_size):
                    for nj in range (0, neighborhood_size):
                        if (neighborhood[ni, nj] > lower_range) and (neighborhood[ni, nj] < upper_range):
                            temp += neighborhood[ni, nj]
                            ctr += 1

                output[i - no_of_pixel_pad, j - no_of_pixel_pad] = temp / ctr

        return output

    def bilinear_interpolation(self, x, y):

        width, height = self.get_width_height()

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, width-1)
        x1 = np.clip(x1, 0, width-1)
        y0 = np.clip(y0, 0, height-1)
        y1 = np.clip(y1, 0, height-1)

        Ia = self.data[y0, x0]
        Ib = self.data[y1, x0]
        Ic = self.data[y0, x1]
        Id = self.data[y1, x1]


        x = np.clip(x, 0, width-1)
        y = np.clip(y, 0, height-1)

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id

    def degamma_srgb(self, clip_range=[0, 65535]):

        # bring data in range 0 to 1
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data = np.divide(data, clip_range[1])

        data = np.asarray(data)
        mask = data > 0.04045

        # basically, if data[x, y, c] > 0.04045, data[x, y, c] = ( (data[x, y, c] + 0.055) / 1.055 ) ^ 2.4
        #            else, data[x, y, c] = data[x, y, c] / 12.92
        data[mask] += 0.055
        data[mask] /= 1.055
        data[mask] **= 2.4

        data[np.invert(mask)] /= 12.92

        # rescale
        return np.clip(data * clip_range[1], clip_range[0], clip_range[1])

    def gamma_srgb(self, clip_range=[0, 65535]):

        # bring data in range 0 to 1
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data = np.divide(data, clip_range[1])

        data = np.asarray(data)
        mask = data > 0.0031308

        # basically, if data[x, y, c] > 0.0031308, data[x, y, c] = 1.055 * ( var_R(i, j) ^ ( 1 / 2.4 ) ) - 0.055
        #            else, data[x, y, c] = data[x, y, c] * 12.92
        data[mask] **= 0.4167
        data[mask] *= 1.055
        data[mask] -= 0.055

        data[np.invert(mask)] *= 12.92

        # rescale
        return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


    def degamma_adobe_rgb_1998(self, clip_range=[0, 65535]):

        # bring data in range 0 to 1
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data = np.divide(data, clip_range[1])

        data = np.power(data, 2.2) # originally raised to 2.19921875

        # rescale
        return np.clip(data * clip_range[1], clip_range[0], clip_range[1])

    def gamma_adobe_rgb_1998(self, clip_range=[0, 65535]):

        # bring data in range 0 to 1
        data = np.clip(self.data, clip_range[0], clip_range[1])
        data = np.divide(data, clip_range[1])

        data = np.power(data, 0.4545)

        # rescale
        return np.clip(data * clip_range[1], clip_range[0], clip_range[1])


    def get_xyz_reference(self, cie_version="1931", illuminant="d65"):

        if (cie_version == "1931"):

            xyz_reference_dictionary = {"A" : [109.850, 100.0, 35.585],\
                                        "B" : [99.0927, 100.0, 85.313],\
                                        "C" : [98.074,  100.0, 118.232],\
                                        "d50" : [96.422, 100.0, 82.521],\
                                        "d55" : [95.682, 100.0, 92.149],\
                                        "d65" : [95.047, 100.0, 108.883],\
                                        "d75" : [94.972, 100.0, 122.638],\
                                        "E" : [100.0, 100.0, 100.0],\
                                        "F1" : [92.834, 100.0, 103.665],\
                                        "F2" : [99.187, 100.0, 67.395],\
                                        "F3" : [103.754, 100.0, 49.861],\
                                        "F4" : [109.147, 100.0, 38.813],\
                                        "F5" : [90.872, 100.0, 98.723],\
                                        "F6" : [97.309, 100.0, 60.191],\
                                        "F7" : [95.044, 100.0, 108.755],\
                                        "F8" : [96.413, 100.0, 82.333],\
                                        "F9" : [100.365, 100.0, 67.868],\
                                        "F10" : [96.174, 100.0, 81.712],\
                                        "F11" : [100.966, 100.0, 64.370],\
                                        "F12" : [108.046, 100.0, 39.228]}

        elif (cie_version == "1964"):

            xyz_reference_dictionary = {"A" : [111.144, 100.0, 35.200],\
                                        "B" : [99.178, 100.0, 84.3493],\
                                        "C" : [97.285, 100.0, 116.145],\
                                        "D50" : [96.720, 100.0, 81.427],\
                                        "D55" : [95.799, 100.0, 90.926],\
                                        "D65" : [94.811, 100.0, 107.304],\
                                        "D75" : [94.416, 100.0, 120.641],\
                                        "E" : [100.0, 100.0, 100.0],\
                                        "F1" : [94.791, 100.0, 103.191],\
                                        "F2" : [103.280, 100.0, 69.026],\
                                        "F3" : [108.968, 100.0, 51.965],\
                                        "F4" : [114.961, 100.0, 40.963],\
                                        "F5" : [93.369, 100.0, 98.636],\
                                        "F6" : [102.148, 100.0, 62.074],\
                                        "F7" : [95.792, 100.0, 107.687],\
                                        "F8" : [97.115, 100.0, 81.135],\
                                        "F9" : [102.116, 100.0, 67.826],\
                                        "F10" : [99.001, 100.0, 83.134],\
                                        "F11" : [103.866, 100.0, 65.627],\
                                        "F12" : [111.428, 100.0, 40.353]}

        else:
            print("Warning! cie_version must be 1931 or 1964.")
            return

        return np.divide(xyz_reference_dictionary[illuminant], 100.0)

    def sobel_prewitt_direction_label(self, gradient_magnitude, theta, threshold=0):

        direction_label = np.zeros(np.shape(gradient_magnitude), dtype=np.float32)

        theta = np.asarray(theta)
        # vertical
        mask = ((theta >= -22.5) & (theta <= 22.5))
        direction_label[mask] = 3.

        # +45 degree
        mask = ((theta > 22.5) & (theta <= 67.5))
        direction_label[mask] = 2.

        # -45 degree
        mask = ((theta < -22.5) & (theta >= -67.5))
        direction_label[mask] = 4.

        # horizontal
        mask = ((theta > 67.5) & (theta <= 90.)) | ((theta < -67.5) & (theta >= -90.))
        direction_label[mask] = 1.

        gradient_magnitude = np.asarray(gradient_magnitude)
        mask = gradient_magnitude < threshold
        direction_label[mask] = 0.

        return direction_label

    def edge_wise_median(self, kernel_size, edge_location):

        # pad two pixels at the border
        no_of_pixel_pad = math.floor(kernel_size / 2)   # number of pixels to pad

        data = self.data
        data = np.pad(data, \
                      (no_of_pixel_pad, no_of_pixel_pad),\
                      'reflect') # reflect would not repeat the border value

        edge_location = np.pad(edge_location,\
                              (no_of_pixel_pad, no_of_pixel_pad),\
                              'reflect') # reflect would not repeat the border value

        width, height = self.get_width_height()
        output = np.empty((height, width), dtype=np.float32)

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
                if (edge_location[i, j] == 1):
                    output[i - no_of_pixel_pad, j - no_of_pixel_pad] = \
                                 np.median(data[i - no_of_pixel_pad : i + no_of_pixel_pad + 1,\
                                                j - no_of_pixel_pad : j + no_of_pixel_pad + 1])
                elif (edge_location[i, j] == 0):
                    output[i - no_of_pixel_pad, j - no_of_pixel_pad] = data[i, j]

        return output


    def nonuniform_quantization(self):

        output = np.zeros(np.shape(self.data), dtype=np.float32)
        min_val = np.min(self.data)
        max_val = np.max(self.data)

        mask = (self.data > (7./8.) * (max_val - min_val))
        output[mask] = 3.

        mask = (self.data > (3./4.) * (max_val - min_val)) & (self.data <= (7./8.) * (max_val - min_val))
        output[mask] = 2.

        mask = (self.data > (1./2.) * (max_val - min_val)) & (self.data <= (3./4.) * (max_val - min_val))
        output[mask] = 1.

        return output


    def __str__(self):
        return self.name


# =============================================================
# function: distance_euclid
#   returns Euclidean distance between two points
# =============================================================
def distance_euclid(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)


# =============================================================
# class: special_functions
#   pass input through special functions
# =============================================================
class special_function:
    def __init__(self, data, name="special function"):
        self.data = np.float32(data)
        self.name = name

    def soft_coring(self, slope, tau_threshold, gamma_speed):
        # Usage: Used in the unsharp masking sharpening Process
        # Input:
        #   slope:                  controls the boost.
        #                           the amount of sharpening, higher slope
        #                           means more aggresssive sharpening
        #
        #   tau_threshold:          controls the amount of coring.
        #                           threshold value till which the image is
        #                           not sharpened. The lower the value of
        #                           tau_threshold the more frequencies
        #                           goes through the sharpening process
        #
        #   gamma_speed:            controls the speed of convergence to the slope
        #                           smaller value gives a little bit more
        #                           sharpened image, this may be a fine tuner
        return slope * self.data * ( 1. - np.exp(-((np.abs(self.data / tau_threshold))**gamma_speed)))


    def distortion_function(self, correction_type="barrel-1", strength=0.1):

        if (correction_type == "pincushion-1"):
            return np.divide(self.data, 1. + strength * self.data)
        elif (correction_type == "pincushion-2"):
            return np.divide(self.data, 1. + strength * np.power(self.data, 2))
        elif (correction_type == "barrel-1"):
            return np.multiply(self.data, 1. + strength * self.data)
        elif (correction_type == "barrel-2"):
            return np.multiply(self.data, 1. + strength * np.power(self.data, 2))
        else:
            print("Warning! Unknown correction_type.")
            return

    def bilateral_filter(self, edge):
        # bilateral filter based upon the work of
        # Jiawen Chen, Sylvain Paris, and Fredo Durand, 2007 work

        # note: if edge data is not provided, image is served as edge
        # this is called normal bilateral filter
        # if edge data is provided, then it is called cross or joint
        # bilateral filter

        # get width and height of the image
        width, height = helpers(self.data).get_width_height()

        # sigma_spatial
        sigma_spatial = min(height, width) / 16.

        # calculate edge_delta
        edge_min = np.min(edge)
        edge_max = np.max(edge)
        edge_delta = edge_max - edge_min

        # sigma_range and sampling_range
        sigma_range = 0.1 * edge_delta
        sampling_range = sigma_range
        sampling_spatial = sigma_spatial

        # derived_sigma_spatial and derived_sigma_range
        derived_sigma_spatial = sigma_spatial / sampling_spatial
        derived_sigma_range = sigma_range / sampling_range

        # paddings
        padding_xy = np.floor(2. * derived_sigma_spatial) + 1.
        padding_z = np.floor(2. * derived_sigma_range) + 1.

        # downsamples
        downsample_width = np.uint16(np.floor((width - 1.) / sampling_spatial) + 1. + 2. * padding_xy)
        downsample_height = np.uint16(np.floor((height - 1.) / sampling_spatial) + 1. + 2. * padding_xy)
        downsample_depth = np.uint16(np.floor(edge_delta / sampling_range) + 1. + 2. * padding_z)

        grid_data = np.zeros((downsample_height, downsample_width, downsample_depth))
        grid_weight = np.zeros((downsample_height, downsample_width, downsample_depth))

        jj, ii = np.meshgrid(np.arange(0, width, 1),\
                             np.arange(0, height, 1))

        di = np.uint16(np.round( ii / sampling_spatial ) + padding_xy + 1.)
        dj = np.uint16(np.round( jj / sampling_spatial ) + padding_xy + 1.)
        dz = np.uint16(np.round( (edge - edge_min) / sampling_range ) + padding_z + 1.)


        for i in range(0, height):
            for j in range(0, width):

                data_z = self.data[i, j]
                if not np.isnan(data_z):
                    dik = di[i, j]
                    djk = dj[i, j]
                    dzk = dz[i, j]

                    grid_data[dik, djk, dzk] = grid_data[dik, djk, dzk] + data_z
                    grid_weight[dik, djk, dzk] = grid_weight[dik, djk, dzk] + 1.


        kernel_width = 2. * derived_sigma_spatial + 1.
        kernel_height = kernel_width
        kernel_depth = 2. * derived_sigma_range + 1.

        half_kernel_width = np.floor(kernel_width / 2.)
        half_kernel_height = np.floor(kernel_height / 2.)
        half_kernel_depth = np.floor(kernel_depth / 2.)

        grid_x, grid_y, grid_z = np.meshgrid(np.arange(0, kernel_width, 1),\
                                             np.arange(0, kernel_height, 1),\
                                             np.arange(0, kernel_depth, 1))

        grid_x = grid_x - half_kernel_width
        grid_y = grid_y - half_kernel_height
        grid_z = grid_z - half_kernel_depth

        grid_r_squared = ( ( np.multiply(grid_x, grid_x) + \
                             np.multiply(grid_y, grid_y) ) / np.multiply(derived_sigma_spatial, derived_sigma_spatial) ) + \
                         ( np.multiply(grid_z, grid_z) / np.multiply(derived_sigma_range, derived_sigma_range) )

        kernel = np.exp(-0.5 * grid_r_squared)
        blurred_grid_data = ndimage.convolve(grid_data, kernel, mode='reflect')
        blurred_grid_weight = ndimage.convolve(grid_weight, kernel, mode='reflect')

        # divide
        blurred_grid_weight = np.asarray(blurred_grid_weight)
        mask = blurred_grid_weight == 0
        blurred_grid_weight[mask] = -2.
        normalized_blurred_grid = np.divide(blurred_grid_data, blurred_grid_weight)
        mask = blurred_grid_weight < -1
        normalized_blurred_grid[mask] = 0.
        blurred_grid_weight[mask] = 0.

        # upsample
        jj, ii = np.meshgrid(np.arange(0, width, 1),\
                             np.arange(0, height, 1))

        di = (ii / sampling_spatial) + padding_xy + 1.
        dj = (jj / sampling_spatial) + padding_xy + 1.
        dz = (edge - edge_min) / sampling_range + padding_z + 1.

        # arrange the input points
        n_i, n_j, n_z = np.shape(normalized_blurred_grid)
        points = (np.arange(0, n_i, 1), np.arange(0, n_j, 1), np.arange(0, n_z, 1))

        # query points
        xi = (di, dj, dz)

        # multidimensional interpolation
        output = interpolate.interpn(points, normalized_blurred_grid, xi, method='linear')

        return output



# =============================================================
# class: synthetic_image_generate
#   creates sysnthetic images for different purposes
# =============================================================
class synthetic_image_generate:
    def __init__(self, width, height, name="synthetic_image"):
        self.name = name
        self.width = width
        self.height = height

    def create_lens_shading_correction_images(self, dark_current=0, flat_max=65535, flat_min=0, clip_range=[0, 65535]):
        # Objective: creates two images:
        #               dark_current_image and flat_field_image
        dark_current_image = dark_current * np.ones((self.height, self.width), dtype=np.float32)
        flat_field_image = np.empty((self.height, self.width), dtype=np.float32)

        center_pixel_pos = [self.height/2, self.width/2]
        max_distance = distance_euclid(center_pixel_pos, [self.height, self.width])

        for i in range(0, self.height):
            for j in range(0, self.width):
                flat_field_image[i, j] = (max_distance - distance_euclid(center_pixel_pos, [i, j])) / max_distance
                flat_field_image[i, j] = flat_min + flat_field_image[i, j] * (flat_max - flat_min)

        dark_current_image = np.clip(dark_current_image, clip_range[0], clip_range[1])
        flat_field_image = np.clip(flat_field_image, clip_range[0], clip_range[1])

        return dark_current_image, flat_field_image

    def create_zone_plate_image(self):
        pass

    def create_color_gradient_image(self):
        pass

    def create_random_noise_image(self, mean=0, standard_deviation=1, seed=0):
        # Creates normally distributed noisy image
        np.random.seed(seed)
        return np.random.normal(mean, standard_deviation, (self.height, self.width))

    def create_noisy_image(self, data, mean=0, standard_deviation=1, seed=0, clip_range=[0, 65535]):
        # Adds normally distributed noise to the data
        return np.clip(data + self.create_random_noise_image(mean, standard_deviation, seed), clip_range[0], clip_range[1])


# =============================================================
# class: create_filter
#   creates different filters, generally 2D filters
# =============================================================
class create_filter:
    def __init__(self, name="filter"):
        self.name = name

    def gaussian(self, kernel_size, sigma):

        # calculate which number to where the grid should be
        # remember that, kernel_size[0] is the width of the kernel
        # and kernel_size[1] is the height of the kernel
        temp = np.floor(np.float32(kernel_size) / 2.)

        # create the grid
        # example: if kernel_size = [5, 3], then:
        # x: array([[-2., -1.,  0.,  1.,  2.],
        #           [-2., -1.,  0.,  1.,  2.],
        #           [-2., -1.,  0.,  1.,  2.]])
        # y: array([[-1., -1., -1., -1., -1.],
        #           [ 0.,  0.,  0.,  0.,  0.],
        #           [ 1.,  1.,  1.,  1.,  1.]])
        x, y = np.meshgrid(np.linspace(-temp[0], temp[0], kernel_size[0]),\
                           np.linspace(-temp[1], temp[1], kernel_size[1]))

        # Gaussian equation
        temp = np.exp( -(x**2 + y**2) / (2. * sigma**2) )

        # make kernel sum equal to 1
        return temp / np.sum(temp)

    def gaussian_separable(self, kernel_size, sigma):

        # calculate which number to where the grid should be
        # remember that, kernel_size[0] is the width of the kernel
        # and kernel_size[1] is the height of the kernel
        temp = np.floor(np.float32(kernel_size) / 2.)

        # create the horizontal kernel
        x = np.linspace(-temp[0], temp[0], kernel_size[0])
        x = x.reshape((1, kernel_size[0])) # reshape to create row vector
        hx = np.exp(-x**2 / (2 * sigma**2))
        hx = hx / np.sum(hx)

        # create the vertical kernel
        y = np.linspace(-temp[1], temp[1], kernel_size[1])
        y = y.reshape((kernel_size[1], 1)) # reshape to create column vector
        hy = np.exp(-y**2 / (2 * sigma**2))
        hy = hy / np.sum(hy)

        return hx, hy

    def sobel(self, kernel_size):
        # Returns the Sobel filter kernels Sx and Sy

        Sx = .25 * np.dot([[1.], [2.], [1.]], [[1., 0., -1.]])

        if (kernel_size > 3):

            n = np.int(np.floor((kernel_size - 5) / 2 + 1))

            for i in range(0, n):

                Sx = (1./16.) * signal.convolve2d(np.dot([[1.], [2.], [1.]], [[1., 2., 1.]]), Sx)

        Sy = np.transpose(Sx)

        return Sx, Sy

    def __str__(self):
        return self.name


# =============================================================
# class: color_conversion
#   color conversion from one color space to another
# =============================================================
class color_conversion:
    def __init__(self, data, name="color conversion"):
        self.data = np.float32(data)
        self.name = name

    def rgb2gray(self):
        return 0.299 * self.data[:, :, 0] +\
               0.587 * self.data[:, :, 1] +\
               0.114 * self.data[:, :, 2]

    def rgb2ycc(self, rule="bt601"):

        # map to select kr and kb
        kr_kb_dict = {"bt601" : [0.299, 0.114],\
                      "bt709" : [0.2126, 0.0722],\
                      "bt2020" : [0.2627, 0.0593]}

        kr = kr_kb_dict[rule][0]
        kb = kr_kb_dict[rule][1]
        kg = 1 - (kr + kb)

        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = kr * self.data[:, :, 0] + \
                          kg * self.data[:, :, 1] + \
                          kb * self.data[:, :, 2]
        output[:, :, 1] = 0.5 * ((self.data[:, :, 2] - output[:, :, 0]) / (1 - kb))
        output[:, :, 2] = 0.5 * ((self.data[:, :, 0] - output[:, :, 0]) / (1 - kr))

        return output

    def ycc2rgb(self, rule="bt601"):

        # map to select kr and kb
        kr_kb_dict = {"bt601" : [0.299, 0.114],\
                      "bt709" : [0.2126, 0.0722],\
                      "bt2020" : [0.2627, 0.0593]}

        kr = kr_kb_dict[rule][0]
        kb = kr_kb_dict[rule][1]
        kg = 1 - (kr + kb)

        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = 2. * self.data[:, :, 2] * (1 - kr) + self.data[:, :, 0]
        output[:, :, 2] = 2. * self.data[:, :, 1] * (1 - kb) + self.data[:, :, 0]
        output[:, :, 1] = (self.data[:, :, 0] - kr * output[:, :, 0] - kb * output[:, :, 2]) / kg

        return output

    def rgb2xyz(self, color_space="srgb", clip_range=[0, 65535]):
        # input rgb in range clip_range
        # output xyz is in range 0 to 1

        if (color_space == "srgb"):

            # degamma / linearization
            data = helpers(self.data).degamma_srgb(clip_range)
            data = np.float32(data)
            data = np.divide(data, clip_range[1])

            # matrix multiplication`
            output = np.empty(np.shape(self.data), dtype=np.float32)
            output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
            output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
            output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

        elif (color_space == "adobe-rgb-1998"):

            # degamma / linearization
            data = helpers(self.data).degamma_adobe_rgb_1998(clip_range)
            data = np.float32(data)
            data = np.divide(data, clip_range[1])

            # matrix multiplication
            output = np.empty(np.shape(self.data), dtype=np.float32)
            output[:, :, 0] = data[:, :, 0] * 0.5767309 + data[:, :, 1] * 0.1855540 + data[:, :, 2] * 0.1881852
            output[:, :, 1] = data[:, :, 0] * 0.2973769 + data[:, :, 1] * 0.6273491 + data[:, :, 2] * 0.0752741
            output[:, :, 2] = data[:, :, 0] * 0.0270343 + data[:, :, 1] * 0.0706872 + data[:, :, 2] * 0.9911085

        elif (color_space == "linear"):

            # matrix multiplication`
            output = np.empty(np.shape(self.data), dtype=np.float32)
            data = np.float32(self.data)
            data = np.divide(data, clip_range[1])
            output[:, :, 0] = data[:, :, 0] * 0.4124 + data[:, :, 1] * 0.3576 + data[:, :, 2] * 0.1805
            output[:, :, 1] = data[:, :, 0] * 0.2126 + data[:, :, 1] * 0.7152 + data[:, :, 2] * 0.0722
            output[:, :, 2] = data[:, :, 0] * 0.0193 + data[:, :, 1] * 0.1192 + data[:, :, 2] * 0.9505

        else:
            print("Warning! color_space must be srgb or adobe-rgb-1998.")
            return

        return output


    def xyz2rgb(self, color_space="srgb", clip_range=[0, 65535]):
        # input xyz is in range 0 to 1
        # output rgb in clip_range

        # allocate space for output
        output = np.empty(np.shape(self.data), dtype=np.float32)

        if (color_space == "srgb"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  3.2406 + self.data[:, :, 1] * -1.5372 + self.data[:, :, 2] * -0.4986
            output[:, :, 1] = self.data[:, :, 0] * -0.9689 + self.data[:, :, 1] *  1.8758 + self.data[:, :, 2] *  0.0415
            output[:, :, 2] = self.data[:, :, 0] *  0.0557 + self.data[:, :, 1] * -0.2040 + self.data[:, :, 2] *  1.0570

            # gamma to retain nonlinearity
            output = helpers(output * clip_range[1]).gamma_srgb(clip_range)


        elif (color_space == "adobe-rgb-1998"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  2.0413690 + self.data[:, :, 1] * -0.5649464 + self.data[:, :, 2] * -0.3446944
            output[:, :, 1] = self.data[:, :, 0] * -0.9692660 + self.data[:, :, 1] *  1.8760108 + self.data[:, :, 2] *  0.0415560
            output[:, :, 2] = self.data[:, :, 0] *  0.0134474 + self.data[:, :, 1] * -0.1183897 + self.data[:, :, 2] *  1.0154096

            # gamma to retain nonlinearity
            output = helpers(output * clip_range[1]).gamma_adobe_rgb_1998(clip_range)


        elif (color_space == "linear"):

            # matrix multiplication
            output[:, :, 0] = self.data[:, :, 0] *  3.2406 + self.data[:, :, 1] * -1.5372 + self.data[:, :, 2] * -0.4986
            output[:, :, 1] = self.data[:, :, 0] * -0.9689 + self.data[:, :, 1] *  1.8758 + self.data[:, :, 2] *  0.0415
            output[:, :, 2] = self.data[:, :, 0] *  0.0557 + self.data[:, :, 1] * -0.2040 + self.data[:, :, 2] *  1.0570

            # gamma to retain nonlinearity
            output = output * clip_range[1]

        else:
            print("Warning! color_space must be srgb or adobe-rgb-1998.")
            return

        return output


    def xyz2lab(self, cie_version="1931", illuminant="d65"):

        xyz_reference = helpers().get_xyz_reference(cie_version, illuminant)

        data = self.data
        data[:, :, 0] = data[:, :, 0] / xyz_reference[0]
        data[:, :, 1] = data[:, :, 1] / xyz_reference[1]
        data[:, :, 2] = data[:, :, 2] / xyz_reference[2]

        data = np.asarray(data)

        # if data[x, y, c] > 0.008856, data[x, y, c] = data[x, y, c] ^ (1/3)
        # else, data[x, y, c] = 7.787 * data[x, y, c] + 16/116
        mask = data > 0.008856
        data[mask] **= 1./3.
        data[np.invert(mask)] *= 7.787
        data[np.invert(mask)] += 16./116.

        data = np.float32(data)
        output = np.empty(np.shape(self.data), dtype=np.float32)
        output[:, :, 0] = 116. * data[:, :, 1] - 16.
        output[:, :, 1] = 500. * (data[:, :, 0] - data[:, :, 1])
        output[:, :, 2] = 200. * (data[:, :, 1] - data[:, :, 2])

        return output


    def lab2xyz(self, cie_version="1931", illuminant="d65"):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 1] = (self.data[:, :, 0] + 16.) / 116.
        output[:, :, 0] = (self.data[:, :, 1] / 500.) + output[:, :, 1]
        output[:, :, 2] = output[:, :, 1] - (self.data[:, :, 2] / 200.)

        # if output[x, y, c] > 0.008856, output[x, y, c] ^ 3
        # else, output[x, y, c] = ( output[x, y, c] - 16/116 ) / 7.787
        output = np.asarray(output)
        mask = output > 0.008856
        output[mask] **= 3.
        output[np.invert(mask)] -= 16/116
        output[np.invert(mask)] /= 7.787

        xyz_reference = helpers().get_xyz_reference(cie_version, illuminant)

        output = np.float32(output)
        output[:, :, 0] = output[:, :, 0] * xyz_reference[0]
        output[:, :, 1] = output[:, :, 1] * xyz_reference[1]
        output[:, :, 2] = output[:, :, 2] * xyz_reference[2]

        return output

    def lab2lch(self):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 0] = self.data[:, :, 0] # L transfers directly
        output[:, :, 1] = np.power(np.power(self.data[:, :, 1], 2) + np.power(self.data[:, :, 2], 2), 0.5)
        output[:, :, 2] = np.arctan2(self.data[:, :, 2], self.data[:, :, 1]) * 180 / np.pi

        return output

    def lch2lab(self):

        output = np.empty(np.shape(self.data), dtype=np.float32)

        output[:, :, 0] = self.data[:, :, 0] # L transfers directly
        output[:, :, 1] = np.multiply(np.cos(self.data[:, :, 2] * np.pi / 180), self.data[:, :, 1])
        output[:, :, 2] = np.multiply(np.sin(self.data[:, :, 2] * np.pi / 180), self.data[:, :, 1])

        return output

    def __str__(self):
        return self.name


# =============================================================
# class: edge_detection
#   detect edges in an image
# =============================================================
class edge_detection:
    def __init__(self, data, name="edge detection"):
        self.data = np.float32(data)
        self.name = name

    def sobel(self, kernel_size=3, output_type="all", threshold=0., clip_range=[0, 65535]):

        Sx, Sy = create_filter().sobel(kernel_size)

        # Gradient in x direction: Gx
        # Gradient in y direction: Gy
        if np.ndim(self.data) > 2:

            Gx = np.empty(np.shape(self.data), dtype=np.float32)
            Gy = np.empty(np.shape(self.data), dtype=np.float32)

            for dimension_idx in range(0, np.shape(self.data)[2]):
                Gx[:, :, dimension_idx] = signal.convolve2d(self.data[:, :, dimension_idx], Sx, mode="same", boundary="symm")
                Gy[:, :, dimension_idx] = signal.convolve2d(self.data[:, :, dimension_idx], Sy, mode="same", boundary="symm")

        elif np.ndim(self.data) == 2:
            Gx = signal.convolve2d(self.data, Sx, mode="same", boundary="symm")
            Gy = signal.convolve2d(self.data, Sy, mode="same", boundary="symm")

        else:
            print("Warning! Data dimension must be 2 or 3.")

        # Gradient magnitude
        G = np.power(np.power(Gx, 2) + np.power(Gy, 2), .5)

        if (output_type == "gradient_magnitude"):
            return G

        # Gradient angle
        theta = np.arctan(np.divide(Gy, Gx)) * 180. / np.pi

        if (output_type == "gradient_magnitude_and_angle"):
            return G, theta

        # Change the threshold according to the clip_range's maximum value
        threshold = threshold * clip_range[1]

        # calculating if the edge is a strong edge
        is_edge = np.zeros(np.shape(self.data), dtype=np.int)
        mask = G > threshold
        is_edge[mask] = 1

        if (output_type == "is_edge"):
            return is_edge


        # Edge direction label
        temp = np.asarray(theta)
        direction_label = np.zeros(np.shape(self.data), dtype=np.float32)

        if np.ndim(self.data > 2):
            for i in range(0, np.shape(self.data)[2]):
                direction_label[:, :, i] = helpers().sobel_prewitt_direction_label(G[:, :, i], theta[:, :, i], threshold)
        else:
            direction_label = helpers().sobel_prewitt_direction_label(G, theta, threshold)

        if (output_type == "all"):
            return G, Gx, Gy, theta, is_edge, direction_label


    def __str__(self):
        return self.name
