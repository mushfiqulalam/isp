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

    def __str__(self):
        return self.name
