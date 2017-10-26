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
# function: get_width_height
#   returns width, height
# =============================================================
def get_width_height(data):
    # We assume data be in height x width x number of channel x frames format
    if (np.ndim(data) > 1):
        size = np.shape(data)
        width = size[1]
        height = size[0]
        return width, height
    else:
        print("data dimension must be 2 or greater")


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
# function: bayer_channel_separation
#   Objective: Outputs four channels of the bayer pattern
#   Input:
#       data:   the bayer data
#       pattern:    rggb, grbg, gbrg, or bggr
#   Output:
#       R, G1, G2, B (Quarter resolution images)
# =============================================================
def bayer_channel_separation(data, pattern):
    if (pattern == "rggb"):
        R = data[::2, ::2]
        G1 = data[::2, 1::2]
        G2 = data[1::2, ::2]
        B = data[1::2, 1::2]
    elif (pattern == "grbg"):
        G1 = data[::2, ::2]
        R = data[::2, 1::2]
        B = data[1::2, ::2]
        G2 = data[1::2, 1::2]
    elif (pattern == "gbrg"):
        G1 = data[::2, ::2]
        B = data[::2, 1::2]
        R = data[1::2, ::2]
        G2 = data[1::2, 1::2]
    elif (pattern == "bggr"):
        B = data[::2, ::2]
        G1 = data[::2, 1::2]
        G2 = data[1::2, ::2]
        R = data[1::2, 1::2]
    else:
        print("pattern must be one of these: rggb, grbg, gbrg, bggr")
        return

    return R, G1, G2, B


# =============================================================
# function: bayer_channel_integration
#   Objective: combine data into a raw according to pattern
#   Input:
#       R, G1, G2, B:   the four separate channels (Quarter resolution)
#       pattern:    rggb, grbg, gbrg, or bggr
#   Output:
#       data (Full resolution image)
# =============================================================
def bayer_channel_integration(R, G1, G2, B, pattern):
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


# =============================================================
# function: shuffle_bayer_pattern
#   convert from one bayer pattern to another
# =============================================================
def shuffle_bayer_pattern(data, input_pattern, output_pattern):

    # Get separate channels
    R, G1, G2, B = bayer_channel_separation(data, input_pattern)

    # return integrated data
    return bayer_channel_integration(R, G1, G2, B, output_pattern)


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

    def __str__(self):
        return self.name
