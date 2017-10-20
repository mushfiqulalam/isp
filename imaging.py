# Note:
#   The functions try to operate in float32 data precision

# Import the libraries
import numpy as np
import math
from matplotlib import pylab as plt
import time
import utility
import debayer

# =============================================================
# class: ImageInfo
#   Helps set up necessary information/metadata of the image
# =============================================================
class ImageInfo:
    def __init__(self, name = "unknown", data = -1, is_show = False):
        self.name   = name
        self.data   = data
        self.size   = np.shape(self.data)
        self.is_show = is_show
        self.color_space = "unknown"
        self.bayer_pattern = "unknown"
        self.channel_gain = (1.0, 1.0, 1.0, 1.0)
        self.bit_depth = 0
        self.black_level = (0, 0, 0, 0)
        self.min_value = np.min(self.data)
        self.max_value = np.max(self.data)
        self.data_type = self.data.dtype

        # Display image only isShow = True
        if (self.is_show):
            plt.imshow(self.data)
            plt.show()

    def set_data(self, data):
        # This function updates data and corresponding fields
        self.data = data
        self.size = np.shape(self.data)
        self.data_type = self.data.dtype
        self.min_value = np.min(self.data)
        self.max_value = np.max(self.data)

    def get_size(self):
        return self.size

    def get_width(self):
        return self.size[1]

    def get_height(self):
        return self.size[0]

    def get_depth(self):
        if np.ndim(self.data) > 2:
            return self.size[2]
        else:
            return 0

    def set_color_space(self, color_space):
        self.color_space = color_space

    def get_color_space(self):
        return self.color_space

    def set_channel_gain(self, channel_gain):
        self.channel_gain = channel_gain

    def get_channel_gain(self):
        return self.channel_gain

    def set_bayer_pattern(self, bayer_pattern):
        self.bayer_pattern = bayer_pattern

    def get_bayer_pattern(self):
        return self.bayer_pattern

    def set_bit_depth(self, bit_depth):
        self.bit_depth = bit_depth

    def get_bit_depth(self):
        return self.bit_depth

    def set_black_level(self, black_level):
        self.black_level = black_level

    def get_black_level(self):
        return self.black_level

    def get_min_value(self):
        return self.min_value

    def get_max_value(self):
        return self.max_value

    def get_data_type(self):
        return self.data_type

    def __str__(self):
        return "Image " + self.name + " info:" + \
                          "\n\tname:\t" + self.name + \
                          "\n\tsize:\t" + str(self.size) + \
                          "\n\tcolor space:\t" + self.color_space + \
                          "\n\tbayer pattern:\t" + self.bayer_pattern + \
                          "\n\tchannel gains:\t" + str(self.channel_gain) + \
                          "\n\tbit depth:\t" + str(self.bit_depth) + \
                          "\n\tdata type:\t" + str(self.data_type) + \
                          "\n\tblack level:\t" + str(self.black_level) + \
                          "\n\tminimum value:\t" + str(self.min_value) + \
                          "\n\tmaximum value:\t" + str(self.max_value)


# =============================================================
# function: black_level_correction
#   subtracts the black level channel wise
# =============================================================
def black_level_correction(raw, black_level):

    print("----------------------------------------------------")
    print("Running black level correction...")

    # create new data so that original raw data do not change
    data = np.zeros(raw.shape)

    # subtract the black levels
    data[::2, ::2]   = np.float32(raw[::2, ::2]) - np.float32(black_level[0])
    data[::2, 1::2]  = np.float32(raw[::2, 1::2]) - np.float32(black_level[1])
    data[1::2, ::2]  = np.float32(raw[1::2, ::2]) - np.float32(black_level[2])
    data[1::2, 1::2] = np.float32(raw[1::2, 1::2]) - np.float32(black_level[3])

    # clip within the range
    data = np.clip(data, 0., None) # upper level not necessary
    data = np.float32(data)

    return data

# =============================================================
# function: channel_gain_white_balance
#   multiply with the white balance channel gains
# =============================================================
def channel_gain_white_balance(data, channel_gain):

    print("----------------------------------------------------")
    print("Running channel gain white balance...")

    # convert into float32 in case they were not
    data = np.float32(data)
    channel_gain = np.float32(channel_gain)

    # multiply with the channel gains
    data[::2, ::2]   = data[::2, ::2] * channel_gain[0]
    data[::2, 1::2]  = data[::2, 1::2] * channel_gain[1]
    data[1::2, ::2]  = data[1::2, ::2] * channel_gain[2]
    data[1::2, 1::2] = data[1::2, 1::2] * channel_gain[3]

    # clipping within range
    data = np.clip(data, 0., None) # upper level not necessary

    return data

# =============================================================
# function: bad_pixel_correction
#   correct for the bad (dead, stuck, or hot) pixels
# =============================================================
def bad_pixel_correction(data, neighborhood_size):

    print("----------------------------------------------------")
    print("Running bad pixel correction...")

    if ((neighborhood_size % 2) == 0):
        print("neighborhood_size shoud be odd number, recommended value 3")
        return data

    # convert to float32 in case they were not
    # Being consistent in data format to be float32
    data = np.float32(data)

    # Separate out the quarter resolution images
    D = {} # Empty dictionary
    D[0] = data[::2, ::2]
    D[1] = data[::2, 1::2]
    D[2] = data[1::2, ::2]
    D[3] = data[1::2, 1::2]

    # number of pixels to be padded at the borders
    no_of_pixel_pad = math.floor(neighborhood_size / 2.)

    for idx in range(0, len(D)): # perform same operation for each quarter

        # display progress
        print("bad pixel correction: Quarter " + str(idx+1) + " of 4")

        img = D[idx]
        width, height = utility.get_width_height(img)

        # pad pixels at the borders
        img = np.pad(img, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # save the middle pixel value
                mid_pixel_val = img[i, j]

                # extract the neighborhood
                neighborhood = img[i - no_of_pixel_pad : i + no_of_pixel_pad+1,\
                                   j - no_of_pixel_pad : j + no_of_pixel_pad+1]

                # set the center pixels value same as the left pixel
                # Does not matter replace with right or left pixel
                # is used to replace the center pixels value
                neighborhood[no_of_pixel_pad, no_of_pixel_pad] = neighborhood[no_of_pixel_pad, no_of_pixel_pad-1]

                min_neighborhood = np.min(neighborhood)
                max_neighborhood = np.max(neighborhood)

                if (mid_pixel_val < min_neighborhood):
                    img[i,j] = min_neighborhood
                elif (mid_pixel_val > max_neighborhood):
                    img[i,j] = max_neighborhood
                else:
                    img[i,j] = mid_pixel_val

        # Put the corrected image to the dictionary
        D[idx] = img[no_of_pixel_pad : height + no_of_pixel_pad,\
                     no_of_pixel_pad : width + no_of_pixel_pad]

    # Regrouping the data
    data[::2, ::2]   = D[0]
    data[::2, 1::2]  = D[1]
    data[1::2, ::2]  = D[2]
    data[1::2, 1::2] = D[3]

    return data

# =============================================================
# class: demosaic
# =============================================================
class demosaic:
    def __init__(self, data, bayer_pattern="rggb", clip_range=[0, 65535], name="demosaic"):
        self.data = np.float32(data)
        self.bayer_pattern = bayer_pattern
        self.clip_range = clip_range
        self.name = name

    def mhc(self, timeshow=False):

        print("----------------------------------------------------")
        print("Running demosaicing...")

        return debayer.debayer_mhc(self.data, self.bayer_pattern, self.clip_range, timeshow)

    def __str__(self):
        return self.name


# =============================================================
# class: lens_shading_correction
#   Correct the lens shading / vignetting
# =============================================================
class lens_shading_correction:
    def __init__(self, data, name="lens_shading_correction"):
        # convert to float32 in case it was not
        self.data = np.float32(data)
        self.name = name

    def flat_field_compensation(self, dark_current_image, flat_field_image):
        # dark_current_image:
        #       is captured from the camera with cap on
        #       and fully dark condition, several images captured and
        #       temporally averaged
        # flat_field_image:
        #       is found by capturing an image of a flat field test chart
        #       with certain lighting condition
        # Note: flat_field_compensation is memory intensive procedure because
        #       both the dark_current_image and flat_field_image need to be
        #       saved in memory beforehand
        print("----------------------------------------------------")
        print("Running lens shading correction with flat field compensation...")

        # convert to float32 in case it was not
        dark_current_image = np.float32(dark_current_image)
        flat_field_image = np.float32(flat_field_image)
        temp = flat_field_image - dark_current_image
        return np.average(temp) * np.divide((self.data - dark_current_image), temp)

    def approximate_mathematical_compensation(self, params, clip_min=0, clip_max=65535):
        # parms:
        #       parameters of a parabolic model y = a*(x-b)^2 + c
        #       For example, params = [0.01759, -28.37, -13.36]
        # Note: approximate_mathematical_compensation require less memory
        print("----------------------------------------------------")
        print("Running lens shading correction with approximate mathematical compensation...")
        width, height = utility.get_width_height(self.data)

        center_pixel_pos = [height/2, width/2]
        max_distance = utility.distance_euclid(center_pixel_pos, [height, width])

        # allocate memory for output
        temp = np.empty((height, width), dtype=np.float32)

        for i in range(0, height):
            for j in range(0, width):
                distance = utility.distance_euclid(center_pixel_pos, [i, j]) / max_distance
                # parabolic model
                gain = params[0] * (distance - params[1])**2 + params[2]
                temp[i, j] = self.data[i, j] * gain

        temp = np.clip(temp, clip_min, clip_max)
        return temp

    def __str__(self):
        return "lens shading correction. There are two methods: " + \
                "\n (1) flat_field_compensation: requires dark_current_image and flat_field_image" + \
                "\n (2) approximate_mathematical_compensation:"

# =============================================================
# class: lens_shading_correction
#   Correct the lens shading / vignetting
# =============================================================
class bayer_denoising:
    def __init__(self, data, name="bayer_denoising"):
        # convert to float32 in case it was not
        self.data = np.float32(data)
        self.name = name

    def utilize_hvs_behavior(self, bayer_pattern, initial_noise_level, hvs_min, hvs_max, threshold_red_blue, clip_range):
        # Objective: bayer denoising
        # Inputs:
        #   bayer_pattern:  rggb, gbrg, grbg, bggr
        #   initial_noise_level:
        # Output:
        #   denoised bayer raw output
        # Source: Based on paper titled "Noise Reduction for CFA Image Sensors
        #   Exploiting HVS Behaviour," by Angelo Bosco, Sebastiano Battiato,
        #   Arcangelo Bruna and Rosetta Rizzo
        #   Sensors 2009, 9, 1692-1713; doi:10.3390/s90301692

        print("----------------------------------------------------")
        print("Running bayer denoising utilizing hvs behavior...")

        # copy the self.data to raw and we will only work on raw
        # to make sure no change happen to self.data
        raw = self.data
        raw = np.clip(raw, clip_range[0], clip_range[1])
        width, height = utility.get_width_height(raw)

        # First make the bayer_pattern rggb
        # The algorithm is written only for rggb pattern, thus convert all other
        # pattern to rggb. Furthermore, this shuffling does not affect the
        # algorithm output
        if (bayer_pattern != "rggb"):
            raw = utility.shuffle_bayer_pattern(self.data, bayer_pattern, "rggb")

        # fixed neighborhood_size
        neighborhood_size = 5 # we are keeping this fixed
                              # bigger size such as 9 can be declared
                              # however, the code need to be changed then

        # pad two pixels at the border
        no_of_pixel_pad = math.floor(neighborhood_size / 2)   # number of pixels to pad

        raw = np.pad(raw, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        # allocating space for denoised output
        denoised_out = np.empty((height, width), dtype=np.float32)

        texture_degree_debug = np.empty((height, width), dtype=np.float32)
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # center pixel
                center_pixel = raw[i, j]

                # signal analyzer block
                half_max = clip_range[1] / 2
                if (center_pixel <= half_max):
                    hvs_weight = -(((hvs_max - hvs_min) * center_pixel) / half_max) + hvs_max
                else:
                    hvs_weight = (((center_pixel - clip_range[1]) * (hvs_max - hvs_min))/(clip_range[1] - half_max)) + hvs_max

                # noise level estimator previous value
                if (j < no_of_pixel_pad+2):
                    noise_level_previous_red   = initial_noise_level
                    noise_level_previous_blue  = initial_noise_level
                    noise_level_previous_green = initial_noise_level
                else:
                    noise_level_previous_green = noise_level_current_green
                    if ((i % 2) == 0): # red
                        noise_level_previous_red = noise_level_current_red
                    elif ((i % 2) != 0): # blue
                        noise_level_previous_blue = noise_level_current_blue

                # Processings depending on Green or Red/Blue
                # Red
                if (((i % 2) == 0) and ((j % 2) == 0)):
                    # get neighborhood
                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # absolute difference from the center pixel
                    d =  np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_red

                    # texture degree analyzer
                    if (d_max <= threshold_red_blue):
                        texture_degree = 1.
                    elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                        texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0.

                    # noise level estimator update
                    noise_level_current_red = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_red

                # Blue
                elif (((i % 2) != 0) and ((j % 2) != 0)):

                    # get neighborhood
                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # absolute difference from the center pixel
                    d =  np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_blue

                    # texture degree analyzer
                    if (d_max <= threshold_red_blue):
                        texture_degree = 1.
                    elif ((d_max > threshold_red_blue) and (d_max <= texture_threshold)):
                        texture_degree = -((d_max - threshold_red_blue) / (texture_threshold - threshold_red_blue)) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0.

                    # noise level estimator update
                    noise_level_current_blue = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_blue

                # Green
                elif ((((i % 2) == 0) and ((j % 2) != 0)) or (((i % 2) != 0) and ((j % 2) == 0))):

                    neighborhood = [raw[i-2, j-2], raw[i-2, j], raw[i-2, j+2],\
                                    raw[i-1, j-1], raw[i-1, j+1],\
                                    raw[i, j-2], raw[i, j+2],\
                                    raw[i+1, j-1], raw[i+1, j+1],\
                                    raw[i+2, j-2], raw[i+2, j], raw[i+2, j+2]]

                    # difference from the center pixel
                    d = np.abs(neighborhood - center_pixel)

                    # maximum and minimum difference
                    d_max = np.max(d)
                    d_min = np.min(d)

                    # calculate texture_threshold
                    texture_threshold = hvs_weight + noise_level_previous_green

                    # texture degree analyzer
                    if (d_max == 0):
                        texture_degree = 1
                    elif ((d_max > 0) and (d_max <= texture_threshold)):
                        texture_degree = -(d_max / texture_threshold) + 1.
                    elif (d_max > texture_threshold):
                        texture_degree = 0

                    # noise level estimator update
                    noise_level_current_green = texture_degree * d_max + (1 - texture_degree) * noise_level_previous_green

                # similarity threshold calculation
                if (texture_degree == 1):
                    threshold_low = threshold_high = d_max
                elif (texture_degree == 0):
                    threshold_low = d_min
                    threshold_high = (d_max + d_min) / 2
                elif ((texture_degree > 0) and (texture_degree < 1)):
                    threshold_high = (d_max + ((d_max + d_min) / 2)) / 2
                    threshold_low = (d_min + threshold_high) / 2

                # weight computation
                weight = np.empty(np.size(d), dtype=np.float32)
                pf = 0.
                for w_i in range(0, np.size(d)):
                    if (d[w_i] <= threshold_low):
                        weight[w_i] = 1.
                    elif (d[w_i] > threshold_high):
                        weight[w_i] = 0.
                    elif ((d[w_i] > threshold_low) and (d[w_i] < threshold_high)):
                        weight[w_i] = 1. + ((d[w_i] - threshold_low) / (threshold_low - threshold_high))

                    pf += weight[w_i] * neighborhood[w_i] + (1. - weight[w_i]) * center_pixel

                denoised_out[i - no_of_pixel_pad, j-no_of_pixel_pad] = pf / np.size(d)
                # texture_degree_debug is a debug output
                texture_degree_debug[i - no_of_pixel_pad, j-no_of_pixel_pad] = texture_degree

        if (bayer_pattern != "rggb"):
            denoised_out = utility.shuffle_bayer_pattern(denoised_out, "rggb", bayer_pattern)

        return np.clip(denoised_out, clip_range[0], clip_range[1]), texture_degree_debug

    def __str__(self):
        return self.name


# =============================================================
# class: nonlinearity
#   apply gamma or degamma
# =============================================================
class nonlinearity:
    def __init__(self, data, name="nonlinearity"):
        self.data = np.float32(data)
        self.name = name

    def by_value(self, value, clip_range):

        print("----------------------------------------------------")
        print("Running nonlinearity by value...")

        # clip within the range
        data = np.clip(self.data, clip_range[0], clip_range[1])
        # make 0 to 1
        data = data / clip_range[1]
        # apply nonlinearity
        return np.clip(clip_range[1] * (data**value), clip_range[0], clip_range[1])

    def by_table(self, table, clip_range):

        print("----------------------------------------------------")
        print("Running nonlinearity by table...")

        pass

    def __str__(self):
        return self.name
