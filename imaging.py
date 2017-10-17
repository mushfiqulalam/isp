# Note:
#   The functions try to operate in float32 data precision

# Import the libraries
import numpy as np
import math
from matplotlib import pylab as plt
import time
import utility

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
    # debug
    print(str(no_of_pixel_pad))

    for idx in range(0, len(D)): # perform same operation for each quarter

        # display progress
        print("bad pixel correction: Quarter " + str(idx+1) + " of 4")

        img = D[idx]
        size = np.shape(img)
        width  = size[1]
        height = size[0]
        # debug
        print("width: " + str(width) + " height: " + str(height))

        # pad pixels at the borders
        img = np.pad(img, \
                     (no_of_pixel_pad, no_of_pixel_pad),\
                     'reflect') # reflect would not repeat the border value

        # debug
        print(str(np.shape(img)))

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
# function: demosaic_mhc
#   demosaicing using Malvar-He-Cutler algorithm
#   http://www.ipol.im/pub/art/2011/g_mhcd/
# =============================================================
def demosaic_mhc(raw, bayer_pattern="rggb", clip_range=[0, 65535], timeshow=False):

    print("----------------------------------------------------")
    print("Running demosaicing...")

    # convert to float32 in case it was not
    raw = np.float32(raw)

    # dimensions
    size = np.shape(raw)
    width  = size[1]
    height = size[0]

    # number of pixels to pad
    no_of_pixel_pad = 2
    raw = np.pad(raw, \
                 (no_of_pixel_pad, no_of_pixel_pad),\
                 'reflect') # reflect would not repeat the border value

    # allocate space for the R, G, B planes
    R = np.empty( (height + no_of_pixel_pad * 2, width + no_of_pixel_pad * 2), dtype = np.float32 )
    G = np.empty( (height + no_of_pixel_pad * 2, width + no_of_pixel_pad * 2), dtype = np.float32 )
    B = np.empty( (height + no_of_pixel_pad * 2, width + no_of_pixel_pad * 2), dtype = np.float32 )

    # create a RGB output
    demosaic_out = np.empty( (height, width, 3), dtype = np.float32 )

    # fill up the directly available values according to the Bayer pattern
    if (bayer_pattern == "rggb"):

        G[::2, 1::2]  = raw[::2, 1::2]
        G[1::2, ::2]  = raw[1::2, ::2]
        R[::2, ::2]   = raw[::2, ::2]
        B[1::2, 1::2] = raw[1::2, 1::2]

        # Green channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # G at Red location
                if (((i % 2) == 0) and ((j % 2) == 0)):
                    G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
                    2. * G[i+1, j], \
                    -1. * R[i+2, j]])
                # G at Blue location
                elif (((i % 2) != 0) and ((j % 2) != 0)):
                    G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
                    2. * G[i+1, j],\
                    -1. * B[i+2, j]])
            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Green: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

        # Red and Blue channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # Green locations in Red rows
                if (((i % 2) == 0) and ((j % 2) != 0)):
                    # R at Green locations in Red rows
                    R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
                     -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
                     -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                      .5 * G[i+2, j]])

                    # B at Green locations in Red rows
                    B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                # Green locations in Blue rows
                elif (((i % 2) != 0) and ((j % 2) == 0)):

                    # R at Green locations in Blue rows
                    R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                    # B at Green locations in Blue rows
                    B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
                    -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
                    -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
                    -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                    .5 * G[i+2, j]])

                # R at Blue locations
                elif (((i % 2) != 0) and ((j % 2) != 0)):
                    R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
                    2. * R[i-1, j-1], 2. * R[i-1, j+1], \
                    -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
                    2. * R[i+1, j-1], 2. * R[i+1, j+1], \
                    -1.5 * B[i+2, j]])

                # B at Red locations
                elif (((i % 2) == 0) and ((j % 2) == 0)):
                    B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
                    2. * B[i-1, j-1], 2. * B[i-1, j+1], \
                    -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
                    2. * B[i+1, j-1], 2. * B[i+1, j+1], \
                    -1.5 * R[i+2, j]])

            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")


    elif (bayer_pattern == "gbrg"):

        G[::2, ::2]   = raw[::2, ::2]
        G[1::2, 1::2] = raw[1::2, 1::2]
        R[1::2, ::2]  = raw[1::2, ::2]
        B[::2, 1::2]  = raw[::2, 1::2]

        # Green channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # G at Red location
                if (((i % 2) != 0) and ((j % 2) == 0)):
                    G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
                    2. * G[i+1, j], \
                    -1. * R[i+2, j]])
                # G at Blue location
                elif (((i % 2) == 0) and ((j % 2) != 0)):
                    G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
                    2. * G[i+1, j],\
                    -1. * B[i+2, j]])
            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Green: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

        # Red and Blue channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # Green locations in Red rows
                if (((i % 2) != 0) and ((j % 2) != 0)):
                    # R at Green locations in Red rows
                    R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
                     -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
                     -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                      .5 * G[i+2, j]])

                    # B at Green locations in Red rows
                    B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                # Green locations in Blue rows
                elif (((i % 2) == 0) and ((j % 2) == 0)):

                    # R at Green locations in Blue rows
                    R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                    # B at Green locations in Blue rows
                    B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
                    -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
                    -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
                    -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                    .5 * G[i+2, j]])

                # R at Blue locations
                elif (((i % 2) == 0) and ((j % 2) != 0)):
                    R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
                    2. * R[i-1, j-1], 2. * R[i-1, j+1], \
                    -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
                    2. * R[i+1, j-1], 2. * R[i+1, j+1], \
                    -1.5 * B[i+2, j]])

                # B at Red locations
                elif (((i % 2) != 0) and ((j % 2) == 0)):
                    B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
                    2. * B[i-1, j-1], 2. * B[i-1, j+1], \
                    -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
                    2. * B[i+1, j-1], 2. * B[i+1, j+1], \
                    -1.5 * R[i+2, j]])

            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

    elif (bayer_pattern == "grbg"):

        G[::2, ::2]   = raw[::2, ::2]
        G[1::2, 1::2] = raw[1::2, 1::2]
        R[::2, 1::2]  = raw[::2, 1::2]
        B[1::2, ::2]  = raw[1::2, ::2]

        # Green channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # G at Red location
                if (((i % 2) == 0) and ((j % 2) != 0)):
                    G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
                    2. * G[i+1, j], \
                    -1. * R[i+2, j]])
                # G at Blue location
                elif (((i % 2) != 0) and ((j % 2) == 0)):
                    G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
                    2. * G[i+1, j],\
                    -1. * B[i+2, j]])
            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Green: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

        # Red and Blue channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # Green locations in Red rows
                if (((i % 2) == 0) and ((j % 2) == 0)):
                    # R at Green locations in Red rows
                    R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
                     -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
                     -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                      .5 * G[i+2, j]])

                    # B at Green locations in Red rows
                    B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                # Green locations in Blue rows
                elif (((i % 2) != 0) and ((j % 2) != 0)):

                    # R at Green locations in Blue rows
                    R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                    # B at Green locations in Blue rows
                    B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
                    -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
                    -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
                    -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                    .5 * G[i+2, j]])

                # R at Blue locations
                elif (((i % 2) != 0) and ((j % 2) == 0)):
                    R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
                    2. * R[i-1, j-1], 2. * R[i-1, j+1], \
                    -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
                    2. * R[i+1, j-1], 2. * R[i+1, j+1], \
                    -1.5 * B[i+2, j]])

                # B at Red locations
                elif (((i % 2) == 0) and ((j % 2) != 0)):
                    B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
                    2. * B[i-1, j-1], 2. * B[i-1, j+1], \
                    -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
                    2. * B[i+1, j-1], 2. * B[i+1, j+1], \
                    -1.5 * R[i+2, j]])

            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

    elif (bayer_pattern == "bggr"):

        G[::2, 1::2]  = raw[::2, 1::2]
        G[1::2, ::2]  = raw[1::2, ::2]
        R[1::2, 1::2] = raw[1::2, 1::2]
        B[::2, ::2]   = raw[::2, ::2]

        # Green channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # G at Red location
                if (((i % 2) != 0) and ((j % 2) != 0)):
                    G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
                    2. * G[i+1, j], \
                    -1. * R[i+2, j]])
                # G at Blue location
                elif (((i % 2) == 0) and ((j % 2) == 0)):
                    G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
                    2. * G[i-1, j], \
                    -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
                    2. * G[i+1, j],\
                    -1. * B[i+2, j]])
            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Green: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

        # Red and Blue channel
        for i in range(no_of_pixel_pad, height + no_of_pixel_pad):

            # to display progress
            t0 = time.process_time()

            for j in range(no_of_pixel_pad, width + no_of_pixel_pad):

                # Green locations in Red rows
                if (((i % 2) != 0) and ((j % 2) == 0)):
                    # R at Green locations in Red rows
                    R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
                     -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
                     -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                      .5 * G[i+2, j]])

                    # B at Green locations in Red rows
                    B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                # Green locations in Blue rows
                elif (((i % 2) == 0) and ((j % 2) != 0)):

                    # R at Green locations in Blue rows
                    R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
                    -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
                    .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
                    -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
                    -1. * G[i+2, j]])

                    # B at Green locations in Blue rows
                    B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
                    -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
                    -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
                    -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
                    .5 * G[i+2, j]])

                # R at Blue locations
                elif (((i % 2) == 0) and ((j % 2) == 0)):
                    R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
                    2. * R[i-1, j-1], 2. * R[i-1, j+1], \
                    -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
                    2. * R[i+1, j-1], 2. * R[i+1, j+1], \
                    -1.5 * B[i+2, j]])

                # B at Red locations
                elif (((i % 2) != 0) and ((j % 2) != 0)):
                    B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
                    2. * B[i-1, j-1], 2. * B[i-1, j+1], \
                    -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
                    2. * B[i+1, j-1], 2. * B[i+1, j+1], \
                    -1.5 * R[i+2, j]])

            if (timeshow):
                elapsed_time = time.process_time() - t0
                print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
                      " | elapsed time: " + str(elapsed_time) + " seconds")

    else:
        print("Invalid bayer pattern. Valid pattern can be rggb, gbrg, grbg, bggr")
        return demosaic_out # This will be all zeros

    # Fill up the RGB output with interpolated values
    demosaic_out[0:height, 0:width, 0] = R[no_of_pixel_pad : height + no_of_pixel_pad, \
                                           no_of_pixel_pad : width + no_of_pixel_pad]
    demosaic_out[0:height, 0:width, 1] = G[no_of_pixel_pad : height + no_of_pixel_pad, \
                                           no_of_pixel_pad : width + no_of_pixel_pad]
    demosaic_out[0:height, 0:width, 2] = B[no_of_pixel_pad : height + no_of_pixel_pad, \
                                           no_of_pixel_pad : width + no_of_pixel_pad]

    demosaic_out = np.clip(demosaic_out, clip_range[0], clip_range[1])
    return demosaic_out

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
        size = np.shape(self.data)
        width  = size[1]
        height = size[0]

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
