import numpy as np
import math
import time
import utility
from scipy import signal

# =============================================================
# function: dbayer_mhc
#   demosaicing using Malvar-He-Cutler algorithm
#   http://www.ipol.im/pub/art/2011/g_mhcd/
# =============================================================
def debayer_mhc(raw, bayer_pattern="rggb", clip_range=[0, 65535], timeshow=False):

    # convert to float32 in case it was not
    raw = np.float32(raw)

    # dimensions
    width, height = utility.helpers(raw).get_width_height()

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")


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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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
                      " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")

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


def fill_channel_directional_weight(data, bayer_pattern):

    #== Calculate the directional weights (weight_N, weight_E, weight_S, weight_W.
    # where N, E, S, W stand for north, east, south, and west.)
    data = np.asarray(data)
    v = np.asarray(signal.convolve2d(data, [[1],[0],[-1]], mode="same", boundary="symm"))
    h = np.asarray(signal.convolve2d(data, [[1, 0, -1]], mode="same", boundary="symm"))

    weight_N = np.zeros(np.shape(data), dtype=np.float32)
    weight_E = np.zeros(np.shape(data), dtype=np.float32)
    weight_S = np.zeros(np.shape(data), dtype=np.float32)
    weight_W = np.zeros(np.shape(data), dtype=np.float32)

    value_N = np.zeros(np.shape(data), dtype=np.float32)
    value_E = np.zeros(np.shape(data), dtype=np.float32)
    value_S = np.zeros(np.shape(data), dtype=np.float32)
    value_W = np.zeros(np.shape(data), dtype=np.float32)

    if ((bayer_pattern == "rggb") or (bayer_pattern == "bggr")):


        # note that in the following the locations in the comments are given
        # assuming the bayer_pattern rggb

        #== CALCULATE WEIGHTS IN B LOCATIONS
        weight_N[1::2, 1::2] = np.abs(v[1::2, 1::2]) + np.abs(v[::2, 1::2])

        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp_h_b = np.hstack((h, np.atleast_2d(h[:, -2]).T))
        weight_E[1::2, 1::2] = np.abs(h[1::2, 1::2]) + np.abs(temp_h_b[1::2, 2::2])

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp_v_b = np.vstack((v, v[-1]))
        weight_S[1::2, 1::2] = np.abs(v[1::2, 1::2]) + np.abs(temp_v_b[2::2, 1::2])
        weight_W[1::2, 1::2] = np.abs(h[1::2, 1::2]) + np.abs(h[1::2, ::2])

        #== CALCULATE WEIGHTS IN R LOCATIONS
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp_v_r = np.delete(np.vstack((v[1], v)), -1, 0)
        weight_N[::2, ::2] = np.abs(v[::2, ::2]) + np.abs(temp_v_r[::2, ::2])

        weight_E[::2, ::2] = np.abs(h[::2, ::2]) + np.abs(h[::2, 1::2])

        weight_S[::2, ::2] = np.abs(v[::2, ::2]) + np.abs(v[1::2, ::2])

        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp_h_r = np.delete(np.hstack((np.atleast_2d(h[:, 1]).T, h)), -1, 1)
        weight_W[::2, ::2] = np.abs(h[::2, ::2]) + np.abs(temp_h_r[::2, ::2])

        weight_N = np.divide(1., 1. + weight_N)
        weight_E = np.divide(1., 1. + weight_E)
        weight_S = np.divide(1., 1. + weight_S)
        weight_W = np.divide(1., 1. + weight_W)

        #== CALCULATE DIRECTIONAL ESTIMATES IN B LOCATIONS
        value_N[1::2, 1::2] = data[::2, 1::2] + v[::2, 1::2] / 2.

        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        value_E[1::2, 1::2] = temp[1::2, 2::2] - temp_h_b[1::2, 2::2] / 2.

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((data, data[-1]))
        value_S[1::2, 1::2] = temp[2::2, 1::2] - temp_v_b[2::2, 1::2] / 2.

        value_W[1::2, 1::2] = data[1::2, ::2] + h[1::2, ::2] / 2.

        #== CALCULATE DIRECTIONAL ESTIMATES IN R LOCATIONS
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        value_N[::2, ::2] = temp[::2, ::2] + temp_v_r[::2, ::2] / 2.

        value_E[::2, ::2] = data[::2, 1::2] - h[::2, 1::2] / 2.

        value_S[::2, ::2] = data[1::2, ::2] - v[1::2, ::2] / 2.

        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(data[:, 1]).T, data)), -1, 1)
        value_W[::2, ::2] = temp[::2, ::2] + temp_h_r[::2, ::2] / 2.

        output = np.zeros(np.shape(data), dtype=np.float32)
        output = np.divide((np.multiply(value_N, weight_N) + \
                            np.multiply(value_E, weight_E) + \
                            np.multiply(value_S, weight_S) + \
                            np.multiply(value_W, weight_W)),\
                            (weight_N + weight_E + weight_S + weight_W))

        output[::2, 1::2] = data[::2, 1::2]
        output[1::2, ::2] = data[1::2, ::2]

        return output

    elif ((bayer_pattern == "gbrg") or (bayer_pattern == "grbg")):

        # note that in the following the locations in the comments are given
        # assuming the bayer_pattern gbrg

        #== CALCULATE WEIGHTS IN B LOCATIONS
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp_v_b = np.delete(np.vstack((v[1], v)), -1, 0)
        weight_N[::2, 1::2] = np.abs(v[::2, 1::2]) + np.abs(temp_v_b[::2, 1::2])

        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp_h_b = np.hstack((h, np.atleast_2d(h[:, -2]).T))
        weight_E[::2, 1::2] = np.abs(h[::2, 1::2]) + np.abs(temp_h_b[::2, 2::2])

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        weight_S[::2, 1::2] = np.abs(v[::2, 1::2]) + np.abs(v[1::2, 1::2])
        weight_W[::2, 1::2] = np.abs(h[::2, 1::2]) + np.abs(h[::2, ::2])

        #== CALCULATE WEIGHTS IN R LOCATIONS
        weight_N[1::2, ::2] = np.abs(v[1::2, ::2]) + np.abs(v[::2, ::2])
        weight_E[1::2, ::2] = np.abs(h[1::2, ::2]) + np.abs(h[1::2, 1::2])

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp_v_r = np.vstack((v, v[-1]))
        weight_S[1::2, ::2] = np.abs(v[1::2, ::2]) + np.abs(temp_v_r[2::2, ::2])

        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp_h_r = np.delete(np.hstack((np.atleast_2d(h[:, 1]).T, h)), -1, 1)
        weight_W[1::2, ::2] = np.abs(h[1::2, ::2]) + np.abs(temp_h_r[1::2, ::2])

        weight_N = np.divide(1., 1. + weight_N)
        weight_E = np.divide(1., 1. + weight_E)
        weight_S = np.divide(1., 1. + weight_S)
        weight_W = np.divide(1., 1. + weight_W)

        #== CALCULATE DIRECTIONAL ESTIMATES IN B LOCATIONS
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        value_N[::2, 1::2] = temp[::2, 1::2] + temp_v_b[::2, 1::2] / 2.

        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        value_E[::2, 1::2] = temp[::2, 2::2] - temp_h_b[::2, 2::2] / 2.

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        value_S[::2, 1::2] = data[1::2, 1::2] - v[1::2, 1::2] / 2.

        value_W[::2, 1::2] = data[::2, ::2] + h[::2, ::2] / 2.

        #== CALCULATE DIRECTIONAL ESTIMATES IN R LOCATIONS
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        value_N[1::2, ::2] = data[::2, ::2] + v[::2, ::2] / 2.
        value_E[1::2, ::2] = data[1::2, 1::2] - h[1::2, 1::2] / 2.

        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((data, data[-1]))
        value_S[1::2, ::2] = temp[2::2, ::2] - temp_v_r[2::2, ::2] / 2.

        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(data[:, 1]).T, data)), -1, 1)
        value_W[1::2, ::2] = temp[1::2, ::2] + temp_h_r[1::2, ::2] / 2.

        output = np.zeros(np.shape(data), dtype=np.float32)
        output = np.divide((np.multiply(value_N, weight_N) + \
                            np.multiply(value_E, weight_E) + \
                            np.multiply(value_S, weight_S) + \
                            np.multiply(value_W, weight_W)),\
                            (weight_N + weight_E + weight_S + weight_W))

        output[::2, ::2] = data[::2, ::2]
        output[1::2, 1::2] = data[1::2, 1::2]

        return output


def fill_br_locations(data, G, bayer_pattern):

    # Fill up the B/R values interpolated at R/B locations
    B = np.zeros(np.shape(data), dtype=np.float32)
    R = np.zeros(np.shape(data), dtype=np.float32)

    data = np.asarray(data)
    G = np.asarray(G)
    d1 = np.asarray(signal.convolve2d(data, [[-1, 0, 0],[0, 0, 0], [0, 0, 1]], mode="same", boundary="symm"))
    d2 = np.asarray(signal.convolve2d(data, [[0, 0, 1], [0, 0, 0], [-1, 0, 0]], mode="same", boundary="symm"))

    df_NE = np.asarray(signal.convolve2d(G, [[0, 0, 0], [0, 1, 0], [-1, 0, 0]], mode="same", boundary="symm"))
    df_SE = np.asarray(signal.convolve2d(G, [[-1, 0, 0], [0, 1, 0], [0, 0, 0]], mode="same", boundary="symm"))
    df_SW = np.asarray(signal.convolve2d(G, [[0, 0, -1], [0, 1, 0], [0, 0, 0]], mode="same", boundary="symm"))
    df_NW = np.asarray(signal.convolve2d(G, [[0, 0, 0], [0, 1, 0], [0, 0, -1]], mode="same", boundary="symm"))

    weight_NE = np.zeros(np.shape(data), dtype=np.float32)
    weight_SE = np.zeros(np.shape(data), dtype=np.float32)
    weight_SW = np.zeros(np.shape(data), dtype=np.float32)
    weight_NW = np.zeros(np.shape(data), dtype=np.float32)

    value_NE = np.zeros(np.shape(data), dtype=np.float32)
    value_SE = np.zeros(np.shape(data), dtype=np.float32)
    value_SW = np.zeros(np.shape(data), dtype=np.float32)
    value_NW = np.zeros(np.shape(data), dtype=np.float32)

    if ((bayer_pattern == "rggb") or (bayer_pattern == "bggr")):

        #== weights for B in R locations
        weight_NE[::2, ::2] = np.abs(d2[::2, ::2]) + np.abs(df_NE[::2, ::2])
        weight_SE[::2, ::2] = np.abs(d1[::2, ::2]) + np.abs(df_SE[::2, ::2])
        weight_SW[::2, ::2] = np.abs(d2[::2, ::2]) + np.abs(df_SW[::2, ::2])
        weight_NW[::2, ::2] = np.abs(d1[::2, ::2]) + np.abs(df_NW[::2, ::2])

        #== weights for R in B locations
        weight_NE[1::2, 1::2] = np.abs(d2[1::2, 1::2]) + np.abs(df_NE[1::2, 1::2])
        weight_SE[1::2, 1::2] = np.abs(d1[1::2, 1::2]) + np.abs(df_SE[1::2, 1::2])
        weight_SW[1::2, 1::2] = np.abs(d2[1::2, 1::2]) + np.abs(df_SW[1::2, 1::2])
        weight_NW[1::2, 1::2] = np.abs(d1[1::2, 1::2]) + np.abs(df_NW[1::2, 1::2])

        weight_NE = np.divide(1., 1. + weight_NE)
        weight_SE = np.divide(1., 1. + weight_SE)
        weight_SW = np.divide(1., 1. + weight_SW)
        weight_NW = np.divide(1., 1. + weight_NW)

        #== directional estimates of B in R locations
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        value_NE[::2, ::2] = temp[::2, 1::2] + df_NE[::2, ::2] / 2.
        value_SE[::2, ::2] = data[1::2, 1::2] + df_SE[::2, ::2] / 2.
        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(data[:, 1]).T, data)), -1, 1)
        value_SW[::2, ::2] = temp[1::2, ::2] + df_SW[::2, ::2] / 2.

        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(temp[:, 1]).T, temp)), -1, 1)
        value_NW[::2, ::2] = temp[::2, ::2] + df_NW[::2, ::2]

        #== directional estimates of R in B locations
        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        value_NE[1::2, 1::2] = temp[::2, 2::2] + df_NE[1::2, 1::2] / 2.
        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((temp, temp[-1]))
        value_SE[1::2, 1::2] = temp[2::2, 2::2] + df_SE[1::2, 1::2] / 2.
        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((data, data[-1]))
        value_SW[1::2, 1::2] = temp[2::2, ::2] + df_SW[1::2, 1::2] / 2.
        value_NW[1::2, 1::2] = data[::2, ::2] + df_NW[1::2, 1::2] / 2.

        RB = np.divide(np.multiply(weight_NE, value_NE) + \
                       np.multiply(weight_SE, value_SE) + \
                       np.multiply(weight_SW, value_SW) + \
                       np.multiply(weight_NW, value_NW),\
                       (weight_NE + weight_SE + weight_SW + weight_NW))

        if (bayer_pattern == "rggb"):

            R[1::2, 1::2] = RB[1::2, 1::2]
            R[::2, ::2] = data[::2, ::2]
            B[::2, ::2] = RB[::2, ::2]
            B[1::2, 1::2] = data[1::2, 1::2]

        elif (bayer_pattern == "bggr"):
            R[::2, ::2] = RB[::2, ::2]
            R[1::2, 1::2] = data[1::2, 1::2]
            B[1::2, 1::2] = RB[1::2, 1::2]
            B[::2, ::2] = data[::2, ::2]


        R[1::2, ::2] = G[1::2, ::2]
        R[::2, 1::2] = G[::2, 1::2]
        R = fill_channel_directional_weight(R, "gbrg")

        B[1::2, ::2] = G[1::2, ::2]
        B[::2, 1::2] = G[::2, 1::2]
        B = fill_channel_directional_weight(B, "gbrg")


    elif ((bayer_pattern == "grbg") or (bayer_pattern == "gbrg")):
        #== weights for B in R locations
        weight_NE[::2, 1::2] = np.abs(d2[::2, 1::2]) + np.abs(df_NE[::2, 1::2])
        weight_SE[::2, 1::2] = np.abs(d1[::2, 1::2]) + np.abs(df_SE[::2, 1::2])
        weight_SW[::2, 1::2] = np.abs(d2[::2, 1::2]) + np.abs(df_SW[::2, 1::2])
        weight_NW[::2, 1::2] = np.abs(d1[::2, 1::2]) + np.abs(df_NW[::2, 1::2])

        #== weights for R in B locations
        weight_NE[1::2, ::2] = np.abs(d2[1::2, ::2]) + np.abs(df_NE[1::2, ::2])
        weight_SE[1::2, ::2] = np.abs(d1[1::2, ::2]) + np.abs(df_SE[1::2, ::2])
        weight_SW[1::2, ::2] = np.abs(d2[1::2, ::2]) + np.abs(df_SW[1::2, ::2])
        weight_NW[1::2, ::2] = np.abs(d1[1::2, ::2]) + np.abs(df_NW[1::2, ::2])

        weight_NE = np.divide(1., 1. + weight_NE)
        weight_SE = np.divide(1., 1. + weight_SE)
        weight_SW = np.divide(1., 1. + weight_SW)
        weight_NW = np.divide(1., 1. + weight_NW)

        #== directional estimates of B in R locations
        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((temp, np.atleast_2d(temp[:, -2]).T))
        value_NE[::2, 1::2] = temp[::2, 2::2] + df_NE[::2, 1::2] / 2.
        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        value_SE[::2, 1::2] = temp[1::2, 2::2] + df_SE[::2, 1::2] / 2.
        value_SW[::2, 1::2] = data[1::2, ::2] + df_SW[::2, 1::2] / 2.

        # repeating the second row at the top of matrix so that sampling does
        # not cause any dimension mismatch, also remove the bottom row
        temp = np.delete(np.vstack((data[1], data)), -1, 0)
        value_NW[::2, 1::2] = temp[::2, ::2] + df_NW[::2, 1::2]

        #== directional estimates of R in B locations
        value_NE[1::2, ::2] = data[::2, 1::2] + df_NE[1::2, ::2] / 2.
        # repeating the column before the last to the right so that sampling
        # does not cause any dimension mismatch
        temp = np.hstack((data, np.atleast_2d(data[:, -2]).T))
        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((temp, temp[-1]))
        value_SE[1::2, ::2] = temp[2::2, 1::2] + df_SE[1::2, ::2] / 2.
        # repeating the row before the last row to the bottom so that sampling
        # does not cause any dimension mismatch
        temp = np.vstack((data, data[-1]))
        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(temp[:, 1]).T, temp)), -1, 1)
        value_SW[1::2, ::2] = temp[2::2, ::2] + df_SW[1::2, ::2] / 2.
        # repeating the second column at the left of matrix so that sampling
        # does not cause any dimension mismatch, also remove the rightmost
        # column
        temp = np.delete(np.hstack((np.atleast_2d(data[:, 1]).T, data)), -1, 1)
        value_NW[1::2, ::2] = temp[::2, ::2] + df_NW[1::2, ::2] / 2.

        RB = np.divide(np.multiply(weight_NE, value_NE) + \
                       np.multiply(weight_SE, value_SE) + \
                       np.multiply(weight_SW, value_SW) + \
                       np.multiply(weight_NW, value_NW),\
                       (weight_NE + weight_SE + weight_SW + weight_NW))

        if (bayer_pattern == "grbg"):

            R[1::2, ::2] = RB[1::2, ::2]
            R[::2, 1::2] = data[::2, 1::2]
            B[::2, 1::2] = RB[::2, 1::2]
            B[1::2, ::2] = data[1::2, ::2]

        elif (bayer_pattern == "gbrg"):
            R[::2, 1::2] = RB[::2, 1::2]
            R[1::2, ::2] = data[1::2, ::2]
            B[1::2, ::2] = RB[1::2, ::2]
            B[::2, 1::2] = data[::2, 1::2]


        R[::2, ::2] = G[::2, ::2]
        R[1::2, 1::2] = G[1::2, 1::2]
        R = fill_channel_directional_weight(R, "rggb")

        B[1::2, 1::2] = G[1::2, 1::2]
        B[::2, ::2] = G[::2, ::2]
        B = fill_channel_directional_weight(B, "rggb")


    return B, R

# # =============================================================
# # function: dbayer_mhc_fast
# #   demosaicing using Malvar-He-Cutler algorithm
# #   http://www.ipol.im/pub/art/2011/g_mhcd/
# # =============================================================
# def debayer_mhc_fast(raw, bayer_pattern="rggb", clip_range=[0, 65535], timeshow=False):
#
#     # convert to float32 in case it was not
#     raw = np.float32(raw)
#
#     # dimensions
#     width, height = utility.helpers(raw).get_width_height()
#
#     # allocate space for the R, G, B planes
#     R = np.empty((height, width), dtype = np.float32)
#     G = np.empty((height, width), dtype = np.float32)
#     B = np.empty((height, width), dtype = np.float32)
#
#     # create a RGB output
#     demosaic_out = np.empty( (height, width, 3), dtype = np.float32 )
#
#     # define the convolution kernels
#     kernel_g_at_rb = [[0., 0., -1., 0., 0.],\
#                       [0., 0., 2., 0., 0.],\
#                       [-1., 2., 4., 2., -1.],\
#                       [0., 0., 2., 0., 0.],\
#                       [0., 0., -1., 0., 0.]] * .125
#
#     kernel_r_at_gr = [[0., 0., .5, 0., 0.],\
#                       [0., -1., 0., -1., 0.],\
#                       [-1., 4., 5., 4., -1.],\
#                       [0., -1., 0., -1., 0.],\
#                       [0., 0., .5, 0., 0.]] * .125
#
#     kernel_b_at_gr = [[0., 0., -1., 0., 0.],\
#                       [0., -1., 4., -1., 0.],\
#                       [.5., 0., 5., 0., .5],\
#                       [0., -1., 4., -1., 0],\
#                       [0., 0., -1., 0., 0.]] * .125
#
#     kernel_r_at_gb = [[0., 0., -1., 0., 0.],\
#                       [0., -1., 4., -1., 0.],\
#                       [.5, 0., 5., 0., .5],\
#                       [0., -1., 4., -1., 0.],\
#                       [0., 0., -1., 0., 0.]] * .125
#
#     kernel_b_at_gb = [[0., 0., .5, 0., 0.],\
#                       [0., -1., 0., -1., 0.],\
#                       [-1., 4., 5., 4., -1.],\
#                       [0., -1., 0., -1., 0.],\
#                       [0., 0., .5, 0., 0.]] * .125
#
#     kernel_r_at_b = [[0., 0., -1.5, 0., 0.],\
#                      [0., 2., 0., 2., 0.],\
#                      [-1.5, 0., 6., 0., -1.5],\
#                      [0., 2., 0., 2., 0.],\
#                      [0., 0., -1.5, 0., 0.]] * .125
#
#     kernel_b_at_r = [[0., 0., -1.5, 0., 0.],\
#                      [0., 2., 0., 2., 0.],\
#                      [-1.5, 0., 6., 0., -1.5],\
#                      [0., 2., 0., 2., 0.],\
#                      [0., 0., -1.5, 0., 0.]] * .125
#
#
#
#     # fill up the directly available values according to the Bayer pattern
#     if (bayer_pattern == "rggb"):
#
#         G[::2, 1::2]  = raw[::2, 1::2]
#         G[1::2, ::2]  = raw[1::2, ::2]
#         R[::2, ::2]   = raw[::2, ::2]
#         B[1::2, 1::2] = raw[1::2, 1::2]
#
#         # Green channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # G at Red location
#                 if (((i % 2) == 0) and ((j % 2) == 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
#                     2. * G[i+1, j], \
#                     -1. * R[i+2, j]])
#                 # G at Blue location
#                 elif (((i % 2) != 0) and ((j % 2) != 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
#                     2. * G[i+1, j],\
#                     -1. * B[i+2, j]])
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Green: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#         # Red and Blue channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # Green locations in Red rows
#                 if (((i % 2) == 0) and ((j % 2) != 0)):
#                     # R at Green locations in Red rows
#                     R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
#                      -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
#                      -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
#                      -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                       .5 * G[i+2, j]])
#
#                     # B at Green locations in Red rows
#                     B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                 # Green locations in Blue rows
#                 elif (((i % 2) != 0) and ((j % 2) == 0)):
#
#                     # R at Green locations in Blue rows
#                     R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                     # B at Green locations in Blue rows
#                     B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
#                     -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
#                     -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
#                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                     .5 * G[i+2, j]])
#
#                 # R at Blue locations
#                 elif (((i % 2) != 0) and ((j % 2) != 0)):
#                     R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
#                     2. * R[i-1, j-1], 2. * R[i-1, j+1], \
#                     -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
#                     2. * R[i+1, j-1], 2. * R[i+1, j+1], \
#                     -1.5 * B[i+2, j]])
#
#                 # B at Red locations
#                 elif (((i % 2) == 0) and ((j % 2) == 0)):
#                     B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
#                     2. * B[i-1, j-1], 2. * B[i-1, j+1], \
#                     -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
#                     2. * B[i+1, j-1], 2. * B[i+1, j+1], \
#                     -1.5 * R[i+2, j]])
#
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#
#     elif (bayer_pattern == "gbrg"):
#
#         G[::2, ::2]   = raw[::2, ::2]
#         G[1::2, 1::2] = raw[1::2, 1::2]
#         R[1::2, ::2]  = raw[1::2, ::2]
#         B[::2, 1::2]  = raw[::2, 1::2]
#
#         # Green channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # G at Red location
#                 if (((i % 2) != 0) and ((j % 2) == 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
#                     2. * G[i+1, j], \
#                     -1. * R[i+2, j]])
#                 # G at Blue location
#                 elif (((i % 2) == 0) and ((j % 2) != 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
#                     2. * G[i+1, j],\
#                     -1. * B[i+2, j]])
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Green: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#         # Red and Blue channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # Green locations in Red rows
#                 if (((i % 2) != 0) and ((j % 2) != 0)):
#                     # R at Green locations in Red rows
#                     R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
#                      -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
#                      -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
#                      -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                       .5 * G[i+2, j]])
#
#                     # B at Green locations in Red rows
#                     B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                 # Green locations in Blue rows
#                 elif (((i % 2) == 0) and ((j % 2) == 0)):
#
#                     # R at Green locations in Blue rows
#                     R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                     # B at Green locations in Blue rows
#                     B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
#                     -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
#                     -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
#                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                     .5 * G[i+2, j]])
#
#                 # R at Blue locations
#                 elif (((i % 2) == 0) and ((j % 2) != 0)):
#                     R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
#                     2. * R[i-1, j-1], 2. * R[i-1, j+1], \
#                     -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
#                     2. * R[i+1, j-1], 2. * R[i+1, j+1], \
#                     -1.5 * B[i+2, j]])
#
#                 # B at Red locations
#                 elif (((i % 2) != 0) and ((j % 2) == 0)):
#                     B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
#                     2. * B[i-1, j-1], 2. * B[i-1, j+1], \
#                     -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
#                     2. * B[i+1, j-1], 2. * B[i+1, j+1], \
#                     -1.5 * R[i+2, j]])
#
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#     elif (bayer_pattern == "grbg"):
#
#         G[::2, ::2]   = raw[::2, ::2]
#         G[1::2, 1::2] = raw[1::2, 1::2]
#         R[::2, 1::2]  = raw[::2, 1::2]
#         B[1::2, ::2]  = raw[1::2, ::2]
#
#         # Green channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # G at Red location
#                 if (((i % 2) == 0) and ((j % 2) != 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
#                     2. * G[i+1, j], \
#                     -1. * R[i+2, j]])
#                 # G at Blue location
#                 elif (((i % 2) != 0) and ((j % 2) == 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
#                     2. * G[i+1, j],\
#                     -1. * B[i+2, j]])
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Green: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#         # Red and Blue channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # Green locations in Red rows
#                 if (((i % 2) == 0) and ((j % 2) == 0)):
#                     # R at Green locations in Red rows
#                     R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
#                      -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
#                      -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
#                      -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                       .5 * G[i+2, j]])
#
#                     # B at Green locations in Red rows
#                     B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                 # Green locations in Blue rows
#                 elif (((i % 2) != 0) and ((j % 2) != 0)):
#
#                     # R at Green locations in Blue rows
#                     R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                     # B at Green locations in Blue rows
#                     B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
#                     -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
#                     -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
#                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                     .5 * G[i+2, j]])
#
#                 # R at Blue locations
#                 elif (((i % 2) != 0) and ((j % 2) == 0)):
#                     R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
#                     2. * R[i-1, j-1], 2. * R[i-1, j+1], \
#                     -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
#                     2. * R[i+1, j-1], 2. * R[i+1, j+1], \
#                     -1.5 * B[i+2, j]])
#
#                 # B at Red locations
#                 elif (((i % 2) == 0) and ((j % 2) != 0)):
#                     B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
#                     2. * B[i-1, j-1], 2. * B[i-1, j+1], \
#                     -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
#                     2. * B[i+1, j-1], 2. * B[i+1, j+1], \
#                     -1.5 * R[i+2, j]])
#
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#     elif (bayer_pattern == "bggr"):
#
#         G[::2, 1::2]  = raw[::2, 1::2]
#         G[1::2, ::2]  = raw[1::2, ::2]
#         R[1::2, 1::2] = raw[1::2, 1::2]
#         B[::2, ::2]   = raw[::2, ::2]
#
#         # Green channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # G at Red location
#                 if (((i % 2) != 0) and ((j % 2) != 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * R[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * R[i, j-2], 2. * G[i, j-1], 4. * R[i,j], 2. * G[i, j+1], -1. * R[i, j+2],\
#                     2. * G[i+1, j], \
#                     -1. * R[i+2, j]])
#                 # G at Blue location
#                 elif (((i % 2) == 0) and ((j % 2) == 0)):
#                     G[i, j] = 0.125 * np.sum([-1. * B[i-2, j], \
#                     2. * G[i-1, j], \
#                     -1. * B[i, j-2], 2. * G[i, j-1], 4. * B[i,j], 2. * G[i, j+1], -1. * B[i, j+2], \
#                     2. * G[i+1, j],\
#                     -1. * B[i+2, j]])
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Green: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#         # Red and Blue channel
#         for i in range(no_of_pixel_pad, height + no_of_pixel_pad):
#
#             # to display progress
#             t0 = time.process_time()
#
#             for j in range(no_of_pixel_pad, width + no_of_pixel_pad):
#
#                 # Green locations in Red rows
#                 if (((i % 2) != 0) and ((j % 2) == 0)):
#                     # R at Green locations in Red rows
#                     R[i, j] = 0.125 * np.sum([.5 * G[i-2, j],\
#                      -1. * G[i-1, j-1], -1. * G[i-1, j+1], \
#                      -1. * G[i, j-2], 4. * R[i, j-1], 5. * G[i,j], 4. * R[i, j+1], -1. * G[i, j+2], \
#                      -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                       .5 * G[i+2, j]])
#
#                     # B at Green locations in Red rows
#                     B[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * B[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * B[i+1,j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                 # Green locations in Blue rows
#                 elif (((i % 2) == 0) and ((j % 2) != 0)):
#
#                     # R at Green locations in Blue rows
#                     R[i, j] = 0.125 * np.sum([-1. * G[i-2, j], \
#                     -1. * G[i-1, j-1], 4. * R[i-1, j], -1. * G[i-1, j+1], \
#                     .5 * G[i, j-2], 5. * G[i,j], .5 * G[i, j+2], \
#                     -1. * G[i+1, j-1], 4. * R[i+1, j],  -1. * G[i+1, j+1], \
#                     -1. * G[i+2, j]])
#
#                     # B at Green locations in Blue rows
#                     B[i, j] = 0.125 * np.sum([.5 * G[i-2, j], \
#                     -1. * G [i-1, j-1], -1. * G[i-1, j+1], \
#                     -1. * G[i, j-2], 4. * B[i, j-1], 5. * G[i,j], 4. * B[i, j+1], -1. * G[i, j+2], \
#                     -1. * G[i+1, j-1], -1. * G[i+1, j+1], \
#                     .5 * G[i+2, j]])
#
#                 # R at Blue locations
#                 elif (((i % 2) == 0) and ((j % 2) == 0)):
#                     R[i, j] = 0.125 * np.sum([-1.5 * B[i-2, j], \
#                     2. * R[i-1, j-1], 2. * R[i-1, j+1], \
#                     -1.5 * B[i, j-2], 6. * B[i,j], -1.5 * B[i, j+2], \
#                     2. * R[i+1, j-1], 2. * R[i+1, j+1], \
#                     -1.5 * B[i+2, j]])
#
#                 # B at Red locations
#                 elif (((i % 2) != 0) and ((j % 2) != 0)):
#                     B[i, j] = 0.125 * np.sum([-1.5 * R[i-2, j], \
#                     2. * B[i-1, j-1], 2. * B[i-1, j+1], \
#                     -1.5 * R[i, j-2], 6. * R[i,j], -1.5 * R[i, j+2], \
#                     2. * B[i+1, j-1], 2. * B[i+1, j+1], \
#                     -1.5 * R[i+2, j]])
#
#             if (timeshow):
#                 elapsed_time = time.process_time() - t0
#                 print("Red/Blue: row index: " + str(i-1) + " of " + str(height) + \
#                       " | elapsed time: " + "{:.3f}".format(elapsed_time) + " seconds")
#
#     else:
#         print("Invalid bayer pattern. Valid pattern can be rggb, gbrg, grbg, bggr")
#         return demosaic_out # This will be all zeros
#
#     # Fill up the RGB output with interpolated values
#     demosaic_out[0:height, 0:width, 0] = R[no_of_pixel_pad : height + no_of_pixel_pad, \
#                                            no_of_pixel_pad : width + no_of_pixel_pad]
#     demosaic_out[0:height, 0:width, 1] = G[no_of_pixel_pad : height + no_of_pixel_pad, \
#                                            no_of_pixel_pad : width + no_of_pixel_pad]
#     demosaic_out[0:height, 0:width, 2] = B[no_of_pixel_pad : height + no_of_pixel_pad, \
#                                            no_of_pixel_pad : width + no_of_pixel_pad]
#
#     demosaic_out = np.clip(demosaic_out, clip_range[0], clip_range[1])
#     return demosaic_out
