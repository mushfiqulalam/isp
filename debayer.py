import numpy as np
import math
import time
import utility

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
