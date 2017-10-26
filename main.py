# ===================================
# Import the libraries
# ===================================
import numpy as np
from matplotlib import pylab as plt
import imaging
import utility
import os,sys

# ===================================
# Which stages to run
# ===================================
do_denoise = True #False
do_black_level_correction = True
do_lens_shading_correction = True
do_bad_pixel_correction = True #False
do_channel_gain_white_balance = True
do_demosaic = True
do_color_correction = True
do_gamma = True
do_tone_mapping = True
do_sharpening = True #False

# ===================================
# Remove all the .png files
os.system("rm images/*.png")
# ===================================

# ===================================
# raw image and set up the metadata
# ===================================
temp = np.fromfile("images/DSC_1339_768x512_rggb.raw", dtype="uint16", sep="")
temp = temp.reshape([512, 768])

raw = imaging.ImageInfo("1339_768x512_rggb", temp)
raw.set_color_space("raw")
raw.set_bayer_pattern("rggb")
raw.set_channel_gain((1.94921875, 1.0, 1.0, 1.34375))
raw.set_bit_depth(14)
raw.set_black_level((600, 600, 600, 600))
raw.set_white_level((15520, 15520, 15520, 15520))
# the ColorMatrix2 found from the metadata
raw.set_color_matrix([[.9020, -.2890, -.0715],\
                      [-.4535, 1.2436, .2348],\
                      [-.0934, .1919,  .7086]])


data = raw.data

# ===================================
# Add noise
# ===================================
if do_denoise:
    noise_mean = 0
    noise_standard_deviation = 100
    seed = 100
    clip_range = [600, 65535]
    data = utility.synthetic_image_generate(\
    raw.get_width(), raw.get_height()).create_noisy_image(\
    data, noise_mean, noise_standard_deviation, seed, clip_range)
else:
    pass

# ===================================
# Black level correction
# ===================================
if do_black_level_correction:
    data = imaging.black_level_correction(data, \
                                          raw.get_black_level(),\
                                          raw.get_white_level(),\
                                          [0, 2**raw.get_bit_depth() - 1])
    utility.imsave(data, "images/out_black_level_correction.png", "uint16")
else:
    pass

# ===================================
# Lens shading correction
# ===================================
if do_lens_shading_correction:
    # normally dark_current_image and flat_field_image are
    # captured in the image quality lab using flat field chart
    # here we are synthetically generating thouse two images
    dark_current_image, flat_field_image = utility.synthetic_image_generate(\
    raw.get_width(), raw.get_height()).create_lens_shading_correction_images(\
                                    0, 65535, 40000)

    # save the dark_current_image and flat_field_image for viewing
    utility.imsave(dark_current_image, "images/dark_current_image.png", "uint16")
    utility.imsave(flat_field_image, "images/flat_field_image.png", "uint16")

    data = imaging.lens_shading_correction(data).flat_field_compensation(\
    dark_current_image, flat_field_image)

    # data = lsc.approximate_mathematical_compensation([0.01759, -28.37, -13.36])
    utility.imsave(data, "images/out_lens_shading_correction.png", "uint16")

else:
    pass

# ===================================
# Bad pixel correction
# ===================================
if do_bad_pixel_correction:
    neighborhood_size = 3
    data = imaging.bad_pixel_correction(data, neighborhood_size)
    utility.imsave(data, "images/out_bad_pixel_correction.png", "uint16")
else:
    pass

# ===================================
# Channel gain for white balance
# ===================================
if do_channel_gain_white_balance:
    data = imaging.channel_gain_white_balance(data,\
                                              raw.get_channel_gain())
    utility.imsave(data, "images/out_channel_gain_white_balance.png", "uint16")
else:
    pass

# ===================================
# Bayer denoising
# ===================================
if do_denoise:

    # bayer denoising parameters
    neighborhood_size = 5
    initial_noise_level = 65535 * 10 / 100
    hvs_min = 1000
    hvs_max = 2000
    clip_range = [0, 65535]
    threshold_red_blue = 1300

    # data is the denoised output, ignoring the second output
    data, _ = imaging.bayer_denoising(data).utilize_hvs_behavior(\
    raw.get_bayer_pattern(), initial_noise_level, hvs_min, hvs_max, threshold_red_blue, clip_range)

    utility.imsave(data, "images/out_bayer_denoising.png", "uint16")
    # utility.imsave(np.clip(texture_degree_debug*65535, 0, 65535), "images/out_texture_degree_debug.png", "uint16")

else:
    pass

# ===================================
# Demosacing
# ===================================
if do_demosaic:
    data = imaging.demosaic(data, raw.get_bayer_pattern()).mhc(False)
    utility.imsave(data, "images/out_demosaic.png", "uint16")
else:
    pass

# ===================================
# Color aliasing correction
# ===================================

# ===================================
# Color correction
# ===================================
if do_color_correction:
    data = imaging.color_correction(data, raw.get_color_matrix()).apply_cmatrix()
    utility.imsave(data, "images/out_color_correction.png", "uint16")
else:
    pass

# ===================================
# Gamma
# ===================================
if do_gamma:

    # by value
    #data = imaging.nonlinearity(data, "gamma").by_value(1/2.2, [0, 65535])

    # by table
    data = imaging.nonlinearity(data, "gamma").by_table("tables/GammaE.txt", "gamma", [0, 65535])

    utility.imsave(data, "images/out_gamma.png", "uint16")

else:
    pass

# ===================================
# Chromatic aberration correction
# ===================================

# ===================================
# Tone mapping
# ===================================
if do_tone_mapping:
    data = imaging.tone_mapping(data).nonlinear_masking(1.0)

    utility.imsave(data, "images/out_tone_mapping.png", "uint16")

else:
    pass

# ===================================
# Memory color enhancement
# ===================================

# ===================================
# Noise reduction
# ===================================

# ===================================
# Sharpening
# ===================================
if do_sharpening:

    data = imaging.sharpening(data).unsharp_masking()

    utility.imsave(data, "images/out_sharpening.png", "uint16")

else:
    pass

# ===================================
# Distortion correction
# ===================================
