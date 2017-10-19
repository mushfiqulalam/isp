# ===================================
# Import the libraries
# ===================================
import numpy as np
from matplotlib import pylab as plt
import imaging
import utility
import os,sys

# ===================================
# Remove all the .png files
os.system("rm images/*.png")
# ===================================

# ===================================
# raw image
# ===================================
temp = np.fromfile("images/DSC_1339_768x512_rggb.raw", dtype="uint16", sep="")
temp = temp.reshape([512, 768])

raw = imaging.ImageInfo("1339_768x512_rggb", temp)
raw.set_color_space("raw")
raw.set_bayer_pattern("rggb")
raw.set_channel_gain((1.94921875, 1.0, 1.0, 1.34375))
raw.set_bit_depth(14)
raw.set_black_level((600, 600, 600, 600))

# ===================================
# Add noise
# ===================================
temp = utility.synthetic_image_generate(raw.get_width(), raw.get_height())
noise_mean = 0
noise_standard_deviation = 500
seed = 100
clip_range = [600, 65535]
data = temp.create_noisy_image(raw.data, noise_mean, noise_standard_deviation, seed, clip_range)

# ===================================
# Black level correction
# ===================================
data = imaging.black_level_correction(data, \
                                      raw.get_black_level())
utility.imsave(data, "images/out_black_level_correction.png", "uint16")

# ===================================
# Lens shading correction
# ===================================
# normally dark_current_image and flat_field_image are
# captured in the image quality lab using flat field chart
# here we are synthetically generating thouse two images
temp = utility.synthetic_image_generate(raw.get_width(), raw.get_height())
dark_current_image, flat_field_image = temp.create_lens_shading_correction_images(\
  0, 65535, 40000)
# save the dark_current_image and flat_field_image for viewing
utility.imsave(dark_current_image, "images/dark_current_image.png", "uint16")
utility.imsave(flat_field_image, "images/flat_field_image.png", "uint16")
temp = imaging.lens_shading_correction(data)
data = temp.flat_field_compensation(dark_current_image, flat_field_image)
# data = lsc.approximate_mathematical_compensation([0.01759, -28.37, -13.36])
utility.imsave(data, "images/out_lens_shading_correction.png", "uint16")

# ===================================
# Bad pixel correction
# ===================================
neighborhood_size = 3
data = imaging.bad_pixel_correction(data, neighborhood_size)
utility.imsave(data, "images/out_bad_pixel_correction.png", "uint16")

# ===================================
# Channel gain for white balance
# ===================================
data = imaging.channel_gain_white_balance(data,\
                                          raw.get_channel_gain())
utility.imsave(data, "images/out_channel_gain_white_balance.png", "uint16")

# ===================================
# Bayer denoising
# ===================================
temp = imaging.bayer_denoising(data)
neighborhood_size = 5
initial_noise_level = 1000
hvs_min = 1000
hvs_max = 5000
clip_range = [0, 65535]
threshold_red_blue = 100
data = temp.utilize_hvs_behavior(raw.get_bayer_pattern(), neighborhood_size, initial_noise_level, hvs_min, hvs_max, threshold_red_blue, clip_range)
utility.imsave(data, "images/out_bayer_denoising.png", "uint16")

# ===================================
# Demosacking
# ===================================
data = imaging.demosaic_mhc(np.uint16(data), raw.get_bayer_pattern(), [0, 65535], False)
utility.imsave(data, "images/out_demosaic.png", "uint16")

# ===================================
# Color aliasing correction
# ===================================

# ===================================
# Color correction
# ===================================

# ===================================
# Gamma
# ===================================
temp = imaging.nonlinearity(data, "gamma")
data = temp.by_value(1/2.2, [0, 65535])
utility.imsave(data, "images/out_gamma.png", "uint16")

# ===================================
# Chromatic aberration correction
# ===================================

# ===================================
# Tone mapping
# ===================================

# ===================================
# Memory color enhancement
# ===================================

# ===================================
# Noise reduction
# ===================================

# ===================================
# Sharpening
# ===================================

# ===================================
# Distortion correction
# ===================================
