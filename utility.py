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
# function: distance_euclid
#   returns Euclidean distance between two points
# =============================================================
def distance_euclid(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)


# =============================================================
# class: synthetic_image_generate
#   creates sysnthetic images for different purposes
# =============================================================
class synthetic_image_generate:
    def __init__(self, name="synthetic_image"):
        self.name = name

    def create_lens_shading_correction_images(self, width, height, dark_current=0, flat_max=65535, flat_min=0, clip_max=65535, clip_min=0):
        dark_current_image = dark_current * np.ones((height, width), dtype=np.float32)
        flat_field_image = np.empty((height, width), dtype=np.float32)

        center_pixel_pos = [height/2, width/2]
        max_distance = distance_euclid(center_pixel_pos, [height, width])

        for i in range(0, height):
            for j in range(0, width):
                flat_field_image[i, j] = (max_distance - distance_euclid(center_pixel_pos, [i, j])) / max_distance
                flat_field_image[i, j] = flat_min + flat_field_image[i, j] * (flat_max - flat_min)

        dark_current_image = np.clip(dark_current_image, clip_min, clip_max)
        flat_field_image = np.clip(flat_field_image, clip_min, clip_max)

        return dark_current_image, flat_field_image

    def create_zone_plate_image(self):
        pass

    def create_color_gradient_image(self):
        pass

    def create_random_noise_image(self):
        pass
