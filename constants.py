# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]