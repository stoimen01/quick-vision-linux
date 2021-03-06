from __future__ import division
import math
import pprint
import scipy.misc
import numpy as np
from scipy.misc import imresize

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def transform_img(i):
    return np.array(i)/127.5 - 1.


def crop_img(file, size):
    return np.array(center_crop(imread(file), size))


def center_crop(x, crop_h, crop_w=None):
    resize_w = crop_h
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def resize(x, size):
    x = np.copy((x+1.)*127.5).astype("uint8")
    y = imresize(x, [size, size])
    return y


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def save_image(image, image_path):
    img = inverse_transform(image)
    return scipy.misc.imsave(image_path, img[0])


def inverse_transform(image):
    return (image+1.)/2.
