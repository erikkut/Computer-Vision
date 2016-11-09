from numpy import *

def unsharpen(im, im2):
    result = im - ((im-im2)/32)

    result = array(result, 'uint8')
    return result
