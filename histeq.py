import numpy
from PIL import Image
from matplotlib.pyplot import *
from numpy import *

def getHisteq(im,nbr_bins=256):
    imhist, bins = histogram(im.flatten(), nbr_bins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    equalized = numpy.interp(im.flatten(), bins[:-1], cdf)

    return equalized.reshape(im.shape)

#im = array(Image.open('input1.jpg').convert('L'))
#gray()
#im2 = getHisteq(im)
#imshow(im2)
#show()
#Image.fromarray(im).convert('RGB').save('input1histeqGray.png')
#Image.fromarray(im2).convert('RGB').save('input1histeq.png')