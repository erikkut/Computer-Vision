import numpy
from PIL import Image
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import *


def blurScale(im, g):
    for x in range(0,10):
        temp = filters.gaussian_filter(im, g)
        im = im + (im - temp) #This way, difference is much easier to see, used to be im = im - temp
        figure("Iteration # {0}".format(x+1))
        imshow(im)
        gray()
    show()

def resizeImgArray(im, percentage):
    width = int(len(im) * percentage / 100)
    height = int(len(im[0]) * percentage / 100)
    size = (height, width)

    temp = Image.fromarray(im, 'L')
    temp = temp.resize(size, Image.ANTIALIAS)
    temp = array(temp)

    return temp

def resizeImgArrayNoise(im, percentage):
    width = int(len(im) * percentage / 100)
    height = int(len(im[0]) * percentage / 100)
    size = (height, width)
    sizeOrig = (len(im[0]), len(im))

    temp = Image.fromarray(im, 'L')
    temp = temp.resize(size, Image.ANTIALIAS)
    temp = temp.resize(sizeOrig, Image.ANTIALIAS)
    temp = array(temp)

    return temp

def resizeScale(im):
    for x in range(0,10):
        temp = resizeImgArrayNoise(im, 90)
        im = im + (im - temp) #difference substracted from original recursively
        figure("iteration # {0} scaling difference @ 10% reduction".format(x+1))
        imshow(im)
        gray()
    show()

def roundTo(x, decimals):
    return np.around(x - 10 ** (-(decimals + 5)), decimals=decimals)

def HOF(im, bars, left, right, bot, top):
    region = im[left:right, bot:top]

    #get FILTER [-1,1] and MAGNITUDE of region
    X, Y = np.gradient(np.array(region, dtype=np.float)) #equivalent to [-1,1] filter
    magnitude = np.sqrt(X**2 + Y**2)
    imshow(magnitude, cmap = cm.gray)##############
    show()#################

    #(angles of direction per pixel)
    directions = arctan2(Y[:], X[:])
    directions = directions[:][:] * 180 / pi#also adding 180 to represent every angle in 360 degrees
    idxNeg = directions[:,:] < 0
    directions[idxNeg] += 360

    h = zeros(bars+1)
    step = 360 / bars

    directions = ndarray.flatten((rint(directions / step)).astype(int))
    magnitude = ndarray.flatten(magnitude.astype(int))

    # create histogram of cumulative gradience magnitude per direction
    for i in range(0, len(directions)):
        h[directions[i]] += 1#+= magnitude[i]
    # ADDING first magnitude to last because 0 = 360 in terms of degrees
    h[0] += h[-1]
    # DELETING last magnitudes because of above
    h[-1] = 0
    h = h[0:-1]

    return h

test = array(Image.open('empire.jpg').convert('L'))
im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = filters.gaussian_filter(im, 2)

region = im[0:128,0:128]

h = HOF(im, 4, 0,128, 0,128)

plot(h)
xlabel('Degree (* 90)')
ylabel('Ocurrences')
title('Histogram')
axis([0, len(h)-1, 0, amax(h)+2])
grid(True)

figure()
imshow(region, cmap = cm.gray)
title('woah')
show()
blurScale(test, 2)
resizeScale(test)


