import cv2
from timeit import default_timer
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

def gradientDirection(im):
    filter = array([[-1,0,1],[-1,0,1],[-1,0,1]])
    xGrad = cv2.filter2D(im, cv2.CV_64F, cv2.flip(filter,-1),(-1,-1))
    yGrad = cv2.filter2D(im, cv2.CV_64F, cv2.flip(transpose(filter),-1),(-1,-1))
    
    directions = arctan2(absolute(yGrad), absolute(xGrad))
    return directions

def HOF(im, bars, left, right, bot, top):
    region = im[left:right, bot:top]
    #region = im[left:right, bot:top]

    #get FILTER [-1,1] and MAGNITUDE of region
    X, Y = np.gradient(np.array(region, dtype=np.float)) #equivalent to [-1,1] filter
    magnitude = np.sqrt(X**2 + Y**2)

    #(angles of direction per pixel)
    directions = arctan2(Y[:], X[:])
    
    directions = directions[:][:] * 180 / pi#also adding 180 to represent every angle in 360 degrees
    idxNeg = directions[:,:] < 0
    directions[idxNeg] += 360

    h = zeros(bars+1)
    step = 360 / bars

    #FLATTEN arrays to make them easier to use, and divide by step to make the histogram
    directions = ndarray.flatten((rint(directions / step)).astype(int))
    magnitude = ndarray.flatten(magnitude)

    #CREATE histogram of cumulative gradience magnitude per direction
    steps = arange(len(h))+1
    for x in range(0, len(h)):
        temp = directions < steps[x]
        h[x] = (sum(magnitude[temp])-sum(h[0:x]))

    # ADDING first magnitude to last because 0 = 360 in terms of degrees
    h[0] += h[-1]
    # DELETING last index of magnitudes because of above
    h[-1] = 0
    h = h[0:-1]
    #ROUND to nearest int
    h = ndarray.flatten(rint(h).astype(int))

    return h

#start = default_timer()
#print(default_timer()-start)

#blurScale(test, 2)
#resizeScale(test)


