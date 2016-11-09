import numpy
from PIL import Image
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import filters
import gradMagnitude


def stdDeviation(im, n, type):
    if(ndim(im) == 2):
        print('good image')
    else:
        print('Image found to be RGB, converting to grayscale...')
        im = Image.fromarray(im).convert('L')
        gray()

    im = array(im)
    if(type == 'slow'):
        R = zeros(im.shape)
        bottom = math.floor((n-1)/2)
        top = math.ceil((n-1)/2)
        xMax = len(im) - top
        yMax = len(im[0]) - top

        for x in range(bottom, xMax):
            for y in range(bottom, yMax):
                patch = im[x-bottom:x+top+1, y-bottom:y+top+1]

                #R[x][y] = np.std(B)
                sum = np.cumsum(patch)
                var2 = sum[len(sum)-1] / (n*n)
                var2 *= var2

                sum = np.cumsum(patch * patch)
                var1 = sum[len(sum)-1] / (n*n)

                var = var1-var2
                if(var < 0):
                    var *= -1
                stdDev = sqrt(var)

                R[x][y] = stdDev


        return R

    if(type == 'fast'):
        xMax = len(im)
        yMax = len(im[0])

        #ADDING THE PADDING
        regionSize = (xMax+n, yMax+n)
        padded = zeros(regionSize)
        padded[n:xMax+n, n:yMax+n] = im[0:xMax, 0:yMax]
        regionSize = (n, n)
        a = b = c = d = zeros(regionSize)

        for y in range(0, xMax+n):
            padded[y,:] = cumsum(padded[y,:])
        for x in range(0, yMax+n):
            padded[:,x] = cumsum(padded[:, x])
        #print(padded)

        xMax += n
        yMax += n
        b = padded[n:xMax, 0:yMax-n]
        c = padded[0:xMax-n, n:yMax]

        a = padded[n:xMax, n:yMax]
        d = padded[0:xMax-n, 0:yMax-n]

        sum = a-b-c+d
        var1 = (sum/(n*n))**2
        var2 = sum*sum/(n*n)
        var = var2-var1
        stdDev = var
        stdDev[:,:] = numpy.sqrt(var[:,:])
        #stdDev = stdDev.astype(int)
        return stdDev





#test = array(Image.open('empire.jpg').convert('L'))
#gray()
#std5 = stdDeviation(test, 2, 'slow')
#std5f = stdDeviation(test, 2, 'fast')
#imshow(std5)
#figure('fast')
#imshow(std5f)
#show()

#sob = filters.sobel(test)
#pre = filters.prewitt(test)
#mag = gradMagnitude.get(test)

#sob = array(sob, 'uint8')
#pre = array(pre, 'uint8')
#mag = np.array(mag)

#Image.fromarray(std2).convert('RGB').save('1std2.jpeg')
#Image.fromarray(std5).convert('RGB').save('1std5.jpeg')
#Image.fromarray(std10).convert('RGB').save('1std10.jpeg')
#Image.fromarray(sob).convert('RGB').save('1Sobel.jpeg')
#Image.fromarray(pre).convert('RGB').save('1Prewitt.jpeg')
#Image.fromarray(mag).convert('RGB').save('1Magnitude.jpeg')
