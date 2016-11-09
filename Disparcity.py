import cv2
import numpy
from PIL import Image
from matplotlib.pyplot import *
from numpy import *

def getDot(im, n, y, x):
    if (x+n+1 > len(im[0]) | y+n+1 > len(im)):
        print("skipped")
        return 0
    temp = zeros((n * 2, n * 2))
    print("ys: {0} to {1}. xs: {2} to {3}".format(y, y+2*n, x, x+2*n))
    temp[:,:] = im[y:y+2*n, x:x+2*n]

    print(dot)
    return dot

n = 2; #Size of regions to search

#make images into grayscale in order for easier comparison
imLeft = cv2.imread('empire.jpg')
grayL = cv2.cvtColor(imLeft,cv2.COLOR_BGR2GRAY)
imRight = cv2.imread('empire.jpg')
grayR = cv2.cvtColor(imRight,cv2.COLOR_BGR2GRAY)

left = array(grayL)
shaep = left.shape
left = cv2.copyMakeBorder(left, 0, (n*2), 0, (n*2), cv2.BORDER_REFLECT)
right = array(grayR)

regionsLeft = zeros((len(left), len(left[0]), (2*n)))
temp = zeros((2*n, 2*n))

for y in range(shaep[0]+1):
    for x in range(shaep[1]+1):
        temp[:,:] = left[y:y+2*n, x:x+2*n]
        dot = numpy.ndarray.flatten(ndarray.dot(temp,temp))
        print("x:{0}  y:{1}".format(x, y))


#    dot = numpy.ndarray.flatten(ndarray.dot(temp,temp))


#test = array(test, 'uint8')
#cv2.imshow(test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



