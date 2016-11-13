#import cv2
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

def colorHistogram(im, n):
    shp = (len(im), len(im[0]))
    step = 255/n

    #separate each color channel
    red = zeros(shp)
    green = zeros(shp)
    blue = zeros(shp)
    red[:,:] = im[:,:,0]
    green[:,:] = im[:,:,1]
    blue[:,:] = im[:,:,2]

    #quantize
    red[:,:] = rint(red[:,:]/step)
    green[:,:] = rint(green[:,:]/step)
    blue[:,:] = rint(blue[:,:]/step)
    red = ndarray.flatten(red).astype(int)
    green = ndarray.flatten(green).astype(int)
    blue = ndarray.flatten(blue).astype(int)

    #fill histogram
    h = np.zeros((3, n))
    for i in range(len(red)):
        h[0][red[i]-1] += 1
        h[1][green[i]-1] += 1
        h[2][blue[i]-1] += 1

    return h

def cubeColorHistogram(im, n):
    width = len(im)
    height = len(im[0])
    depth = len(im[0][0])
    h = zeros((n,n,n))
    step = 255/n

    #quantize cubically
    indices = zeros(im.shape)
    indices[:,:,:] = rint(im[:,:,:]/step)
    indices = indices.astype(int)

    #fill cubic histogram
    for x in range(width):
        for y in range(height):
            h[indices[x][y][0]-1][indices[x][y][1]-1][indices[x][y][2]-1] += 1

    return h

def censusTransform(gray):
    width = len(gray)
    height = len(gray[0])
    census = zeros(gray.shape)
    binaryMatrix = [[1,2,4],[8,16,32],[64,128,256]]
    region = zeros((3,3))
    gray = cv2.copyMakeBorder(gray,2,2,2,2,cv2.BORDER_CONSTANT,value=0)#pad image for easier computations

    for x in range(2,width-1):
        for y in range(2,height-1):
            region = gray[x-2:x+1, y-2:y+1]
            threshold = sum(region)/9
            region[:,:] = rint(region[:,:]/threshold)
            idxONE = region[:,:] < 0
            region[idxONE] = 1
            census[x][y] = sum(region*binaryMatrix)#This converts from binary to decimal

    return census

def getHamming(a, b):
    assert len(a) == len(b)
    return sum(c1 != c2 for c1, c2 in zip(a, b))

def hamming(im1, im2):
    ham = zeros(shape(im1))
    census1 = censusTransform(im1).astype(int)
    census2 = censusTransform(im2).astype(int)

    for x in range(len(im1)):
        for y in range(len(im1[0])):
            ham[x][y] = getHamming(bin(census1[x][y]), bin(census2[x][y]))
    return ham

# im = cv2.imread('empire.jpg')#PLACE YOUR IMAGE HERE
# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#
# h = colorHistogram(im, 10)
# print("COLOR HISTOGRAM")
# print(h)
# print()
#
# h3 = cubeColorHistogram(im, 3)
# z, x, y = nonzero(h3)
# fig = figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# show()
#
# c = censusTransform(gray)
# print("CENSUS TRANSFORM")
# print(c)
# print()
#
# ham = hamming(gray, gray)
# print("HAMMING")
# print(ham)