from PIL import Image
from matplotlib import pyplot
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import filters
from scipy.signal import convolve2d

from gradMagnitude import get

#bring image, make grayscale
im = array(Image.open('empire.jpg').convert('L'))
gray()
imshow(im)

#create new image with Prewitt-filtered original image
imPrewitt = zeros(im.shape)
filters.prewitt(im, 1, imPrewitt)
imPrewittMagnitude = get(imPrewitt)
pyplot.figure("Prewitt Gradient Magnitude")
imshow(imPrewittMagnitude)

#create new image with Sobel-filtered original image
imSobel = zeros(im.shape)
filters.sobel(im, 1, imSobel)
imSobelMagnitude = get(imSobel)
figure('Sobel Gradient Magnitude')
imshow(imSobelMagnitude)

#create new images wirh [-1,1] and [-1,1]T filters
filt = np.array([-1,0])
xArr, yArr = np.gradient(np.array(im, dtype=np.float))
for x in range(1, len(im)-1):
    xArr[x, :] = np.convolve(xArr[x, :], filt)
for y in range(0, len(im[x])):
    yArr[y, :] = np.convolve(yArr[y, :], filt)
simpleMagnitude = np.sqrt(xArr**2 + yArr**2)
figure('[-1,0] & [-1,0]T Filters Magnitude')
imshow(simpleMagnitude)

show()
