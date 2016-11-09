from PIL import Image
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import *

im = Image.open('empire.jpg').convert('L')
gray()
im2 = filters.gaussian_filter(im, 5)
q = im

for x in range(0, 800):
    for y in range(0, 569):
        q[x][y] = im[x][y] / (im[x][y] * im2[x][y])

figure('orig')
imshow(im)
figure('blur = 5')
imshow(im2)
figure('quotient')
imshow(q)

show()