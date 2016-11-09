import cv2
from PIL import Image
from matplotlib.pyplot import *
from numpy.ma import *
from pylab import *
import rof
from scipy.ndimage import filters

im = array(Image.open('empire.jpg').convert('L'))
im2 = filters.gaussian_filter(im, 5)
U,T = rof.denoise(im2,im2)

figure()
gray()
imshow(U)
axis('equal')
axis('off')
show()