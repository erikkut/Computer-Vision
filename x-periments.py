from PIL import Image
from matplotlib import pyplot
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import filters

# im = array(Image.open('C:/Users/Eric/Desktop/bill_of_sale.jpg').convert('L'))
im = array(Image.open('C:/Users/Eric/Desktop/bill_of_sale.jpg'))
# gray()
imshow(im)

imMod = zeros(im.shape)
imMod = filters.gaussian_filter(im, 1)
imshow(imMod)
show()

im = Image.fromarray(imMod)
im.save('C:/Users/Eric/Desktop/bill_sale.jpg')
