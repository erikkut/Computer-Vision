import rof
from histeq import getHisteq
from PIL import Image
from matplotlib.pyplot import *
from numpy.ma import array
from scipy.ndimage import *
from unsharpMask import unsharpen


def window(name, image):
    figure(name)
    imshow(image)

#GET IMAGES
im = array(Image.open('empire.jpg'))
imGray = array(Image.open('empire.jpg').convert('L'))
gray()
Image.fromarray(imGray).convert('RGB').save('Lab1/grayscale original.jpeg')

#BLUR IMAGES USING GAUSSIAN FILTER
gammas = [2, 5, 10]
im2 = filters.gaussian_filter(im, gammas[0])
im3 = filters.gaussian_filter(im, gammas[1])
im4 = filters.gaussian_filter(im, gammas[2])
#higher gamma leads to more blurred image because of the higher std deviation (pixel intensities are more spread out)
im2Gray = filters.gaussian_filter(imGray, gammas[0])
im3Gray = filters.gaussian_filter(imGray, gammas[1])
im4Gray = filters.gaussian_filter(imGray, gammas[2])

#GAUSSIAN BLUR AND UNSHARP MASKING WITH GAMMA 2
Image.fromarray(im2).convert('RGB').save('Lab1/gaussian blur of 2.jpeg')
unsharp2 = unsharpen(im, im2)
Image.fromarray(unsharp2).convert('RGB').save('Lab1/unsharp masking gaussian blur of 2.jpeg')
unsharp2Gray = unsharpen(imGray, im2Gray)
Image.fromarray(unsharp2Gray).convert('RGB').save('Lab1/unsharp masking gaussian blur of 2 gray.jpeg')

#GAUSSIAN BLUR AND UNSHARP MASKING WITH GAMMA 5
Image.fromarray(im3).convert('RGB').save('Lab1/gaussian blur of 5.jpeg')
temp = unsharpen(im, im3)
Image.fromarray(temp).convert('RGB').save('Lab1/unsharp masking gaussian blur of 5.jpeg')
temp = unsharpen(imGray, im3Gray)
Image.fromarray(temp).convert('RGB').save('Lab1/unsharp masking gaussian blur of 5 gray.jpeg')

#GAUSSIAN BLUR AND UNSHARP MASKING WITH GAMMA 10
Image.fromarray(im4).convert('RGB').save('Lab1/gaussian blur of 10.jpeg')
temp = unsharpen(im, im4)
Image.fromarray(temp).convert('RGB').save('Lab1/unsharp masking gaussian blur of 10.jpeg')
temp = unsharpen(imGray, im4Gray)
Image.fromarray(temp).convert('RGB').save('Lab1/unsharp masking gaussian blur of 10 gray.jpeg')

#QUOTIENT IMAGES FROM BLURS OF GAMMA 2,5,10 RESPECTIVELY
im2Quotient = 100*(imGray/im2Gray)
Image.fromarray(im2Quotient).convert('RGB').save('Lab1/Quotient gaussian blur of 2 gray.jpeg')
im3Quotient = 100*(imGray/im3Gray)
Image.fromarray(im3Quotient).convert('RGB').save('Lab1/Quotient gaussian blur of 5 gray.jpeg')
im4Quotient = 100*(imGray/im4Gray)
Image.fromarray(im4Quotient).convert('RGB').save('Lab1/Quotient gaussian blur of 10 gray.jpeg')

#HISTOGRAM EQUALIZATIONS ON QUOTIENT IMAGES AND OTHER SAMPLE IMAGES
im2H = getHisteq(im2Quotient)
Image.fromarray(im2H).convert('RGB').save('Lab1/Histeq on Quotient gaussian blur of 2 gray.png')
im3H = getHisteq(im3Quotient)
Image.fromarray(im3H).convert('RGB').save('Lab1/Histeq on Quotient gaussian blur of 5 gray.png')
im4H = getHisteq(im4Quotient)
Image.fromarray(im4H).convert('RGB').save('Lab1/Histeq on Quotient gaussian blur of 10 gray.png')
temp = getHisteq(array(Image.open('dark.jpg').convert('L')))
temp = array(temp, 'uint8')
Image.fromarray(temp).convert('RGB').save('Lab1/Histeq on dark image sample.png')
temp = getHisteq(array(Image.open('bright.jpg').convert('L')))
temp = array(temp, 'uint8')
Image.fromarray(temp).convert('RGB').save('Lab1/Histeq on bright image sample.png')

#RUDIN OSHER FATEMI ON BLURRED WITH GAMMA 2,5,10 AND OTHER SAMPLE IMAGES (SALT AND PEPPER NOISE)
temp = rof.denoise(im2Gray, im2Gray)
Image.fromarray(temp).convert('RGB').save('Lab1/rof on gamma blur 2.png')
temp = rof.denoise(im3Gray, im3Gray)
Image.fromarray(temp).convert('RGB').save('Lab1/rof on gamma blur 5.png')
temp = rof.denoise(im4Gray, im4Gray)
Image.fromarray(temp).convert('RGB').save('Lab1/rof on gamma blur 10.png')
salted = array(Image.open('salted.jpg').convert('L'))
temp = rof.denoise(salted, salted)
Image.fromarray(temp).convert('RGB').save('Lab1/rof on salt noise.png')
peppered = array(Image.open('peppered.jpg').convert('L'))
temp = rof.denoise(peppered, peppered)
Image.fromarray(temp).convert('RGB').save('Lab1/rof on pepper noise.png')