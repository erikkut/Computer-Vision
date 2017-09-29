import cv2
from numpy import *
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, io, img_as_float, transform
import time

#https://facedetection.com/datasets/

def findDeep(region, im, histTemplate):
    # INITIALIZE VARIABLES
    smallest = zeros(10)
    info = []

    z = 0
    for i in range(55,145,10):
        width = int(len(im) * i / 100)
        height = int(len(im[0]) * i / 100)
        shape = (width, height)
        temp1, temp2 = find(region, transform.resize(region, shape), pixelsPerCell, stepSize)
        smallest[z] = temp1
        info.append(temp2)
        z += 1

    print(smallest.min())
    resultIdx = smallest.argmin()-1
    print(info[resultIdx])
    return info[resultIdx]

def find(region, im, pixelsPerCell, stepSize):
    #pixelsPerCell: does not need many to calculate accurately. Can go up to (32,32)
    #stepSize: needs large number in order to calculate accurately (minimum is like 16)
    widR = len(region)
    widIm = len(im)
    stepX = int(widR/stepSize)

    lenR = len(region[0])
    lenIm = len(im[0])
    stepY = int(lenR/stepSize)

    histToCompareTo = histTemplate
    smallestHist = float('inf')
    smallest = []

    for x in range(0, widIm-widR, stepX):
        for y in range(0, lenIm-lenR, stepY):
            window = im[x:x+widR, y:y+lenR]
            window = img_as_float(window)

            temp = hog(window, orientations=8, pixels_per_cell=pixelsPerCell, cells_per_block=(1, 1))

            difference = temp-histToCompareTo#substract histograms
            difference = square(difference)
            # difference = abs(difference)
            tempSum = sum(difference)#compute cumulative sum of histogram to compare to previous

            if(tempSum < smallestHist):
                smallestHist = tempSum
                smallest = [(y, x), (y+lenR, x+widR)]

    return smallestHist, smallest

# INITIALIZE VARIABLES
pixelsPerCell = (8, 8)
stepSize = 16

#BRING IN EYE TEMPLATE
eyeTemplate = io.imread('Final Project/average male eyes gray-float.jpg')
eyeTemplate = img_as_float(eyeTemplate)

#COMPUTE ITS HISTOGRAM, NEEDS TO BE COMPUTED ONLY ONCE
histTemplate = hog(eyeTemplate, orientations=8, pixels_per_cell=pixelsPerCell, cells_per_block=(1, 1))

# fd, hog_image = hog(eyeTemplate, orientations=8, pixels_per_cell=pixelsPerCell, cells_per_block=(1, 1), visualise = True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_adjustable('box-forced')
#
# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# ax1.set_adjustable('box-forced')
# plt.show()

# frame = io.imread('Final Project/average male face.jpg')
frame = io.imread('Final Project/test1.jpg')
frame = color.rgb2gray(frame)
frame = img_as_float(frame)

start_time = time.time()
coo = findDeep(eyeTemplate, frame, histTemplate)
print("Time to calculate one frame: %s ms" % (time.time() - start_time))

original = io.imread('Final Project/test1.jpg')
cv2.rectangle(original, coo[0], coo[1], (0, 255, 0), 2)
plt.imshow(original)
plt.show()
