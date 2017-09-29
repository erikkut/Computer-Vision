import cv2
from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color, exposure, io, img_as_float, transform
import time

def resizeImg(im, percentage):
    width = int(len(im) * percentage / 100)
    height = int(len(im[0]) * percentage / 100)
    shape = (width, height)

    temp = transform.resize(im, shape)

    return temp

def find(region, im, pixelsPerCell, stepSize):
    #pixelsPerCell: does not need many to calculate accurately. Can go up to (32,32)
    #stepSize: needs large number in order to calculate accurately (minimum is like 16)

    #DECREASE IMAGE TO SEARCH FOR BY 5% 10 TIMES AN SEARCH FOR EACH ONE (10th one is original)
    resized = []
    sizes = []
    stepSizes = []
    for i in range(55,100,5):
        width = int(len(im) * i / 100)
        height = int(len(im[0]) * i / 100)
        shape = (width, height)
        temp = transform.resize(region, shape)
        step = (width/stepSize, height/stepSize)

        resized.append(temp)
        sizes.append(shape)
        stepSizes.append(step)
    resized.append(region)
    sizes.append((len(region), len(region[0])))

    # for i in range(0,10):
    #     plt.imshow(resized[i], cmap=plt.cm.gray)
    #     plt.show()

    # [0] is width
    widIm = len(im)
    lenIm = len(im[0])

    histToCompareTo = histTemplate
    smallestHist = float('inf')
    smallestX = 0;
    smallestY = 0;

    for x in range(0, widIm-sizes[len(sizes)][0], stepX):
        for y in range(0, lenIm-sizes[len(sizes)][1], stepY):
            window = im[x:x+widR, y:y+lenR]
            window = img_as_float(window)

            # bleh = copy(im)
            # cv2.rectangle(bleh, (y,x), (y+lenR,x+widR), (0, 255, 0), 2)
            # plt.imshow(bleh, cmap=plt.cm.gray)
            # plt.show()

            temp = hog(window, orientations=8, pixels_per_cell=pixelsPerCell, cells_per_block=(1, 1))

            difference = temp-histToCompareTo#substract histograms
            difference = square(difference)
            # difference = abs(difference)
            tempSum = sum(difference)#compute cumulative sum of histogram to compare to previous

            if(tempSum < smallestHist):
                smallestHist = tempSum
                smallestX = x
                smallestY = y

    coordinates = [(smallestY, smallestX), (smallestY + lenR, smallestX + widR)]
    return coordinates

pixelsPerCell = (8,8)
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
coo = find(eyeTemplate, frame ,pixelsPerCell, stepSize)
print("Time to calculate one frame: %s ms" % (time.time() - start_time))

original = io.imread('Final Project/test1.jpg')
cv2.rectangle(original, coo[0], coo[1], (0, 255, 0), 2)
plt.imshow(original)
plt.show()
