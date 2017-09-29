from PIL import ImageDraw
from exercise4 import HOF, roundTo
from PIL import Image
from matplotlib.pyplot import *
from numpy import *
from scipy.ndimage import *
import matplotlib.pyplot as plt
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

def drawSquare(im, coordinates):
    mid = 0.5
    shape = (coordinates[3]-coordinates[1], coordinates[2]-coordinates[0])
    formatRegion = zeros(shape)
    formatRegion[:,:] = im[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
    imshow(formatRegion)
    show()

    print("starting x {0}, ending {1}".format(coordinates[0], coordinates[2]))
    print("starting y {0}, ending {1}".format(coordinates[1], coordinates[3]))
    idxLarge = formatRegion[:,:] > mid
    formatRegion[idxLarge] -= mid
    idxSmall = formatRegion[:,:] < mid
    formatRegion[idxSmall] += mid

    #im[coordinates[0]:coordinates[2], coordinates[1]:coordinates[3]] = formatRegion[:,:]
    im[coordinates[0]:coordinates[2], coordinates[1]] = formatRegion[0, :]
    im[coordinates[0]:coordinates[2], coordinates[3]] = formatRegion[shape[1]-1, :]
    im[coordinates[0], coordinates[1]:coordinates[3]] = formatRegion[:, 0]
    im[coordinates[2], coordinates[1]:coordinates[3]] = formatRegion[: shape[1]-1]

    return im

def gray2rgb(im):
    w, h = im.shape
    temp = np.empty((w, h, 3), dtype=np.uint8)
    temp[:, :, 0] = im
    temp[:, :, 1] = im
    temp[:, :, 2] = im
    return temp

def find(region, im):
    angles = 8
    lenR = len(region)
    widR = len(region[0])
    lenIm = len(im)
    widIm = len(im[0])

    histToCompareTo = HOF(region, angles, 0,widR-1, 0,lenR-1)
    smallestHist = sum(histToCompareTo)
    smallestX = 0;
    smallestY = 0;

    for x in range(0, widIm-widR-1):
        for y in range(0, lenIm-lenR-1):
            temp = HOF(im, angles, x,x+widR, y,y+lenR)#get testing cell

            difference = temp-histToCompareTo#substract histograms
            difference = square(difference)
            difference = absolute(difference)
            tempSum = sum(difference)#compute cumulative sum of histogram to compare to previous

            if(tempSum < smallestHist):
                smallestHist = tempSum
                smallestX = x
                smallestY = y

    print(smallestHist)
    print("Histogram of region: {0}".format(histToCompareTo))
    print("Histogram found: {0}".format(HOF(im, angles, smallestX,smallestX+len(region[0]), smallestY, smallestY+len(region))))
    coordinates = (smallestX, smallestY, smallestX+len(region[0]), smallestY+len(region))
    return coordinates

###############################################################################
# load the image, clone it, and setup the mouse callback function
image = cv2.imread("Lab 2 Test Image.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()

    if(size(refPt) > 3):
        break

    # if the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
################################################################################

test = array(Image.open("Lab 2 Test Image.jpg").convert('L'))
region = test[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
coordinates = find(region, test)

coo = [(coordinates[2], coordinates[3]),(coordinates[0], coordinates[1])]##originally 0,1,2,3

figure('region to search for')
imshow(region, cmap = cm.gray)

test = Image.fromarray(test).convert('RGB')
test = array(test)
cv2.rectangle(test, coo[0], coo[1], (0, 255, 0), 2)
cv2.imshow("found area", test)

show()