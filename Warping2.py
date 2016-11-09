import cv2
import numpy
from PIL import Image
from matplotlib.pyplot import *
from numpy import *

def realValueIndex(im, realX, realY):
    x = int(realX)
    y = int(realY)
    rHalf2 = realY - y
    rHalf1 = 1 - rHalf2
    cHalf2 = realX - x
    cHalf1 = 1 - cHalf2

    pixel = zeros(3)
    if(y+1 == len(im)):
        y-=1
    if(x+1 == len(im[0])):
        x-=1
    for i in range(3):
        pixel[i] += (rHalf1 * cHalf1) * im[y][x][i]
        pixel[i] += (rHalf2 * cHalf1) * im[y + 1][x][i]
        pixel[i] += (rHalf1 * cHalf2) * im[y][x + 1][i]
        pixel[i] += (rHalf2 * cHalf2) * im[y + 1][x + 1][i]

    return pixel

def getDistance(xLen, yLen):
    result = sqrt(xLen*xLen+yLen*yLen)
    return result

def getDistanceCoo(x1, y1, x2, y2):
    x = x1-x2
    y = y1-y2
    result = sqrt(x*x+y*y)
    return result

# initialize the list of reference points
refPt = []
temp = []
done = 0
def click_and_crop(event, x, y, flags, param):
    global refPt, done, temp

    # if the left mouse button is clicked, record (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 3, (124,252,0), thickness=2)
        cv2.imshow("image", image)
        temp.append((x,y))
        #refPt.append((x,y))
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)
        cv2.imshow("image", image)
        temp.append((x,y))
        #refPt.append((x,y))
        refPt.append(temp)
        temp = []
    if event == cv2.EVENT_RBUTTONDOWN:
        done = 1
        if(len(temp)%2 == 0):
            print("Selected points accepted, running algorithm\n")
        else:
            print("\n***ERROR: Odd number of points selected***\n")

##############################################################################################################
# load the image, clone it, and setup the mouse callback function
print("Choose points to morph\nOnly even number of points will be accepted.\nPress 'r' to reset points.\n\nLeft button down to select starting points, left button up to select ending points.\nPress Right mouse button down to run algorithm\n")
image = cv2.imread("warpingPic.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the selected points
    if key == ord("r"):
        image = clone.copy()
        refPt = []
    if done == 1:
        break

cv2.destroyAllWindows()
##############################################################################################################

#append anchors to target function
cols = len(image[0])
rows = len(image)
e = 0.000000000000000000000000000000000000000001
refPt.append([(0,0),(0,0)])
refPt.append([(cols,0), (cols,0)])
refPt.append([(0,rows), (0,rows)])
refPt.append([(cols,rows), (cols, rows)])
cooDist = []
distances = zeros(len(refPt))
distancesX = zeros(len(refPt))
distancesY = zeros(len(refPt))
for i in range(len(refPt)):
    cooDist.append((refPt[i][0][0]-refPt[i][1][0],refPt[i][0][1]-refPt[i][1][1]))
    distances[i] = getDistance(cooDist[i][0], cooDist[i][1])
    distancesX[i] = cooDist[i][0]
    distancesY[i] = cooDist[i][1]
allControlDists = zeros((rows, cols, len(distances)))
controlXs = zeros(len(distances))
controlYs = zeros(len(distances))
for i in range(len(distances)):
    controlXs[i] = refPt[i][1][0]
    controlYs[i] = refPt[i][1][1]

morphed = zeros(image.shape)
save = image
image = array(clone, 'uint8')

for r in range(rows):
    sys.stdout.write('\r'+str(int(r/(rows-1)*100))+'%')
    sys.stdout.flush()
    for c in range(cols):
        allControlDists[r][c][:] = (1/(getDistanceCoo(c,r,controlXs[:],controlYs[:])+e))
        weight = zeros(len(distances))
        denominator = sum(allControlDists[r][c])
        weight[:] = (allControlDists[r][c][:])/denominator
        newX = sum(distancesX[:]*weight[:])
        newY = sum(distancesY[:]*weight[:])

        if(r+newY >= rows):
            newY = -r
        if(c+newX >= cols):
            newX = -c
        newX+=c
        newY+=r
        morphed[r][c] = realValueIndex(image, newX, newY)


morphed = array(morphed, 'uint8')
cv2.imshow('save', image)
cv2.imshow('Warped image', morphed)
cv2.imwrite('warped1.jpg', morphed)
cv2.imwrite('warped1ORIG.jpg', save)
cv2.waitKey(0)
cv2.destroyAllWindows()

