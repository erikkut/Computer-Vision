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
    for i in range(3):
        pixel[i] += (rHalf1 * cHalf1) * im[y][x][i]
        pixel[i] += (rHalf2 * cHalf1) * im[y + 1][x][i]
        pixel[i] += (rHalf1 * cHalf2) * im[y][x + 1][i]
        pixel[i] += (rHalf2 * cHalf2) * im[y + 1][x + 1][i]

    return pixel

# initialize the list of reference points
refPt = []
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # if the left mouse button is clicked, record (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x,y), 3, (124,252,0), thickness=2)
        cv2.imshow("image", image)
        refPt.append((x,y))

##############################################################################################################
# load the image, clone it, and setup the mouse callback function
print("Choose points in clockwise order, starting with top-left corner\nWhen 4 points are selected, algorithm will start.\nPress 'r' to reset points.\nPress 'q' to quit.")
image = cv2.imread("homographyPic.jpg")
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
    #if 4 points have been selected, break from the loop
    elif  len(refPt) == 4:
        break
    # if the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
##############################################################################################################

avgX = int(( (refPt[1][0]-refPt[0][0])+(refPt[2][0]-refPt[3][0]) )/2)
avgY = int(( (refPt[3][1]-refPt[0][1])+(refPt[2][1]-refPt[1][1]) )/2)

shape = (avgY, avgX, 3)
canvas = zeros(shape)

m = avgY
n = avgX
im = array(clone)
for r in range(0, m):
    start = add( multiply(r/m, refPt[3]), multiply((m-r)/m, refPt[0]) )
    end = add( multiply(r/m, refPt[2]), multiply((m-r)/m, refPt[1]) )
    for c in range(0, n):
        p = add( multiply(end, c/n), multiply((n-c)/n, start) )
        pixel = realValueIndex(im, p[0], p[1])
        canvas[r][c] = pixel


canvas = array(canvas, 'uint8')

cv2.imshow("Original Image w/ Points taken", image)
cv2.imshow("Fixed Image", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

inTL = [0,0]; inTR = [n-1,0]; inBL = [0, m-1]; inBR = [n-1,m-1]
#inPoints = array([inTL, inTR, inBR, inBL], dtype = "float32")
inPoints = array([inTR, inTL, inBR, inBL], dtype = "float32")

outTL = [refPt[0][0],refPt[0][0]]; outTR = [refPt[1][0], refPt[0][1]]; outBL = [refPt[3][0], refPt[3][1]]; outBR = [refPt[2][0], refPt[2][1]];
#outPoints = array([outTL, outTR, outBR, outBL], dtype = "float32")
outPoints = array([outTR, outTL, outBR, outBL], dtype = "float32")

print("\n\nin points image shape: {0} \nand the in-points:".format(canvas.shape))
print(inPoints)
print("\nout-points:")
print(outPoints)
h = cv2.findHomography(inPoints, outPoints)#OUT TO IN
print("\n homography matrix:")
print(h[0])

cv2.imwrite('homography_square_result.jpg', canvas)

#THIS used the homography matrix to manually calculate the warp effect.
im = zeros(im.shape)
inP = zeros(3)
temp = zeros(3)
inP[2] = 1
arr = array(image)
for r in range(0, m):
    for c in range(0, n):
        inP[0] = c
        inP[1] = r
        temp = np.matmul(h[0], inP)

        y = int(temp[0]/temp[2]+.5)
        x = int(temp[1]/temp[2]+.5)
        im[x][-y] = canvas[r][-c]

cv2.imwrite('homography_skewed_result.jpg', im)

im = array(im, 'uint8')
cv2.imshow("Projected image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()



