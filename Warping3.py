import cv2
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
morph1 = cv2.imread("lab2/morphFirst.jpg")
morph2 = cv2.imread("Lab2/morphSecond.jpg")
image = zeros(morph1.shape)
image = cv2.addWeighted( morph1, .5, morph2, .5, 0)

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

#Save the chosen points
save = array(image, 'uint8')
cv2.imwrite('morph_points_chosen.jpg', save)

#morph2 refPt (selected sources become dests and vice versa)
refPt2 = []
for i in range(len(refPt)):
    refPt2.append([(refPt[i][1][0],refPt[i][1][1]),(refPt[i][0][0],refPt[i][0][1])])

#append anchors to target functions, and initialize e
cols = len(image[0])
rows = len(image)
e = 0.000000000000000000000000000000000000000001
refPt.append([(0, 0), (0, 0)])
refPt2.append([(0, 0), (0, 0)])
refPt.append([(cols, 0), (cols, 0)])
refPt2.append([(cols, 0), (cols, 0)])
refPt.append([(0, rows), (0, rows)])
refPt2.append([(0, rows), (0, rows)])
refPt.append([(cols, rows), (cols, rows)])
refPt2.append([(cols, rows), (cols, rows)])

# initialize: allControlDists will be used to store normalized eucledian distances
#            controlXs contains destination x coordinates
#            controlYs contains destination y coordinates
allControlDists = zeros(len(refPt))
controlXs = zeros(len(refPt))
controlYs = zeros(len(refPt))
controlXs2 = zeros(len(refPt))
controlYs2 = zeros(len(refPt))
for i in range(len(refPt)):
    controlXs[i] = refPt[i][1][0]
    controlYs[i] = refPt[i][1][1]
    controlXs2[i] = refPt2[i][1][0]
    controlYs2[i] = refPt2[i][1][1]

#initialized here, filled inside algorithm since constantly changing
distances = zeros(len(refPt))
distancesX = zeros(len(refPt))
distancesY = zeros(len(refPt))
distances2 = zeros(len(refPt))#for morph 2 (different step measures)
distancesX2 = zeros(len(refPt))
distancesY2 = zeros(len(refPt))
weight = zeros(len(distances))
#canvases for resulting images
morphed1 = zeros(image.shape)
morphed2 = zeros(image.shape)

n = 60#frames#frames#[important], first frame will be morph1 alone, and one xtra frame will be added for morph2 alone
step = 1/n#used to calculate stepped distances

for z in range(n+1):
    #CURRENT STEP
    currStep = step*(z+1)
    currStep2 = 1-(z+1)*step

    #initialize: distances contains eucledian distances from source to dest
    #            distancesX contains only distances from source x to dest x
    #            distancesY contains only distances form source y to dest y
    for i in range(len(refPt)):
        tempX = (refPt[i][0][0]-refPt[i][1][0])*currStep
        tempY = (refPt[i][0][1]-refPt[i][1][1])*currStep
        distances[i] = getDistance(tempX, tempY)
        distancesX[i] = tempX
        distancesY[i] = tempY

        tempX2 = (refPt2[i][0][0]-refPt2[i][1][0])*currStep2
        tempY2 = (refPt2[i][0][1]-refPt2[i][1][1])*currStep2
        distances2[i] = getDistance(tempX2, tempY2)
        distancesX2[i] = tempX2
        distancesY2[i] = tempY2

    #start algorithm
    for r in range(rows):
        sys.stdout.write('\r' + str(int(r / (rows - 1) * 100)) + '% pic #' + str(z + 1) + ' out of ' + str(n+1))
        sys.stdout.flush()
        for c in range(cols):
            #morph1
            allControlDists[:] = (1 /(getDistanceCoo(c, r, controlXs[:], controlYs[:])*currStep+e))
            denominator = sum(allControlDists)
            weight[:] = (allControlDists[:]) / denominator
            newX = sum(distancesX[:] * weight[:])
            newY = sum(distancesY[:] * weight[:])

            if (r + newY >= rows):
                newY = -r
            if (c + newX >= cols):
                newX = -c
            newX += c
            newY += r
            morphed1[r][c] = realValueIndex(morph1, newX, newY)

            #morph2
            allControlDists[:] = (1 / (getDistanceCoo(c, r, controlXs2[:], controlYs2[:])*currStep2+e))
            denominator = sum(allControlDists)
            weight[:] = (allControlDists[:]) / denominator
            newX = sum(distancesX2[:] * weight[:])
            newY = sum(distancesY2[:] * weight[:])

            if (r + newY >= rows):
                newY = -r
            if (c + newX >= cols):
                newX = -c
            newX += c
            newY += r
            morphed2[r][c] = realValueIndex(morph2, newX, newY)

    morphed1 = array(morphed1, 'uint8')
    morphed2 = array(morphed2, 'uint8')
    final = cv2.addWeighted(morphed1, 1-(z/n), morphed2, (z/n), 0)
    cv2.imwrite('morph{0}.jpg'.format(z), final)
