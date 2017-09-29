import cv2
from numpy import *
import time


def realValueIndex(im, realX, realY):
    x = int(realX)
    y = int(realY)
    rHalf2 = realY - y
    rHalf1 = 1 - rHalf2
    cHalf2 = realX - x
    cHalf1 = 1 - cHalf2

    pixel = zeros(3)
    if (y + 1 == len(im)):
        y -= 1
    if (x + 1 == len(im[0])):
        x -= 1
    for i in range(3):
        pixel[i] += (rHalf1 * cHalf1) * im[y][x][i]
        pixel[i] += (rHalf2 * cHalf1) * im[y + 1][x][i]
        pixel[i] += (rHalf1 * cHalf2) * im[y][x + 1][i]
        pixel[i] += (rHalf2 * cHalf2) * im[y + 1][x + 1][i]

    return pixel


def getDistance(xLen, yLen):
    result = sqrt(xLen * xLen + yLen * yLen)
    return result


def getDistanceCoo(x1, y1, x2, y2):
    x = x1 - x2
    y = y1 - y2
    result = sqrt(x * x + y * y)
    return result


def morph(image, refPt):
    # append anchors to target function
    cols = len(image[0])
    rows = len(image)
    e = 0.000000000000000000000000000000000000000001
    refPt.append([(0, 0), (0, 0)])
    refPt.append([(cols, 0), (cols, 0)])
    refPt.append([(0, rows), (0, rows)])
    refPt.append([(cols, rows), (cols, rows)])
    cooDist = []
    distances = zeros(len(refPt))
    distancesX = zeros(len(refPt))
    distancesY = zeros(len(refPt))
    for i in range(len(refPt)):
        cooDist.append((refPt[i][0][0] - refPt[i][1][0], refPt[i][0][1] - refPt[i][1][1]))
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
    image = array(image, 'uint8')

    for r in range(rows):
        for c in range(cols):
            allControlDists[r][c][:] = (1 / (getDistanceCoo(c, r, controlXs[:], controlYs[:]) + e))
            weight = zeros(len(distances))
            denominator = sum(allControlDists[r][c])
            weight[:] = (allControlDists[r][c][:]) / denominator
            newX = sum(distancesX[:] * weight[:])
            newY = sum(distancesY[:] * weight[:])

            if (r + newY >= rows):
                newY = -r
            if (c + newX >= cols):
                newX = -c
            newX += c
            newY += r
            morphed[r][c] = realValueIndex(image, newX, newY)

    return morphed


def findFace(im):
    start_time = time.time()

    # CONVERT TO GRAYSCALE
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    save = im.copy()

    # GET FACE FEATURE COORDINATES FOR ALL FACES
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # DRAW RECTANGLES AROUND FOUND FEATURES
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        w1 = int(round(w * .1))
        h1 = int(round(h * .25))
        # cv2.rectangle(im, (x+w1, y+h1), (x + w-w1, y + h-h1), (255, 0, 0), 2)

        oldTL = (x, y)
        oldTR = (x + w, y)
        oldBL = (x, y + h)
        oldBR = (x + w, y + h)

        newX = x + w1
        newY = y + h1
        newX2 = x + w - w1
        newY2 = y + h - h1

        newTL = (newX, newY)
        newTR = (newX2, newY)
        newBL = (newX, newY2)
        newBR = (newX2, newY2)

        morphPoints = []
        # morphPoints.append([newTL, oldTL])
        # morphPoints.append([newTR, oldTR])
        # morphPoints.append([newBL, oldBL])
        # morphPoints.append([newBR, oldBR])
        morphPoints.append([oldTL, newTL])
        morphPoints.append([oldTR, newTR])
        morphPoints.append([oldBL, newBL])
        morphPoints.append([oldBR, newBR])

        cv2.circle(im, (int(round(x + w / 2)), int(round(y + h / 2))), 3, (0, 255, 0), 2)
        save = morph(save, morphPoints)

    # leftEye = (int(round(eyes[0][0]+eyes[0][2]/2)), int(round(eyes[0][1]+eyes[0][3]/2)))
    # cv2.circle(im, leftEye, 3, (124, 252, 0), thickness=2)
    #
    # rightEye = (int(round(eyes[1][0]+eyes[1][2]/2)), int(round(eyes[1][1]+eyes[1][3]/2)))

    cv2.imshow('highlighted', im)
    cv2.waitKey(1)

    print("Time to calculate one frame: %s ms" % (time.time() - start_time))

    save = array(save, 'uint8')
    # im = array(im, 'uint8')
    # cv2.imshow("Morphed", save)
    # cv2.imshow("Faces found", im)
    # cv2.waitKey(0)

    return save


    # cap = cv2.VideoCapture('resizedTestVideo.mp4')
    # codec = cv2.VideoWriter_fourcc(* 'XVID')
    # out = cv2.VideoWriter('try2.avi', codec, 24, (300,170))
    #
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #
    #     if(ret):
    #         frame = findFace(frame)
    #         # cv2.imshow('frame', frame)
    #         # cv2.waitKey()
    #         out.write(frame)
    #     else:
    #         break
    #
    #     # cv2.imshow('frame',gray)
    #     # cv2.waitKey()
    #
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    # #CREATE HAAR CASCADE
    # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #
    # cap = cv2.VideoCapture('resizedTestVideo.mp4')
    # # codec = cv2.VideoWriter_fourcc(* 'XVID')
    # # out = cv2.VideoWriter('highlighted.avi', codec, 24, (300,300))
    #
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #
    #     if(ret):
    #         # CONVERT TO GRAYSCALE
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #         # GET FACE FEATURE COORDINATES FOR ALL FACES
    #         faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #         eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #
    #         # DRAW RECTANGLES AROUND FOUND FEATURES
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #         for (x, y, w, h) in eyes:
    #             cv2.circle(frame, (int(round(x + w / 2)), int(round(y + h / 2))), 3, (0, 255, 0), 2)
    #         # out.write(frame)
    #     else:
    #         break
    #
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey()
    #
    # cap.release()
    # # out.release()
    # cv2.destroyAllWindows()


cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture('resizedTestVideo.mp4')  # 'resizedTestVideo.mp4'
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('highlighted.avi', codec, 24, (300, 170))

z = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if (ret==False):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(frame, (x,y), (x+w, int(round(y+h/2))), (0, 255, 0), 2)

    window = zeros(frame.shape)
    if (len(faces) > 0):
        x1 = faces[0][0]
        y1 = faces[0][1]
        window = gray[y1:int(round(y1 + faces[0][3] / 2)), x1:x1 + faces[0][2]]
    window = array(window, 'uint8')
    eyes = eyeCascade.detectMultiScale(window, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(eyes)

    for (x, y, w, h) in eyes:
        x += x1
        y += y1
        cv2.circle(frame, (int(round(x + w / 2)), int(round(y + h / 2))), 3, (0, 255, 0), 2)

    save = zeros(frame.shape)
    save = array(frame, 'uint8')
    for (x, y, w, h) in eyes:
        x += x1
        y += y1
        w1 = int(round(w * .1))
        h1 = int(round(h * .25))

        oldTL = (x, y)
        oldTR = (x + w, y)
        oldBL = (x, y + h)
        oldBR = (x + w, y + h)

        newX = x + w1
        newY = y + h1
        newX2 = x + w - w1
        newY2 = y + h - h1

        newTL = (newX, newY)
        newTR = (newX2, newY)
        newBL = (newX, newY2)
        newBR = (newX2, newY2)

        morphPoints = []
        # morphPoints.append([newTL, oldTL])
        # morphPoints.append([newTR, oldTR])
        # morphPoints.append([newBL, oldBL])
        # morphPoints.append([newBR, oldBR])
        morphPoints.append([oldTL, newTL])
        morphPoints.append([oldTR, newTR])
        morphPoints.append([oldBL, newBL])
        morphPoints.append([oldBR, newBR])

        save = morph(save, morphPoints)
        save = array(save, 'uint8')

    # Display the resulting frame
    # cv2.imshow('Video', frame)  # window
    out.write(save)
    print(z)
    # cv2.imshow('frame', gray)
    # cv2.waitKey()
    z += 1

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()