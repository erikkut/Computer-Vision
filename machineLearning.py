import cv2
from matplotlib.pyplot import *
from numpy import *
from readLoad import load_mnist
from sklearn.cluster import KMeans

def colorClustering(im, clusters, n):
    image_array = np.reshape(im, (len(im) * len(im[0]), len(im[0][0])))  # convert to 2-d matrix to handle easier
    image_array = array(image_array, 'float64')

    kmean = KMeans(n_clusters=clusters, n_init=n).fit(image_array)
    labels = kmean.predict(image_array)
    clusters = kmean.cluster_centers_

    image = np.zeros(im.shape)
    z = 0
    for i in range(len(im)):
        for j in range(len(im[0])):
            image[i][j] = clusters[labels[z]]
            z += 1

    image = array(image, 'uint8')
    return image

def intensityClustering(trainIms, testIms, clusters, n):
    trainIms = np.reshape(trainIms, (len(trainIms), len(trainIms[0])*len(trainIms[0][0])))#reshape into array the size of number of images with each index contaning a flattened array of the corresponding images
    trainIms = array(trainIms, 'float64')

    testIms = np.reshape(testIms, (len(testIms), len(testIms[0])*len(testIms[0][0])))#reshape testing images in same manner
    testIms = array(testIms, 'float64')

    kmean = KMeans(n_clusters=clusters, n_init=n).fit(testIms)#trainIms
    labels = kmean.predict(testIms)
    predicted = zeros(len(labels))
    clusters = kmean.cluster_centers_

    print(clusters)

    #for z in range(len(labels)):
     #   predicted[z] = clusters[labels[z]]

    print("Predicted Labels:")
    print(predicted)
    return predicted



# im = cv2.imread('Sara.jpg')
# im = array(im)
# clustered1 = colorClustering(im, 5, 10)
# clustered2 = colorClustering(im, 15, 10)
# clustered3 = colorClustering(im, 25, 10)
# cv2.imshow('Original', im)
# cv2.imshow('K-Means 5-color clustering', clustered1)
# cv2.imshow('K-Means 15-color clustering', clustered2)
# cv2.imshow('K-Means 25-color clustering', clustered3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

trainingArrImages, trainingLabels = load_mnist("training", np.arange(10), "C:/Users/Eric/PycharmProjects/Computer Vision/Lab3/")
testingArrImages, testingLabels = load_mnist("testing", np.arange(10), "C:/Users/Eric/PycharmProjects/Computer Vision/Lab3/")
predictedLabels = intensityClustering(trainingArrImages, testingArrImages, 10, 10)
print("Actual Labels:")
testingLabels = np.reshape(testingLabels, (len(testingLabels)))
print(testingLabels)
result = testingLabels-predictedLabels
error = count_nonzero(result)
print("Number of erronous predictions:")
print(error)
print("% Error:")
print(error/len(result))
print("Lenght of testing and length of predicted")
print(len(testingLabels))
print(len(predictedLabels))
#imshow(images.mean(axis=0), cmap=cm.gray)
#show()