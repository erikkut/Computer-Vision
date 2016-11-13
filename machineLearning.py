from matplotlib.pyplot import *
from numpy import *

from featureSetsQuiz import colorHistogram
from readLoad import load_mnist
from sklearn.cluster import KMeans
import time

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

def intensityClustering(clusters, n):
    start_time = time.time()

    #GET MNIST DATASET
    path = "C:/Users/Owner/Documents/GitHub/Computer-Vision/Lab3/"
    trainIms, trainingLabels = load_mnist("training", np.arange(10), path)
    testIms, testingLabels = load_mnist("testing", np.arange(10),path)

    #RESHAPE DATASET TO FEED INTO KMEANS
    trainIms = np.reshape(trainIms, (len(trainIms), len(trainIms[0])*len(trainIms[0][0])))#reshape into array the size of number of images with each index contaning a flattened array of the corresponding images
    trainIms = array(trainIms, 'float64')
    testIms = np.reshape(testIms, (len(testIms), len(testIms[0])*len(testIms[0][0])))#reshape testing images in same manner
    testIms = array(testIms, 'float64')
    testingLabels = reshape(testingLabels, (len(testingLabels)))

    #RUN KMEANS ON TRAINING SET AND PREDICT CLUSTERS ON TESTING SET
    kmean = KMeans(n_clusters=clusters, n_init=n).fit(trainIms)
    predictedLabels = kmean.predict(testIms)

    #FIGURE OUT WHAT CLUSTER BELONGS TO WHAT NUMBER BY USING DIFFERENT ARRAY OPERATIONS
    linkTable = zeros((clusters, 10))
    randomIndex = (random.rand(1000) * 10000).astype('int')
    for i in randomIndex:
        linkTable[predictedLabels[i]][testingLabels[i]] += 1  # Sequence is important, rows most be predicted Labels
    transformTable = zeros(clusters)
    for i in range(len(linkTable)):
        transformTable[i] = argmax(linkTable[i])
    for i in range(len(predictedLabels)):
        predictedLabels[i] = transformTable[predictedLabels[i]]

    #CALCULATE RESULTS AND ERROR
    result = testingLabels - predictedLabels
    error = count_nonzero(result)
    print("Number of erronous predictions: %d / %d" % (error, len(result)))
    print("Percentage error: %f" %(error/len(result)))

    print("kmeans running time: %s minutes" % ((time.time() - start_time)/60))
    return predictedLabels

def unpickle(file):
    import pickle
    f = open(file, 'rb')
    dict = pickle.load(f, encoding = 'latin1')
    f.close()
    return dict

def colorHistogramClustering(clusters, n):
    start_time = time.time()

    #GET CIFAR DATASET
    filePath = "C:/Users/Owner/Documents/GitHub/Computer-Vision/Lab3/cifar-10-batches-py/data_batch_"
    dict_1 = unpickle(filePath+"1")
    dict_2 = unpickle(filePath+"2")

    #RESHAPE THE DATA TO OBTAIN ARRAY CONTAINING 10,000 RGB IMAGES
    batchIms1 = reshape(dict_1['data'], (10000,32,32,3), order='F')
    imsTesting = reshape(dict_2['data'], (10000,32,32,3), order='F')

    #TURN EACH SET OF IMAGES INTO THEIR CORRESPONDING COLOR HISTOGRAM
    bins = 3
    colorHist1 = zeros((10000, 3*bins))
    colorHistTesting = zeros((10000, 3*bins))
    for i in range(10000):
        colorHist1[i] = ndarray.flatten(colorHistogram(batchIms1[i], bins))
        colorHistTesting[i] = ndarray.flatten(colorHistogram(imsTesting[i], bins))

    #FEED HISTOGRAMS TO KMEANS
    kmean = KMeans(n_clusters=clusters, n_init=n).fit(colorHist1)
    predictedLabels = kmean.predict(colorHistTesting)

    #DETERMINE WHAT CLUSTERS POINT TO WHAT CLASSIFIER
    linkTable = zeros((clusters, 10))
    randomIndex = (random.rand(1000) * 10000).astype('int')
    for i in randomIndex:
        linkTable[predictedLabels[i]][dict_1['labels'][i]] += 1  # Sequence is important, rows most be predicted Labels
    transformTable = zeros(clusters)
    for i in range(len(linkTable)):
        transformTable[i] = argmax(linkTable[i])
    for i in range(len(predictedLabels)):
        predictedLabels[i] = transformTable[predictedLabels[i]]

    #CALCULATE RESULTS AND ERROR
    result = dict_1['labels'] - predictedLabels
    error = count_nonzero(result)
    print("Number of erronous predictions: %d / %d" % (error, len(result)))
    print("Percentage error: %f" %(error/len(result)))

    print("kmeans running time: %s minutes" % ((time.time() - start_time)/60))
    return predictedLabels


#predictedLabels = intensityClustering(10, 10)
colorHistogramClustering(10, 10)

#imshow(images.mean(axis=0), cmap=cm.gray)
#show()