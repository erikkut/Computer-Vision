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
    bins = 10
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
        linkTable[predictedLabels[i]][dict_2['labels'][i]] += 1  # Sequence is important, rows most be predicted Labels
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

def gradientHistogram(im, bars):
    # get FILTER [-1,1] and MAGNITUDE of region
    X, Y = np.gradient(np.array(im, dtype=np.float))  # equivalent to [-1,1] filter
    magnitude = np.sqrt(X ** 2 + Y ** 2)

    # (angles of direction per pixel)
    directions = arctan2(Y[:], X[:])

    directions = directions[:][:] * 180 / pi  # also adding 180 to represent every angle in 360 degrees
    idxNeg = directions[:, :] < 0
    directions[idxNeg] += 360

    h = zeros(bars + 1)
    step = 360 / bars

    # FLATTEN arrays to make them easier to use, and divide by step to make the histogram
    directions = ndarray.flatten((rint(directions / step)).astype(int))
    magnitude = ndarray.flatten(magnitude)

    # CREATE histogram of cumulative gradience magnitude per direction
    steps = arange(len(h)) + 1
    for x in range(0, len(h)):
        temp = directions < steps[x]
        h[x] = (sum(magnitude[temp]) - sum(h[0:x]))

    # ADDING first magnitude to last because 0 = 360 in terms of degrees
    h[0] += h[-1]
    # DELETING last index of magnitudes because of above
    h[-1] = 0
    h = h[0:-1]
    # ROUND to nearest int
    h = ndarray.flatten(rint(h).astype(int))

    return h

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gradientHistogramClustering(clusters, n):
    start_time = time.time()

    #GET CIFAR DATASET
    filePath = "C:/Users/Owner/Documents/GitHub/Computer-Vision/Lab3/cifar-10-batches-py/data_batch_"
    dict_1 = unpickle(filePath+"1")
    dict_2 = unpickle(filePath+"2")

    #RESHAPE THE DATA TO OBTAIN ARRAY CONTAINING 10,000 RGB IMAGES
    batchIms1 = reshape(dict_1['data'], (10000,32,32,3), order='F')
    imsTesting = reshape(dict_2['data'], (10000,32,32,3), order='F')

    #TURN EACH SET OF IMAGES INTO THEIR CORRESPONDING HISTOGRAM OF GRADIENTS
    bins = 45;
    gradHistTraining = zeros((10000, bins))
    gradHistTesting = zeros((10000, bins))
    for i in range(10000):
        gradHistTraining[i] = gradientHistogram(rgb2gray(batchIms1[i]), bins)
        gradHistTesting[i] = gradientHistogram(rgb2gray(imsTesting[i]), bins)

    #FEED HISTOGRAMS TO KMEANS
    kmean = KMeans(n_clusters=clusters, n_init=n).fit(gradHistTraining)
    predictedLabels = kmean.predict(gradHistTesting)

    #DETERMINE WHAT CLUSTERS POINT TO WHAT CLASSIFIER
    linkTable = zeros((clusters, 10))
    randomIndex = (random.rand(1000) * 10000).astype('int')
    for i in randomIndex:
        linkTable[predictedLabels[i]][dict_2['labels'][i]] += 1  # Sequence is important, rows most be predicted Labels
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
#colorHistogramClustering(10, 10)
#gradientHistogramClustering(20, 10)

#imshow(images.mean(axis=0), cmap=cm.gray)
#show()