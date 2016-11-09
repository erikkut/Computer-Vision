from matplotlib.pyplot import *

def get(im):
    #get gradience x & y vectors
    X, Y = np.gradient(np.array(im, dtype=np.float))
    #turn into gradience magnitude 2-d array
    magnitude = np.sqrt(X**2 + Y**2)
    return magnitude
