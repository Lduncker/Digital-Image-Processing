import numpy as np
import matplotlib.pyplot as plt

IM_WIDTH = 1024
IM_HEIGHT = 737
NUM_INPUT = IM_WIDTH * IM_HEIGHT
NUM_HIDDEN = 20
NUM_OUTPUT = IM_WIDTH * IM_HEIGHT

def relu (z):
    return np.maximum(0, z)

def reluprime(z):
    z[z > 0] = 1
    z[z <= 0] = 0
    return z
    
def forward_prop (x, y, W1, b1, W2, b2):
    z = W1 @ x + np.reshape(b1, (b1.shape[0],1))
    h = relu(z)
    yhat = (W2 @ h + b2).squeeze()
    
    loss = 1/2 * np.mean((y-yhat) ** 2)
    #can normalize here if desired
    return loss, x, z, h, yhat
   
def back_prop (X, y, W1, b1, W2, b2):
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    
    g = ((np.atleast_2d(yhat - y).T @ W2) * reluprime(z.T)).T
    
    gradW1 = g @ x.T + 1e-4 * W1
    gradb1 = np.mean(g, axis=1)
    gradW2 = np.atleast_2d(yhat - y) @ h.T + 1e-4 * W2
    gradb2 = np.mean(yhat - y)
    
    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-6, batchSize = 32, numEpochs = 500):
    #shuffle the indices for training, then use them to access training images/labels together
    shuffledIndices = np.random.permutation(trainX.shape[1])
    
    #number of epochs
    for e in range(numEpochs):
        for i in range(int(trainX.shape[1] / batchSize)):
            miniBatch = shuffledIndices[i*batchSize:(i+1)*batchSize]
            
            #perform backward prop to get the gradients needed
            gradW1, gradb1, gradW2, gradb2 = back_prop(trainX[:, miniBatch], trainY[miniBatch], W1, b1, W2, b2)
            
            #update weights and biases
            W1 = W1 - epsilon * gradW1
            W2 = W2 - epsilon * gradW2
            b1 = b1 - epsilon * gradb1
            b2 = b2 - epsilon * gradb2
        
        #Get a look at how accuracy is improving
        loss, a, b, c, d = forward_prop(testX, testY, W1, b1, W2, b2)
        print("loss at epoch ", e)
        print(loss)
    return W1, b1, W2, b2

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)
        
        #data augmentation
        flippedImages = np.fliplr(images)
        np.hstack((images, flippedImages))
        np.hstack((labels, labels))
        
        noise = np.random.normal(0, 1e-2, size=(2304, 5000))
        noisedImages = noise
        np.hstack((images, noisedImages))
        np.hstack((labels, labels))

    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).

    return images - mu, labels, mu

#

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY, mu = loadData("tr")
        testX, testY, _ = loadData("te", mu)

    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = np.mean(trainY)

    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)
    
    show_weight_vectors(W1)
