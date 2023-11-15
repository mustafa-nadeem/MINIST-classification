#importing libraries
import logging
import math
import matplotlib.pyplot as plt

#importing data
from data_utils import *

#loading images and labels into variables via numpy, from the data_utils.py module 
trainImages = load_images("train-images-idx3-ubyte.gz")
trainLabels = load_labels("train-labels-idx1-ubyte.gz")
testImages = load_images("t10k-images-idx3-ubyte.gz")
testLabels = load_labels("t10k-labels-idx1-ubyte.gz")

logging.critical(testImages[0])

class NeuralNetwork():
    def __init__(self):
        #generate weights : 10 numbers, 784 pixels/neurons
        self.theta1 = np.random.rand(10, 784)
        self.theta2 = np.random.rand(10, 784)
        #generate biases    
        self.bias1 = np.random.rand(10, 1)
        self.bias2 = np.random.rand(10, 1)


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def ReLU(self, x):
        return np.maximum(0, x)
    
    
    def forwardPropoagation(self, imageTrain):
        #first layer - matrix dot multiplication of weights with input layer + bias
        z1 = imageTrain.dot(self.theta1) + self.bias1
        #feeding z1 into activation function for non-linearity
        a1 = self.sigmoid(z1)

        #second layer - dot multiplication of weights with output of first layer + bias
        z2 = a1.dot(self.theta2) + self.bias2
        #feeding z2 into activation function again
        a2 = self.sigmoid(z2)

        return a1, z1, a2, z2
    
    def backwardPropagation():
        ...
    
    '''
    def fit(self, imageTrain):

        continueTraining = True

        while continueTraining:

            
            # input layer  

            # forward propagation

            # backward propagation:
            #     reverse process through derivatives?
            #         derivative of 2nd layer, then 1st layer?
            #     find error
            
            # update parameters and continue loop

            # loop ends if no significant change in accuracy 
            
    '''


#nn = NeuralNetwork()
#nn.forwardPropoagation()