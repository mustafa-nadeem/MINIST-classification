#importing libraries
import logging
import math
import matplotlib.pyplot as plt

#importing data
from data_utils import *


#loading images and labels into variables via numpy, from the data_utils.py module 
trainingImages = np.array(load_images("train-images-idx3-ubyte.gz"))
trainingLabels = np.array(load_labels("train-labels-idx1-ubyte.gz"))
testImages = np.array(load_images("t10k-images-idx3-ubyte.gz"))
testLabels = np.array(load_labels("t10k-labels-idx1-ubyte.gz"))


#logging.critical((testLabels[0]))
print("training images: ", trainingImages.shape)
print("training labels: ", trainingLabels.shape)



'''
trainImages is 60,000 images/elements,
with each element itself being an array of length 784 (all pixels of a 28x28 image),
with each element in that array being a number 0 - 255 representing pixel brightness

'''


# For trainImages to be dot multiplied, it needs to be turned into a numpy array
# trainImgTest = np.array(trainImages)
# but then also needs to be transposed because (10, 784) isn't compatible with (60000, 784)
# (10, 784) is compatible with (784, 60000)




class NeuralNetwork():
    def __init__(self):

        #generate weights
        self.w1 = np.random.rand(784, 16)
        self.w2 = np.random.rand(16, 10)
        #generate biases    
        self.bias1 = np.random.rand(1, 16)
        self.bias2 = np.random.rand(1, 10)

        # #generate weights
        # self.w1 = np.random.rand(16, 784)
        # self.w2 = np.random.rand(10, 16)
        # #generate biases    
        # self.bias1 = np.random.rand(16, 1)
        # self.bias2 = np.random.rand(10, 1)


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def ReLU(self, x):
        return np.maximum(0, x)
    
    def ReLUDerivative(self, x):
        ...
    
    def activation(self, x, af):
        if af == "1":
            #Sigmoid
            return self.sigmoid(x)
        elif af == "2":
            #ReLU
            return self.ReLU(x)
        elif af == "3":
            #Softmax - placeholder for now
            return self.ReLU(x) 
    
    def activationDerivatiive(self, x, af):
        if af == "1":
            #Sigmoid derivative
            return self.sigmoidDerivative(x)
        elif af == "2":
            #ReLU
            return self.ReLU(x)
        elif af == "3":
            #Softmax - placeholder for now
            return self.ReLU(x) 

    def oneHotEncode(self, label):
        '''
        create new numpy array with size equal to 60000, 10
        for every array in numpy array, traverse and change to 1 for corresponding num
        in label
        '''
        oneHotEncoded = np.zeros((60000, 10))
        for x, label in enumerate(label):
            oneHotEncoded[x, label] = 1
        
        return oneHotEncoded

        
    def loss(self, labels, output):

        #Mean Squared Error
        loss = 1 / len(output) * np.sum((output - labels) ** 2, axis = 0)

        return loss
        
    
    def fitOld(self, lr, epochs, trainImg, trainLabels, activationFunc):

        oneHotLabels = self.oneHotEncode(trainLabels)
        print("one-hot encoded labels shape: ", oneHotLabels.shape)
        
        for epoch in range (0, epochs):
            for img, l in zip(trainImg, oneHotLabels):
                '''Forward Propagation'''

                img.shape += (1,)
                l.shape += (1,)

                #input layer to hidden layer - matrix dot multiplication of weights with input layer + bias
                z1 = self.w1.dot(img) + self.bias1
                print("z1 shape ", z1.shape)
                print("z1: ", z1)
                #feeding z1 into activation function for non-linearity
                a1 = self.activation(z1, activationFunc)
                print("a1 shape ", a1.shape)
                print("a1: ", a1)
                
                #hidden layer to output layer - dot multiplication of weights with output of first layer + bias
                z2 = self.w2.dot(a1) + self.bias2
                print("z2 shape ", z2.shape)
                print("z2 ", z2)
                #feeding z2 into activation function again
                a2 = self.activation(z2, activationFunc)
                print("a2 shape ", a2.shape)
                print("a2 ", a2)

                '''Cost / Loss'''

                loss = self.loss(l, a2.T)

                '''Back Propagation'''

                da2 = (l - a2) * (a2 * (1 - a2))
                da1 = da2.T.dot(self.w2) * (a1 * (1 - a1))
                self.w2 -= lr * a1.dot(da2.T)
                self.w1 -= lr * trainImg.T.dot(da1)

            # _______________________________________________

            # '''Back Propagation'''

            # #Derivative loss (Mean Squared Error) = labels - a2
            # #Derivative activation function (sigmoid) = a2 * (1 - a2)
            # #da2 will be reused to find effect of w1 and w2 on output because it's common among both derivatives 
            # da2 = (oneHotLabels.T - a2) * (a2 * (1 - a2))
            # print("da2 shape: ", da2.shape)

            # #da2 is part of the chain rule multiplication of derivatives for da1
            # #Derivative of z2 is w2
            # #Derivative of activation function, a1 (sigmoid) =  a1 * (1 - a1)
            # print("w2 shape: ", self.w2.shape)
            # da1 = da2.T.dot(self.w2).T * (a1 * (1 - a1))
            # #da1 = da2.T.dot(self.w2) * (a1 * (1 - a1))
            # print("da1 shape: ", da1.shape)

            # ''' 
            # a2:  (5, 1) (10, 60000)
            # a2_delta:  (5, 1) (10, 60000)
            # a1_delta:  (5, 4) (10, 10)
            # w2 updated:  (4, 1) (10, 10) [784, 10]
            # w1 updated:  (3, 4) (10, 784)
            # layer structure: 3 4 1 | 784 10 10
            # '''

            # #updating weights
            # #dz2/dw2 = a1
            # #dz1/dw1 = trainImg
            # self.w2 -= lr * a1.T.dot(da2)
            # self.w1 -= lr * trainImg.T.dot(da1)
                
    def fit (self, lr, epochs, trainImg, trainLabels, activationFunc):

        oneHotLabels = self.oneHotEncode(trainLabels)
        print("one-hot encoded labels shape: ", oneHotLabels.shape)

        
        for epoch in range (0, epochs):
            ''' Forward prop '''
            #input layer to hidden layer - matrix dot multiplication of weights with input layer + bias
            z1 = np.dot(trainImg, self.w1) + self.bias1
            #feeding z1 into activation function for non-linearity
            a1 = self.activation(z1, activationFunc)
            print("a1 shape: ", a1.shape)

            #hidden layer to output layer - dot multiplication of weights with output of first layer + bias
            z2 = np.dot(a1, self.w2) + self.bias2
            #feeding z2 into activation function again
            a2 = self.activation(z2, activationFunc)
            print("a2 shape: ", a2.shape)

            ''' Back prop'''
            #oneHotLabels - a2 is derivative of loss function
            #a2 * (1 - a2) is derivative of sigmoid
            #derivative a2 = derivative loss * derivative sigmoid
            da2 = (oneHotLabels - a2) * (a2 * (1 - a2))
            print("da2: ", da2.shape)

            #derivative a1 = derivative a2 * derivative z2 * derivative a1
            da1 = da2.dot(self.w2.T) * (a1 * (1 - a1))

            #updating weights
            
            #effect of w2 on loss = derivative loss * derivative sigmoid * derivative z2
            #da2 = derivative loss * derivative sigmoid
            self.w2 -= lr * a1.T.dot(da2)
            print("w2 updated: ", self.w2.shape)

            #effect of w1 on loss = derivative a1 * derivative z1
            #derivative z1 = trainImg
            self.w1 -= lr * trainImg.T.dot(da1)
            print("w1 updated: ", self.w1.shape)


            


nn = NeuralNetwork()
activationChoice = "1" #input("Choose an activation function\n 1 - Sigmoid\n 2 - ReLU\n 3 - Softmax\n")
learningRate = 0.01 #input("Enter a learning rate")
epochs = 1 #input("Enter number of epochs")
nn.fit(learningRate, epochs, trainingImages, trainingLabels, activationChoice)
print("done")

'''
Trying to match with example Lab solution, trying to do all images at once instead of loop method
x:  (5, 3) | trainingImages is (60000, 784) 
y:  (1, 5) [transpose to (5, 1)] | trainingLabels is (60000,) [converts to (60000, 10)]
w1: (3, 4) | (784, 16) 
z1: (5, 3) x (3, 4) = (5, 4) | (60000, 784) x (784, 16) = (60000, 16)
w2: (4, 1) | (16, 10)
z2: (5, 4) x (4, 1) = (5, 1) | (60000, 16) x (16, 10) = (60000, 10) 

backprop:
da2: (5, 1) | (60000, 10)
da1: (5, 4) | (60000, 16)
'''