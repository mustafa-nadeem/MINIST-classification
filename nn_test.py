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
'''
trainImages is 60,000 images/elements,
with each element itself being an array of length 784 (all pixels of a 28x28 image),
with each element in that array being a number 0 - 255 representing pixel brightness
For trainImages to be dot multiplied, it needs to be turned into a numpy array
'''


class NeuralNetwork():
    def __init__(self):

        #generate weights
        self.w1 = 2 * np.random.rand(784, 16) - 1 
        self.w2 = 2 * np.random.rand(16, 10) - 1 
        #generate biases    
        self.b1 = 2 * np.random.rand(1, 16) - 1
        self.b2 = 2 * np.random.rand(1, 10) - 1
    
    
    def activation(self, x, af):
        if af == "1":
            #Sigmoid
            return 1 / (1 + np.exp(-x))
        elif af == "2":
            #ReLU
            return np.maximum(0, x)
    
    def activationDerivatiive(self, x, af):
        if af == "1":
            #Sigmoid derivative
            return x * (1.0 - x)
        elif af == "2":
            #ReLU derivative
            return (x >= 0) * 1
    
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def oneHotEncode(self, label):
        '''
        create new numpy array with shape (60000, 10)
        for every array in numpy array, traverse and change to 1 for corresponding num
        in label
        '''
        oneHotEncoded = np.zeros((60000, 10))
        for x, label in enumerate(label):
            oneHotEncoded[x, label] = 1
        
        return oneHotEncoded

        
    def loss(self, label, output):

        #Mean Squared Error
        loss = 1 / len(output) * np.sum((output - label) ** 2, axis = 0)

        return loss

                
    def fit (self, lr, epochs, trainImg, trainLabels, activationFunc):
        
        #required for cost/loss 
        oneHotLabels = self.oneHotEncode(trainLabels)
        print("one-hot encoded labels shape: ", oneHotLabels.shape)

        #for updating biases
        #identityMatrix = np.ones((60000, 1))
        identityMatrix = np.ones((1, 1))

        correct = 0
        
        for epoch in range (0, epochs):
            for img, label in zip(trainImg, oneHotLabels):
                img.shape += (1,)
                label.shape += (1,)

                ''' Forward prop '''
                #input layer to hidden layer - matrix dot multiplication of weights with input layer + bias
                z1 = np.dot(img.T, self.w1) + self.b1
                #feeding z1 into activation function for non-linearity
                a1 = self.activation(z1, activationFunc)
                
                #hidden layer to output layer - dot multiplication of weights with output of first layer + bias
                z2 = np.dot(a1, self.w2) + self.b2
                #feeding into activation function again
                a2 = self.activation(z2, activationFunc)

                #probabilities = np.apply_along_axis(self.softmax, axis = 1, arr = a2)
                #print("probabilities: ", probabilities[0])
                
                
                error = np.apply_along_axis(self.loss, axis = 1, arr = label, output = a2)
                correct += int(np.argmax(a2) == np.argmax(label))

                '''
                The highest probability of softmax is the one it chooses as its guess for what the label corresponds to
                The label is an array 00000100000
                And softmax is an array of floats
                Meaning to produce a guess, based on the calculated probabilities, if you do label - probabilities
                if the answer is > 0, the guess was correct.
                i.e. label = 5 / 0000010000
                    guess = highest probability of array
                    test if guess is correct = label[highestProbability] - guess[highestProbability]
                    if answer > 0, then correct
                '''
                
                ''' Back prop '''
                #delta a2 = dL/da2 * da2/dz2
                #dL/da2 = oneHotLabels - a2 (Mean Squared Error derivative)
                da2 = (a2 - label.T) * self.activationDerivatiive(a2, activationFunc)
                '''
                da2 = (oneHotLabels - results) * derivative softmax * self.activationDerivatiive(a2, activationFunc)
                '''
                
                #delta a1 = delta a2 * dz2/da1 * da1/dz1
                #da1/dz1 = w2
                da1 = da2.dot(self.w2.T) * self.activationDerivatiive(a1, activationFunc)

                #--Updating Weights and Biases--

                #dL/dw2 = da2 * dz2/dw2
                #da2 = dL/da2 * da2/dz2
                #dz2/dw2 = a1
                self.w2 -= lr * a1.T.dot(da2)

                #dL/db2 = da2 * dz2/db2
                #dz2/db2 = 1
                self.b2 -= lr * da2 #np.dot(identityMatrix.T, da2)

                #dL/dw1 = delta a1 * dz1/dw1
                #dz1/dw1 = trainImg
                self.w1 -= lr * img.dot(da1)
                
                #dL/db1 = delta a1 * dz1/db1
                #dz1/db1 = 1
                self.b1 -= lr * da1 #np.dot(identityMatrix.T, da1)
            
            print("epoch: ", epoch)
            print(f"Accuracy: {round((correct / trainImg.shape[0]) * 100, 2)}%")
            correct = 0
        
    def predict(self, x, activationFunc):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.activation(z1, activationFunc)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.activation(z2, activationFunc)
        return a2

'''
#Here is a simpler version of the neural network for testing

# The numbers being fed into sigmoid need to be close to 0, otherwise, the output of the sigmoid function
# will be 1 for every element in the numpy array. This is because sigmoid uses e^x.
# When x is a negative number like -6000, the number produced from e^x becomes too small so the sigmoid function just
# does 1 / 1 = 1 for every value. Sometimes even produces overflow errors
# Question is why do the values become so weird on subsequent loops?

def sigmoidTester(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

nn = NeuralNetwork()
epochs = 50
tr = trainingImages / 255.0
w1 = 2 * np.random.rand(784, 16) - 1
w2 = 2 * np.random.rand(16, 10) - 1
print("w2 ", w2[0])
y = nn.oneHotEncode(trainingLabels) 

for epoch in range (0, epochs):
    #~~forward prop~~
    z1 = np.dot(tr, w1)
    #print("z1 ", z1[0]) #This produces expected numbers for the first loop but subsequent loops look weird
    a1 = sigmoidTester(z1)
    #print("a1 ", a1[0])
    z2 = np.dot(a1, w2)
    #print("z2 ", z2[0])
    a2 = sigmoidTester(z2)
    #print("a2 ", a2[0])
    
    #~~backward prop~~
    da2 = (y - a2) * (a2 * (1.0 - a2))
    #print("da2 ", da2[0])
    da1 = da2.dot(w2.T) * (a1 * (1.0 - a1))
    #print("da1 ", da1[0])
    w2 -= 0.5 * a1.T.dot(da2)
    print("a1.T.dot(da2): ", a1.T.dot(da2)[0])
    print("w2 updated", w2)
    w1 -= 0.5 * tr.T.dot(da1)
    #print("w1 updated", w1[0])
'''


nn = NeuralNetwork()
activationChoice = "1" #input("Choose an activation function\n 1 - Sigmoid\n 2 - ReLU")
learningRate = 0.01 #input("Enter a learning rate")
epochs = 5 #input("Enter number of epochs")

#converts to 0 - 1 range to avoid overflow from activation function
trainingImages = trainingImages / 255 

nn.fit(learningRate, epochs, trainingImages, trainingLabels, activationChoice)
print("training complete")
while True:
    index = int(input("Enter a number between 0 - 59999: "))
    yHat = nn.predict(trainingImages, activationChoice)
    print("prediction: ", yHat[index].argmax(), " | ", yHat[index])
    print("actual: ", trainingLabels[index], " | ", nn.oneHotEncode(trainingLabels)[index])



'''
Trying to match with example Lab solution, trying to do all images at once instead of loop method
x:  (5, 3) | trainingImages is (60000, 784) 
y:  (1, 5) [transpose to (5, 1)] | trainingLabels is (60000,) [converts to (60000, 10)]
w1: (3, 4) | (784, 16) 
z1: (5, 3) x (3, 4) = (5, 4) | (60000, 784) x (784, 16) = (60000, 16) | (1, 784) x (784, 16) = (1, 16)
w2: (4, 1) | (16, 10) | (16, 10) x (10, 10) = (16, 10) 
z2: (5, 4) x (4, 1) = (5, 1) | (60000, 16) x (16, 10) = (60000, 10) 
'''