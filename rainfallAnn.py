

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#Learning rate and Momentum
p = 0.001
alpha = 0.01


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.deltaWeight = 0.0

class Neuron:
    def __init__(self,previousLayer=None):
        #Previous layer Connections
        self.connectedTo = []
        self.output = 0.0
        self.error = 0.0
        self.gradient = 0.0
        #If applies to input and bias neurons
        if previousLayer == None:
            pass
        else:
            for neuron in previousLayer:
                newConnection = Connection(neuron)
                self.connectedTo.append(newConnection)


    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def sigmoid(self, output):
        return (1/(1+(math.e ** (-output * 1.0))))

    def sigmoidDifferential(self, output):
        return (output * (1.0-output))

    def setError(self, error):
        self.error = error

    def addError(self, error):
        self.error += error
    
    def forward(self):
        outputSum = 0.0
        if len(self.connectedTo) == 0:
            return
        for connections in self.connectedTo:
            outputSum += connections.connectedNeuron.getOutput() * connections.weight
        self.output = self.sigmoid(outputSum)
        
    def backward(self):

        #Calculate gradient
        self.gradient = self.error * self.sigmoidDifferential(self.output)

        for connection in self.connectedTo:
            connection.deltaWeight = p * (self.gradient * connection.connectedNeuron.output) + alpha * connection.deltaWeight
            connection.weight += connection.deltaWeight
            connection.connectedNeuron.addError(connection.weight * self.gradient)
        self.error = 0
        
class Network:
    def __init__(self,structure):
        self.net = []
        for layer in structure:

            #No previous layer (Input)
            if len(self.net) == 0:
                newLayer = [Neuron(None) for x in range(layer)]
                

            #Creates layer of neurons in network
            else:
                newLayer = [Neuron(self.net[-1]) for x in range(layer)]

            #Bias
            newLayer.append(Neuron(None))
            newLayer[-1].setOutput(1)
            self.net.append(newLayer)
    

    def forwardPass(self):
        for x in range(1,len(self.net)):
            for neuron in self.net[x]:
                neuron.forward()

    def backwardPass(self, correctOutput):
        #Calculating error between output obtained and correct output
        for x in range(len(correctOutput)):
            self.net[-1][x].setError(correctOutput[x]-self.net[-1][x].getOutput())

        #Backpropogation of each neuron
        for layer in self.net[::-1]:
            for neuron in layer:
                neuron.backward()

    #Set input layer neuron values
    def setInput(self, enteredInput):
        for x in range(len(enteredInput)):
            self.net[0][x].setOutput(enteredInput[x])

    #Root mean square for overall network error
    def getError(self, correctOutput):
        error = 0

        for i in range(len(correctOutput)):
            networkError = (correctOutput[i] - self.net[-1][i].getOutput()) 
            error = (error + networkError) ** 2
        error = error / len(correctOutput)
        return math.sqrt(error)

    #Retrieving output from network
    def getResults(self):
        correctOutput = []
        for neuron in self.net[-1]:
            correctOutput.append(neuron.getOutput())

        #Remove bias
        correctOutput.pop()
        
        return correctOutput
    


#destandardizing function
def destandardize(x,rmax,rmin):
    return (((x+0.1)/0.8)(rmax-rmin)+rmin)


#Creating Neural Net
ann = Network([3,5,1])



#Train
train = pd.read_excel("train.xlsx")

print("Training : ")
validation = pd.read_excel("validation.xlsx")
for x in range(501):
    err = 0
    for i in range(train.shape[0]):
        ann.setInput([train.loc[i][0], train.loc[i][1], train.loc[i][2]])
        ann.forwardPass()
        ann.backwardPass([train.loc[i][3]])
        err += ann.getError([train.loc[i][3]])

    if x % 100 == 0:
        print("Training Epoch",x, "Error Avg :",err/(train.shape[0]))
        err2 = 0
        for i in range(validation.shape[0]):
            ann.setInput([validation.loc[i][0], validation.loc[i][1], validation.loc[i][2]])
            ann.forwardPass()
            err2 += ann.getError([train.loc[i][3]])
        print("Validation Epoch",x, "Error Avg :",err2/(validation.shape[0]))


        


print("Training finished")

testing = pd.read_excel("test.xlsx")
err3 = 0
xRange = []
observed = []
expected = []
for i in range(testing.shape[0]):
    ann.setInput([testing.loc[i][0], testing.loc[i][1], testing.loc[i][2]])
    ann.forwardPass()
    xRange.append(i)
    observed.append(testing.loc[i][3])
    expected.append(ann.getResults())
    print("Testing",i, "Error :",ann.getError([testing.loc[i][3]]))
    err3 += ann.getError([testing.loc[i][3]])
print("Testing Error Avg :",err3/(testing.shape[0]))


plt.xlabel("Days")
plt.ylabel("Rainfall")
#Red is observed
plt.plot(xRange,observed,'r')

#Blue is expected
plt.plot(xRange,expected,'b')
plt.show()




















                    











