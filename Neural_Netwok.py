# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:36:07 2019

@author: edgar.youssef
"""

import math
import numpy as np
import cv2 as cv
from PIL import Image

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output);
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight;
            dendron.weight = dendron.weight + dendron.dWeight;
            dendron.connectedNeuron.addError(dendron.weight * self.gradient);
        self.error = 0;


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword();

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output


def main():
    topology = []
    topology.append(2048)
    topology.append(40)
    topology.append(18)
    topology.append(1)
    net = Network(topology)
    Neuron.eta = 0.09
    Neuron.alpha = 0.5
    
    img = Image.open('sim1.jpg').convert('RGBA')
    arr = np.array(img)
    flat_arr1 = arr.ravel()
    flat_arr1 = [i/sum(flat_arr1) for i in flat_arr1]
    
    img = Image.open('sim2.jpg').convert('RGBA')
    arr = np.array(img)
    flat_arr2 = arr.ravel()
    flat_arr2 = [i/sum(flat_arr2) for i in flat_arr2]
    
    img = Image.open('sim3.jpg').convert('RGBA')
    arr = np.array(img)
    flat_arr3 = arr.ravel()
    flat_arr3 = [i/sum(flat_arr3) for i in flat_arr3]
    
    while True: 
     err = 0
     inputs = [flat_arr1,flat_arr2]
     outputs = [[0], [1]]
     for i in range(len(inputs)):
        net.setInput(inputs[i])
        net.feedForword()
        net.backPropagate(outputs[i])
        err = err + net.getError(outputs[i])
     print ("error: ", err)
     if err < 0.11:
         break

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        c = input("type 3nd input :")
        net.setInput([a, b, c])
        net.feedForword()
        print (net.getThResults())


if __name__ == '__main__':
    main()