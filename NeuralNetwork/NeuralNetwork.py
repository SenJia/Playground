#
#This code is a simple implementation of neural network. 
#The 'NeuralNetwork' class contains initialisation, forwardpass, backpasstest and run methods. 
#The 'Layer' class is a container of weights, activations, loss and deltas for each layer. 
#Besides, all those operation methods are included in the 'OP' class(more will be added).
#
#Author: Sen Jia
#

import numpy as np
import random
from sklearn.datasets import load_iris


class OP:
    @staticmethod
    def norm_init(num):
        return np.random.normal(0,0.1,num)
    
    @staticmethod
    def sigmoid(x,derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            return x * (1 - x)

    @staticmethod
    def squared_error(output,target):
        loss = output - target
        loss = np.sum(np.power(loss,2))
        return loss

class Layer:
    def __init__(self,num,prev_num,init_method):
        """
        Initialising a layer, weigths and weight_deltas are in the size of i by j, where i is the number of units in previous layer and j is the number of units in current layer. Activations and loss are in the size of j by 1.
        """
        self.weights = np.random.rand(prev_num,num)
        self.unit_delta = np.zeros_like(self.weights)
        self.activations = np.zeros((num,1))
        self.unit_loss = np.zeros_like(self.activations)

class NeuralNetwork:
    def __init__(self,layers,max_iters,lr_step,learning_rate=0.1,batch_size=32):
        """    
        initialise NN architecture except input layer. Each initialised layer contains a 1D array of activations, a 2D array of weights, a 1D array of loss and a 2D array of deltas.
        """
        self._eta = learning_rate
        self._max_iters = max_iters
        self._batch_size = batch_size
        self._lr_step = lr_step

        self._batch_iter = None
        self.layers = []
        for i,num in enumerate(layers):
            if i == 0:   # ignore input layer
                continue
            else:
                prev_num = layers[i-1]
                self.layers.append(Layer(num,prev_num,OP.norm_init))  # initialise current layer, the number of neurons in the previous layer is required.
        self.num_class = layers[-1]

    def forward(self,inputs,actFunc):
        """
        Computing forward pass, derivative is set to false.
        """
        derivative = False
        for i,layer in enumerate(self.layers):
            if i == 0:
                netV = np.dot(inputs,layer.weights)
            else:
                inputs = self.layers[i-1].activations
                netV = np.dot(inputs,layer.weights)
            layer.activations = actFunc(netV)

    def backward(self,inputs,output,target,actFunc):
        """
        Computing backpropagation, derivative is set to true. Both the loss for each unit and the delta for each weight are computed within this method.
        """
        derivative = True
        for i in reversed(range(len(self.layers))):
            sig_prime = actFunc(self.layers[i].activations,derivative)
            if i == len(self.layers) - 1: # compute loss and delta for the last layer.
                self.layers[i].unit_loss = np.multiply((target-output),sig_prime)
            else:
                next_layer = self.layers[i+1]
                self.layers[i].unit_loss = np.multiply(sig_prime,np.dot(next_layer.weights,next_layer.unit_loss))


            if i == 0: # compute delta for the first layer.
                self.layers[i].unit_delta = np.add(self._eta*np.outer(inputs,self.layers[i].unit_loss),self.layers[i].unit_delta)
            else:
                self.layers[i].unit_delta = np.add(self._eta*np.outer(self.layers[i-1].activations,self.layers[i].unit_loss),self.layers[i].unit_delta)

               
    def update(self,num_sample):
        for i in range(len(self.layers)):
            self.layers[i].unit_delta /= float(num_sample)
            self.layers[i].weights += self.layers[i].unit_delta
            self.layers[i].unit_delta.fill(0.)

    def batch_iterator(self,trainingSet):
        batch=[]
        for i in range(self._batch_size):
            try:
                sample = self._batch_iter.next()
            except:
                random.shuffle(trainingSet)
                self._batch_iter = iter(trainingSet)
                sample = self._batch_iter.next()
            batch.append(sample)
        return batch

    def run(self,wholeSet):
        counter = 0
        iter_counter = 0
        total_num = len(wholeSet)
        train_ratio = 0.8
        train_size = int(total_num*train_ratio)
        random.shuffle(wholeSet)
        trainingSet = wholeSet[:train_size]
        testSet = wholeSet[train_size:]
        self._batch_iter = iter(trainingSet)
        while iter_counter < self._max_iters:
            batch = self.batch_iterator(trainingSet)
            loss = 0
            num_sample = len(batch)
            for sample in batch:
                feat, label = sample[0], sample[1]
                inputs = np.array(feat)
                target = np.zeros(self.num_class)
                target[label] = 1
                self.forward(inputs,OP.sigmoid)
                output=self.layers[-1].activations
                self.backward(inputs,output,target,OP.sigmoid)
                loss += OP.squared_error(output,target)
            self.update(num_sample)
            loss /= len(batch)
            counter += self._batch_size
            iter_counter += 1
            if iter_counter % 1000 == 0:
                print ("Iteration: {0} \t Loss: {1} ".format(iter_counter,loss))
            if iter_counter % 5000 == 0:
                acc = self.test(testSet)
                print ("Iteration: {0} \t Test Accuracy: {1} ".format(iter_counter,acc))
            if iter_counter%self._lr_step == 0:
                self._eta *= 0.1

    def test(self,testSet):
        correct = 0.0
        error = 0.0
        for sample in testSet:
            feat, label = sample[0], sample[1]
            inputs = np.array(feat)
            self.forward(inputs,OP.sigmoid)
            output = self.layers[-1].activations
            prediction = output.argmax()
            if prediction == label:
                correct += 1
            else:
                error += 1 
        return correct / (correct + error)

def main():
    # Teach network XOR function
    data = load_iris()
    feat = data.data
    label = data.target
    wholeSet = zip(feat,label)

    layers = []
    input_dim = 4
    layers.append(input_dim)
    hidden = [3]
    layers.extend(hidden)
    num_class = 3
    layers.append(num_class)
    # create a network with two input, two hidden, and one output nodes
    n = NeuralNetwork(layers,max_iters=50000,batch_size=32,lr_step=30000)
    n.run(wholeSet)


if __name__ == '__main__':
    main()
    
