
import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():

    def __init__(self):
        size_vector = 24
        number_inputs = 128
        lSizes = [8,8,16,4]

        #OUTCHANNEL = # number of neurons
        #INCHANNEL = # dimension of source
        #KERNEL_SIZE = # of bloc each of dim inchannel
        #Stride = size of jumpe between each bloc
        self.conv1 = Conv1D(in_channel=24,
            out_channel= 8,#lSizes[1] * 31,
            kernel_size=8,
            stride=4)#in_channel, out_channel, kernel_size, stride,weight_init_fn=None, bias_init_fn=None)
        self.conv2 = Conv1D(in_channel=8,
            out_channel= 16,
            kernel_size=1,
            stride=1)
        self.conv3 = Conv1D(in_channel=16,
            out_channel= 4,
            kernel_size=1,
            stride=1)
        self.layers = [self.conv1,ReLU(),self.conv2,ReLU(),self.conv3,Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        w1,w2,w3 = weights
        transp = w1.T
        a,b = transp.shape
        x = transp.reshape(self.conv1.out_channel,self.conv1.kernel_size,self.conv1.in_channel)

        self.conv1.W = (w1.T.reshape(self.conv1.out_channel,self.conv1.kernel_size,self.conv1.in_channel)).transpose((0,2,1))
        self.conv2.W = (w2.T.reshape(self.conv2.out_channel,self.conv2.kernel_size,self.conv2.in_channel)).transpose((0,2,1))
        self.conv3.W = (w3.T.reshape(self.conv3.out_channel,self.conv3.kernel_size,self.conv3.in_channel)).transpose((0,2,1))
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1D(in_channel=24,
            out_channel= 2,#lSizes[1] * 31,
            kernel_size=2,
            stride=2)#in_channel, out_channel, kernel_size, stride,weight_init_fn=None, bias_init_fn=None)
        self.conv2 = Conv1D(in_channel=2,
            out_channel= 8,
            kernel_size=2,
            stride=2)
        self.conv3 = Conv1D(in_channel=8,
            out_channel= 4,
            kernel_size=2,
            stride=1)
        self.layers = [self.conv1,ReLU(),self.conv2,ReLU(),self.conv3,Flatten()]

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, weights):
        
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        np.set_printoptions(threshold=sys.maxsize)
        w1, w2, w3 = weights
        
        in_channel = 24
        ker_size = 2
        out_channel = 2
        self.conv1.W = w1.T[:self.conv1.out_channel,:self.conv1.in_channel * self.conv1.kernel_size].reshape((self.conv1.out_channel,self.conv1.kernel_size,self.conv1.in_channel)).transpose((0,2,1))

        
        np_neurons_to = 8
        in_channel = 2
        ker_size = 2
        self.conv2.W = w2.T[:self.conv2.out_channel,:self.conv2.in_channel * self.conv2.kernel_size].reshape((self.conv2.out_channel,self.conv2.kernel_size,self.conv2.in_channel)).transpose((0,2,1))
        
        np_neurons_to = 4
        in_channel = 8
        ker_size = 2
        self.conv3.W = w3.T[:self.conv3.out_channel,:self.conv3.in_channel * self.conv3.kernel_size].reshape((self.conv3.out_channel,self.conv3.kernel_size,self.conv3.in_channel)).transpose((0,2,1))
        

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
