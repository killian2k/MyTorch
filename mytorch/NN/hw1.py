

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.output_sizes = np.hstack((hiddens,self.output_size))
        self.input_output = zip(np.hstack((self.input_size,hiddens)), self.output_sizes)
        self.output = np.zeros(None)
        self.AllLayers = []
        self.bn_layers = []


        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        #Linear: (in_feature, out_feature, weight_init_fn, bias_init_fn)
        self.linear_layers = [Linear(int(t[0]),int(t[1]),weight_init_fn,bias_init_fn) for t in self.input_output]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            #BatchNorm(self, in_feature, alpha=0.9)
            self.bn_layers = [BatchNorm(self.output_sizes[i]) for i in range(num_bn_layers)]#hiddens]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        val = x
        self.AllLayers = []

        for i in range(self.nlayers):
            val = self.linear_layers[i].forward(val)
            self.AllLayers = np.hstack((self.AllLayers,self.linear_layers[i],False))
            if(i < len(self.bn_layers)):
                val = self.bn_layers[i].forward(val,not self.train_mode)
                self.AllLayers = np.hstack((self.AllLayers,self.bn_layers[i],False))
            val = self.activations[i].forward(val)
            self.AllLayers = np.hstack((self.AllLayers,self.activations[i],True))
        self.output = val
        return val

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for l in self.linear_layers:
            l.dW.fill(0.0)
            l.db.fill(0.0)
        for l in self.bn_layers:
            l.dgamma.fill(0.0)
            l.dbeta.fill(0.0)


    def step(self): 
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for l in self.linear_layers:
            # Update weights and biases here
            #lay = linear_layers[i]
        
            l.momentum_W = self.momentum * l.momentum_W - self.lr*l.dW
            l.momentum_b = self.momentum * l.momentum_b - self.lr*l.db
            l.W += l.momentum_W
            l.b += l.momentum_b


        # Do the same for batchnorm layers

        for l in self.bn_layers:
            l.beta -= self.lr * l.dbeta
            l.gamma -= self.lr * l.dgamma

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.loss = self.criterion.forward(self.output,labels)
        self.dLoss = self.criterion.derivative()
        layerDelta = self.dLoss
        '''
        
        print("LAYERS",self.AllLayers,"\n\n\n\n")
        print("dloss",dLoss,self.AllLayers[2].derivative())
        sigma = dLoss * self.AllLayers[2].derivative()
        self.AllLayers[0].backward(sigma)
        print(sigma,dLoss)


        '''

        for i in reversed(range(int(len(self.AllLayers)/2))):
            if(self.AllLayers[2*i+1]): #if its a function
                layerDelta = layerDelta * self.AllLayers[2*i].derivative()
            else:
                #initial_der = np.dot(initial_der,self.AllLayers[2*i].W.T)
                layerDelta = self.AllLayers[2*i].backward(layerDelta)
        

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    num_batches = len(trainx)/batch_size
    print(num_batches)
    print(len(trainx))
    #nepochs = 24 #TO REMOVE!!!!

    mlp.train()
    for e in range(nepochs):

        # Per epoch setup ...
        '''print(trainx.shape,trainy.shape)
        pair = np.array([trainx,trainy]).T
        np.random.shuffle(pair)
        tX, tY = pair.T'''
        p = np.random.permutation(len(trainx))
        tX,tY = trainx[p],trainy[p]

        mlp.zero_grads()
        for b in range(0, len(trainx), batch_size):
            # Train ...
            x = tX[b:b+batch_size]
            y = tY[b:b+batch_size]
            mlp.zero_grads()
            mlp.forward(x)
            mlp.backward(y)
            mlp.step()
            training_losses[e] += mlp.total_loss(y)
            training_errors[e] += mlp.error(y)
        mlp.eval()
        print("training errors", training_errors[e], " for ", e)

        for b in range(0, len(valx), batch_size):

            # Val ...
            x = valx[b:b+batch_size]
            y = valy[b:b+batch_size]
            mlp.forward(x)
            mlp.error(y)
            validation_losses[e] += mlp.total_loss(y)
            validation_errors[e] += mlp.error(y)
        # Accumulate data...
        print("validation errors", validation_errors[e], " for ", e)


    # Cleanup ...
    training_losses /= len(trainx)
    training_errors /= len(trainx)
    validation_errors /= len(valx)
    validation_losses /= len(valx)
    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    #raise NotImplemented
