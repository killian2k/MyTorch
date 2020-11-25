import numpy as np
import math

"""
In the linear.py file, attributes have been added to the Linear class to make
implementing Adam easier, check them out!

self.mW = np.zeros(None) #mean derivative for W
self.vW = np.zeros(None) #squared derivative for W
self.mb = np.zeros(None) #mean derivative for b
self.vb = np.zeros(None) #squared derivative for b
"""

class adam():
    def __init__(self, model, beta1=0.9, beta2=0.999, eps = 1e-8):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = self.model.lr
        self.t = 0 # Number of Updates

    def step(self):
        '''
        * self.model is an instance of your MLP in hw1/hw1.py, it has access to
          the linear layer's list.
        * Each linear layer is an instance of the Linear class, and thus has
          access to the added class attributes dicussed above as well as the
          original attributes such as the weights and their gradients.
        '''
        self.t += 1
        for ll in self.model.linear_layers:
          ll.mW = self.beta1 * ll.mW + (1-self.beta1) * ll.dW
          ll.vW = self.beta2 * ll.vW + (1-self.beta2) * ll.dW**2
          ll.mb = self.beta1 * ll.mb + (1-self.beta1) * ll.db
          ll.vb = self.beta2 * ll.vb + (1-self.beta2) * ll.db**2
          mWHat = ll.mW/(1-self.beta1**self.t)
          mbHat = ll.mb/(1-self.beta1**self.t)
          vWHat = ll.vW/(1-self.beta2**self.t)
          vbHat = ll.vb/(1-self.beta2**self.t)
          ll.W -= self.lr*mWHat/(np.sqrt(vWHat+self.eps))
          ll.b -= self.lr*mbHat/(np.sqrt(vbHat+self.eps))




