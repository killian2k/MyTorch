import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.forw_val = []

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        def logSumExp(x):
            m = np.median(x)
            return m + np.log(np.exp(np.add(x,m * -1.)).sum())
        
        
                   
        self.logits = x
        self.labels = y
        self.forw_val = np.zeros(y.shape)
        dim = 0
        if(len(x.shape) > 0):
            dim = x.shape[0]
        res = np.zeros(dim)
        for i in range(dim):
            log_soft = (x[i] - logSumExp(x[i]))
            sum = -y[i].T.dot(log_soft)
            res[i] = sum
            self.forw_val[i] = np.exp(log_soft)
        
        return res        

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        
        return -self.labels + self.forw_val.reshape(self.labels.shape)
