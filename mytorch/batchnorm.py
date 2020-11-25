import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        if eval:
            x_hat = (x-self.running_mean)/np.sqrt(self.running_var + self.eps)
            return (self.gamma * x_hat + self.beta)

        self.x = x

        self.mean = self.x.mean(axis=0)
        self.var = np.var(self.x,axis=0)
        self.norm = (self.x - self.mean)/(np.sqrt(self.var + self.eps))
        self.out = self.gamma * self.norm + self.beta

        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        x_avg = self.x - self.mean
        inv_std = 1/np.sqrt(self.var + self.eps)
        dX_hat = delta * self.gamma
        dVar = -0.5*np.sum(dX_hat * x_avg,axis=0)*np.power(inv_std,3)
        dMu = np.sum(-dX_hat*inv_std,axis=0) - 2 * np.mean(x_avg,axis=0) * dVar
        self.dbeta=np.sum(delta,axis=0,keepdims=True)
        self.dgamma=np.sum(delta*self.norm,axis=0,keepdims=True)
        m = delta.shape[0]
        return inv_std*dX_hat + dVar*2*x_avg/m + dMu/m

