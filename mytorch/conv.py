# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.L_in = 0
        self.x = 0

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size (#number_images), in_channel (RGB), input_size(SIZE_IMG))

            self.W (np.array): (out_channel, in_channel, kernel_size)
            self.b (np.array): (out_channel)    
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        (out_channel, in_channel, kernel_size) = self.W.shape
        (N,C_in,self.L_in) = x.shape
        self.x = x

        res1 = []
        for i in range(N):
            tempRes = np.array([x[i,:,k:k+kernel_size] for k in range(0,self.L_in-kernel_size+1,self.stride)])
            res1.append((np.tensordot(self.W,tempRes,axes=([2,1],[2,1])).T + self.b).T)

        return np.array(res1)
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
         Return:
            dx (np.array): (batch_size, in_channel, input_size)

            to modify: 
                dw (out_channel,in_channel, kernel_size)
                db (out_channel)
        """
        
        (out_channel, in_channel, kernel_size) = self.W.shape
        (a,b,c) = delta.shape
        
        
        new_delta = np.zeros((a,b,(c)*self.stride))
        new_delta[:,:,0:c*self.stride:self.stride] = delta
        
        left_pad = kernel_size-1
        right_pad = kernel_size-1
        new_delta = np.pad(new_delta,((0,0),(0,0),(left_pad,right_pad)),'constant', constant_values=(0,0))
        
        (N,C_in,third) = new_delta.shape

        #Invert the weights
        inv_weights = self.W[:,:,::-1]

        res1 = []
        for i in range(N):
            tempRes = np.array([new_delta[i,:,k:k+kernel_size] for k in range(0,third-kernel_size+1,1)])
            j = kernel_size-1
            tempRes = tempRes[:self.L_in]
            res1.append(np.tensordot(inv_weights,tempRes,axes=([2,0],[2,1])))


        
        self.db += delta.sum((0,2))

        (out_channel, in_channel, kernel_size) = self.W.shape
        N,C_in,t = self.x.shape
        (a,b,c) = delta.shape
        
        for x in range(c):
            m = x * self.stride
            self.dW[:,:] += np.tensordot(delta[:,:,x],self.x[:,:,m:m+kernel_size],axes=[0,0])
        
        return np.array(res1)


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape((self.b,self.c*self.w))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        (a,bc) = delta.shape
        return delta.reshape((self.b,self.c,self.w))
