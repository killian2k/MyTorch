import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        (self.N,self.C_in,self.L_in,self.H_in) = x.shape

        self.pidx = []
        u = []

        for b in range(self.N):
            for j in range(self.C_in):
                m = 0
                for _x in range(0,self.L_in-self.kernel+1,self.stride):
                    n = 0
                    for _y in range(0,self.H_in-self.kernel+1,self.stride):
                        a = x[b,j,_x:_x+self.kernel,_y:_y+self.kernel]
                        self.pidx.append(np.unravel_index(np.argmax(a, axis=None), a.shape))
                        u.append(np.max(a))
                        n+= 1
                    m+=1

        self.pidx = np.array(self.pidx).reshape(self.N,self.C_in,m,n,2)

        return np.array(u).reshape(self.N,self.C_in,m,n)

    
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros((self.N, self.C_in,self.L_in,self.H_in))
        for b in range(self.N):
            for j in range(self.C_in):
                for _x in range(delta.shape[2]):
                    for _y in range(delta.shape[3]):
                        u,v = self.pidx[b,j,_x,_y]
                        dx[b,j,u + _x*self.stride,v + _y * self.stride] = delta[b,j,_x,_y]
        return dx

class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        (self.N,self.C_in,self.L_in,self.H_in) = x.shape

        self.pidx = []
        u = []

        for b in range(self.N):
            for j in range(self.C_in):
                m = 0
                for _x in range(0,self.L_in-self.kernel+1,self.stride):
                    n = 0
                    for _y in range(0,self.H_in-self.kernel+1,self.stride):
                        a = x[b,j,_x:_x+self.kernel,_y:_y+self.kernel]
                        self.pidx.append(np.unravel_index(np.argmax(a, axis=None), a.shape))
                        u.append(np.mean(a))
                        n+= 1
                    m+=1

        self.pidx = np.array(self.pidx).reshape(self.N,self.C_in,m,n,2)
        return np.array(u).reshape(self.N,self.C_in,m,n)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros((self.N, self.C_in,self.L_in,self.H_in))
        for b in range(self.N):
            for j in range(self.C_in):
                for _x in range(delta.shape[2]):
                    for _y in range(delta.shape[3]):
                        u,v = self.pidx[b,j,_x,_y]
                        dx[b,j,_x * self.stride:_x*self.stride + self.kernel,_y * self.stride:_y* self.stride + self.kernel] += delta[b,j,_x,_y]/(self.kernel**2)

        return dx
