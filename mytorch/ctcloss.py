import numpy as np
from loss import Criterion
from loss import SoftmaxCrossEntropy

class CTCLoss(Criterion):
    """
    CTC loss
    """
    def __init__(self, blank = 0):
        """
        Argument:
            blank (int, optional) – blank label index. Default 0.
        """
        #-------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.target_length = None
        self.input_length = None
        self.alphas = []
        self.betas = []
        self.gammas = []
        self.S_exts = []

        self.blank = blank
        #<---------------------------------------------

    def __call__(self, a, b, c, d):
        #-------------------------------------------->
        # Don't Need Modify
        return self.forward(a, b, c, d)
        #<---------------------------------------------

    
    def forward(self, logits, target, input_length, target_length):
        #-------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_length = input_length
        self.target_length = target_length

        #<---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths 
        #####  and then the mean over the batch is taken

        B, L = target.shape
        total_loss = np.zeros(B)


        for b in range(B):
            logi = logits[:,b]
            S_ext, Skip_Conn = self._ext_seq_blank(target[b,:target_length[b]])
            alpha = self._forward_prob(logits[:input_length[b], b, :], S_ext, Skip_Conn)
            beta = self._backward_prob(logits[:input_length[b], b, :], S_ext, Skip_Conn)
            gamma = self._post_prob(alpha, beta)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.gammas.append(gamma)
            self.S_exts.append(S_ext)

            local_loss = 0
            for t in range(gamma.shape[0]):
                for r in range(gamma.shape[1]):
                    a = logi[t, S_ext[r]]
                    local_loss -= gamma[t, r] * np.log(a)

            total_loss[b] = local_loss

        return np.average(total_loss)


    def derivative(self):

        #-------------------------------------------->
        # Don't Need Modify
        L, B, H = self.logits.shape
        dy = np.zeros((L, B, H))
        #<---------------------------------------------


        N = self.alphas[0].shape[0]
        T = self.logits.shape[0]


        for b in range(B):
            l = self.logits[:,b]
            #-------------------------------------------->
            # Computing CTC Derivative for single batch
            #<---------------------------------------------
            
            #-------------------------------------------->
            #dy[:,b,:] = 
            for t in range(self.gammas[b].shape[0]):
                dy[t,b].fill(0)
                for i in range(self.gammas[b].shape[1]):
                    #print(self.S_exts[b][i],self.gammas[b][t,i] / l[t,self.S_exts[b][i]])
                    dy[t,b,self.S_exts[b][i]] -= self.gammas[b][t,i] / l[t,self.S_exts[b][i]]
            #<---------------------------------------------
        
        return dy


    def _ext_seq_blank(self, target):
        """
        Argument:
            target (np.array, dim = 1) - target output
        Return:
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Connect (np.array, dim = 1) - skip connections
        """
        S_ext = []
        #Skip_Connect = []

        #-------------------------------------------->

        # Your Code goes here
        S_ext = np.zeros(2 * target.shape[0] + 1, dtype=int)
        Skip_Connect = S_ext.copy()
        S_ext.fill(self.blank)
        for i,el in enumerate(target):
            S_ext[1 + 2*i] = el
            if i is not 0:
                Skip_Connect[1 + 2*i] = 1
        #<---------------------------------------------

        return S_ext, Skip_Connect

    def _forward_prob(self, logits, S_ext, Skip_Conn):
        """
        Argument:
            logits (np.array, dim = (input_len, channel)) - predict probabilities
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Conn
        Return:
            alpha (np.array, dim = (output len, out channel)) - forward probabilities
        """
        N, T = len(S_ext), len(logits)
        alpha = np.zeros(shape = (T, N))

        #-------------------------------------------->
        alpha[0,0] = logits[0,S_ext[0]]
        alpha[0,1] = logits[0,S_ext[1]] 
        alpha[0,2:] = 0
        for t in range(1,T):
            alpha[t,0] = alpha[t-1,0] * logits[t,S_ext[0]]
            for i in range(1,N):
                alpha[t,i] = alpha[t-1,i-1] + alpha[t-1,i]
                if Skip_Conn[i]:
                    alpha[t,i] += alpha[t-1,i-2]
                alpha[t,i] *= logits[t,S_ext[i]]
        #<---------------------------------------------

        return alpha

    def _backward_prob(self, logits, S_ext, Skip_Conn):
        """
        Argument:
            logits (np.array, dim = (input len, channel)) - predict probabilities
            S_ext (np.array, dim = 1) - extended sequence with blanks
            Skip_Conn - 
        Return:
            beta (np.array, dim = (output len, out channel)) - backward probabilities
        """
        N, T = len(S_ext), len(logits)
        beta = np.zeros(shape = (T, N))

        #-------------------------------------------->
        beta[-1] = 0
        beta[-1,-1] = 1
        beta[-1,-2] = 1
        
        for j in reversed(range(T-1)):
            t = j+1
            beta[j,-1] = beta[t,-1] * logits[t,S_ext[-1]]
            for l in reversed(range(N-1)):
                i = l
                beta[j,i] = beta[t,i] * logits[t,S_ext[i]] + beta[t,i+1]*logits[t,S_ext[i+1]]
                if i < N-3 and Skip_Conn[i+2]:
                    beta[j,i] += beta[t,i+2] * logits[t,S_ext[i+2]]
        
        #<---------------------------------------------

        return beta


    def _post_prob(self, alpha, beta):
        """
        Argument:
            alpha (np.array) - forward probability
            beta (np.array) - backward probability
        Return:
            gamma (np.array) - posterior probability
        """
        #-------------------------------------------->
        T,N = alpha.shape
        sumgamma = np.zeros(T)
        gamma = np.zeros((T,N))


        
        # Your Code goes here
        for t in range(T):
            sumgamma[t] = 0
            for i in range(N):
                gamma[t,i] = alpha[t,i] * beta[t,i]
                sumgamma[t] += gamma[t,i]

            for i in range(N):
                gamma[t,i] = gamma[t,i] / sumgamma[t]

        #<---------------------------------------------

        return gamma
