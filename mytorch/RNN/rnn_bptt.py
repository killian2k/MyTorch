import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

# RNN Phoneme Seq-to-Seq
class RNN_Phoneme_BPTT(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                    RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        Initialize weights

        Parameters
        ----------
        rnn_weights:
        [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
         [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, x_lens, h_0=None):

        """
        RNN forward, multiple layers, multiple time steps

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)
            Input with padded form
        x_lens: (batch_size, )
            Input length
        h_0 : (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        out : (batch_size, seq_len, output_size)
            Output logits in padded form
        out_lens: (batch_size, )
            Output length
        """

        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        output_size = self.output_size
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        # store outputs at each time step, [batch_size * (seq_len, output_size)]
        out = np.zeros((batch_size, seq_len, output_size))
        self.x = x
        self.hiddens.append(hidden.copy())

        # Iterate through the sequence
            # Iterate over the length of self.rnn (through the layers)
                # Run the rnn cell with the correct parameters and update
                    # the parameters as needed. Update hidden.
            # Similar to above, append a copy of the current hidden array to the hiddens list

            # Get the output of last hidden layer and feed it into output layer 
            # Save current step output

        # Return output and output length
        u = x
        for s in range(seq_len):
            b = x[:,s]

            for layer in range(len(self.rnn)):
                new_h = self.rnn[layer](b,hidden[layer])
                b = new_h
                hidden[layer] = new_h.copy()

            self.hiddens.append(hidden.copy())
            out[:,s] = self.output_layer(new_h)
            out_lens = x_lens

        return out, out_lens

    def backward(self, delta, delta_lens):

        """
        RNN Back Propagation Through Time (BPTT) after CTC Loss

        Parameters
        ----------
        delta : (batch_size, seq_lens, output_size)
        gradient w.r.t. each time step output dY(i), i = 0, ...., seq_len - 1

        delta_lens : (batch_size)
        sequence length for each sample

        Returns
        -------
        dh_0 : (num_layers, batch_size, hidden_size)
        gradient w.r.t. the initial hidden states
        """

        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))


        '''
        Pseudocode:
        * Get delta mask from delta_lens with 0 and 1
            delta_mask * delta sets gradient at padding time steps to 0
        * Iterate from 0 to batch_size - 1
        
            * Iterate in reverse order time (from b_th seq_len - 1 to 0)
                * Get dh[-1] from backward from output layer
                * Iterate in reverse order of layers (from num_layer - 1 to 0)
                    * Get h_prev_l either from hiddens or x depending on the layer
                        (Recall that hiddens has an extra initial hidden state)
                    * Use dh and hiddens to get the other parameters for the backward method
                        (Recall that hiddens has an extra initial hidden state)
                    * Update dh with the new dh from the backward pass of the rnn cell
                    * If you aren't at the first layer, you will want to add
                        dx to the gradient from l-1th layer.
            * Save dh_0 at current b_th sample
        '''

        # Attention: For Linear output layer backward, "derivative" function is added 
        #            to compute with given delta and input x 
        #            (same thing as Tanh.derivative(state = None))
        
        #* Get delta mask from delta_lens with 0 and 1
        #    delta_mask * delta sets gradient at padding time steps to 0
        #dh = dh_0.copy()
        for i,leng in enumerate(delta_lens):
            delta[i,leng:].fill(0)
        
        for seq_index in reversed(range(seq_len)) : #Time step
            seq_in_hiddens = seq_index+1

        #   * Get dh[-1] from backward from output layer
            dh_0[-1] += self.output_layer.derivative(delta[:,seq_index],self.hiddens[seq_in_hiddens][-1])

        #   * Iterate in reverse order of layers (from num_layer - 1 to 0)
            for index_layer in reversed(range(self.num_layers)):
                h_prev_l = self.x[:,seq_index,:] if index_layer == 0 else self.hiddens[seq_in_hiddens][index_layer-1]
                
                delta_param = dh_0[index_layer]
                h = self.hiddens[seq_in_hiddens][index_layer]
                h_prev_t = self.hiddens[seq_in_hiddens-1][index_layer]

                new_dx,new_dh = self.rnn[index_layer].backward(delta_param,h,h_prev_l,h_prev_t)
                dh_0[index_layer] = new_dh
                if index_layer is not 0:
                    dh_0[index_layer-1] += new_dx
            
            
        return dh_0