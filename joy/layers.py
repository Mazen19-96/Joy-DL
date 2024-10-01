# Creation: 10.1.2021
# Author: Mazen Saleh
# Contact: Rmen1996@gmail.com
from typing import Dict , Union ,Tuple
import copy
import numpy as np
from Joy.tensor import Tensor
from Joy.functional import Parameter
from Joy.utils import col2im, im2col
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    computes output = inputs @ w + b
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        
        # Compute the gradients
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        # Return the new downstream gradient
        return grad @ self.params["w"].T



class Conv2d(Layer):
    """
    Convolutional Neural Networks (CNN) are a class of Neural Networks that uses convolutional filters.
    Their particularity is their ability to synthesize information and learn spatial features.
    They are mainly used in Image Analysis, but are also known as *sliding windows* in Natural Language Processing (NLP).

    ``Conv2d`` network applies a 2D convolution on a 4D tensor.
    """

    def __init__(self, in_channels, out_channels, filter_size, stride, pad):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        # initialized parameters follow a uniform distribution [-bound, bound]
        bound = 1 / (in_channels * np.product(filter_size))
        self.weight = Parameter.uniform((out_channels, in_channels, *filter_size), -bound, bound)
        self.bias = Parameter.zeros((out_channels, ))

    def forward(self, inputs):
        FN, C, FH, FW = self.weight.shape
        N, C, H, W = inputs.shape

        # TODO: display a warning if the stride does not match the input image size
        out_h = int((H + 2 * self.pad - FH) // self.stride) + 1
        out_w = int((W + 2 * self.pad - FW) // self.stride) + 1

        # Convolution
        col = im2col(inputs, FH, FW, self.stride, self.pad)
        col_weight = self.weight.reshape(FN, -1).T
        # Linear computation
        out = np.dot(col, col_weight) + self.bias
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # Save in the cache for manual back propagation
        self._cache['x'] = inputs
        self._cache['x_col'] = col
        self._cache['weight_col'] = col_weight

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.weight.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Parameters gradient
        db = np.sum(dout, axis=0)
        dw_col =np.dot(self._cache['x_col'].T, dout)
        dw = dw_col.transpose(1, 0).reshape(FN, C, FH, FW)
        # Downstream gradient
        dcol = np.dot(dout, self._cache['weight_col'].T)
        dx = col2im(dcol, self._cache['x'].shape, FH, FW, self.stride, self.pad)

        # Save the gradients
        # NOTE: we don't need to save column gradients as they wont be used during the optimization process.
        self._grads['bias'] = db
        self._grads['weight'] = dw

        return dx

    def inner_repr(self):
        """Display the inner parameter of a CNN"""
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, " \
               f"filter_size={self.filter_size}, stride={self.stride}, pad={self.pad}, " \
               f"bias={True if self.bias is not None else False}"

class MaxPool2d(Layer):
    """
    A ``Pooling`` layer extract features from a multi dimensional ``Tensor`` and map them into another one.
    This extraction is used to decrease the dimension of the input, and often used after a convolutional layer.
    """

    def __init__(self, pool_size, stride, pad):
        super().__init__()
        # Make sure the pool_size is a 2-d filter
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        # Initialize
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """Forward pass."""
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride)
        out_w = int(1 + (W - self.pool_size[1]) / self.stride)
        # Reshape the input into a 2-d tensor
        col = im2col(x, *self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, np.product(self.pool_size))

        # Keep track of the argmax indices for manual back-propagation
        argmax = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h + 2*self.pad, out_w + 2*self.pad, C).transpose(0, 3, 1, 2)

        # Save the parameters in the cache for manual back-propagation
        self._cache['x'] = x
        self._cache['argmax'] = argmax

        return out

    def backward(self, dout):
        """Manual backward pass for a MaxPool2d layer."""
        dout = dout.transpose(0, 2, 3, 1)
        # Initialize
        pool_size = np.product(self.pool_size)
        dmax = np.zeros((dout.size, pool_size))

        # Get the cache
        x = self._cache['x']
        argmax = self._cache['argmax']
        dmax[np.arange(argmax.size), argmax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, x.shape, *self.pool_size, self.stride, self.pad)

        return dx

    def inner_repr(self):
        """Display the inner parameter of a CNN"""
        return f"pool_size={self.pool_size}, stride={self.stride}, pad={self.pad}"



"""
Defines a basic Recurrent Neural Network.
"""

# atch template (batch_size, seq_length, inputs_length)
class RNN(Layer):
    """
    Recurrent neural network (RNN) is a type of neural network that has been successful in modelling sequential data,
    e.g. language, speech, protein sequences, etc.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Initialize all the weights (input - hidden - output)
        self.weight_ih = Parameter.orthogonal(shape=(input_dim, hidden_dim))
        self.weight_hh = Parameter.orthogonal(shape=(hidden_dim, hidden_dim))
        self.weight_ho = Parameter.orthogonal(shape=(hidden_dim, input_dim))
        # Initialize all the biases (hidden - output)
        self.bias_h = Parameter.zeros(shape=(hidden_dim,))
        self.bias_o = Parameter.zeros(shape=(input_dim,))
        # Initialize the first hidden cell
        self.weight_h0 = np.zeros(shape=(1, hidden_dim))

    def forward(self, inputs):
        """
        Computes the forward pass of a vanilla RNN.

        .. math::
            \begin{align*}
                h_{0} &= 0   \\
                h_t &= \text{tanh}(x \cdot W_{ih} + h_{t-1} \cdot W_{hh} + b_{h})    \\
                y &= h_t \cdot W_{ho} + b_{o}  \\
            \end{align*}


        Args:
            inputs (Tensor): sequence of inputs to be processed

         Returns:
             outputs (Tensor): predictions :math:`y`.
             hidden_states (Tensor): concatenation of all hidden states :math:`h_t`.
        """
        hidden_states = Tensor([self.weight_h0])
        outputs = Tensor([])
        # Initialize hidden_cell_0 (with zeros)
        hidden_state = hidden_states[0]
        # For each element in input sequence
        for t in range(inputs.shape[0]):
            # Compute new hidden state
            def tanh(x):
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

            hidden_state = tanh(np.dot(inputs[t], self.weight_ih) +
                                     np.dot(hidden_state, self.weight_hh) + self.bias_h)
            # Compute output
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            out =sigmoid(
                np.dot(hidden_state, self.weight_ho) + self.bias_o)
            # Save results and continue
            outputs = np.append(outputs, out)
            hidden_states = np.append(hidden_states, hidden_state)

        # Save in the cache (for manual back-propagation)
        self._cache['hidden_states'] = hidden_states

        return outputs, hidden_states

    # manual backpropagation
    def backward(self, dout):
        """
        Computes the backward pass of a vanilla RNN.
        Save gradients parameters in the ``_grads`` parameter.

        Args:
            dout (Tensor): upstream gradient.

        Returns:
            Tensor: downstream gradient
        """
        # Initialize gradients as zero
        dw_ih =np.zeros_like(self.weight_ih)
        dw_hh = np.zeros_like(self.weight_hh)
        dw_ho = np.zeros_like(self.weight_ho)
        db_h = np.zeros_like(self.bias_h)
        db_o = np.zeros_like(self.bias_o)
        # Get the cache
        hidden_states = self._cache['hidden_states']
        inputs = self._cache['x']

        # Keep track of hidden state derivative and loss
        dh_t = np.zeros_like(hidden_states[0])

        # For each element in output sequence
        # NB: We iterate backwards s.t. t = N, N-1, ... 1, 0
        for t in reversed(range(dout.shape[0])):
            # Back-propagate into output sigmoid
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
                
            
            def sigmoid_prime(x):
                return sigmoid * (1 - sigmoid)


            do =sigmoid_prime(dout[t])
            db_o += do
            # Back-propagate into weight_ho
            dw_ho += np.dot(hidden_states[t].T, do)
            # Back-propagate into h_t
            dh = np.dot(do, self.weight_ho.T) + dh_t
            # Back-propagate through non-linearity tanh
            def tanh(x):
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            def tanh_prime(x):
                return 1-tanh(x) * tanh (x)

            df = tanh_prime(hidden_states[t]) * dh
            db_h += df
            # Back-propagate into weight_ih
            dw_ih += np.dot(inputs[t].T, df)
            # Back-propagate into weight_hh
            dw_hh += np.dot(hidden_states[t - 1].T, df)
            dh_t = np.dot(df, self.weight_hh.T)

        #dx grad
        # dx = nets.dot(dout, self.weight_ih)

        # Save gradients
        self._grads["weight_ih"] = dw_ih
        self._grads["weight_hh"] = dw_hh
        self._grads["weight_ho"] = dw_ho
        self._grads["bias_h"] = db_h
        self._grads["bias_o"] = db_o

        return None
