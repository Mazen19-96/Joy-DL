# Creation: 3.1.2021
# Author: Mazen Saleh
# Contact: Rmen1996@gmail.com
from typing import Callable
import numpy as np
from Joy.tensor import Tensor
from Joy.layers import Layer
"""
Activation is responsible for adding
non-linearity function to the output of a neural network model,
Without an activation function, a neural network is simply a linear regression. 
math:
Y=Activation(sum(inputs * weight)+ bias)

"""

class Activation(Layer):
    
    F=Callable[[Tensor] , Tensor]
    def __init__(self, f: F ,f_prime: F) -> None:
        super().__init__()
        self.f= f
        self.f_prime=f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:    
        return self.f_prime(self.inputs) * grad


#Tanh
def tanh(x:Tensor) -> Tensor:
    """
    he hyperbolic tangent activation function is also referred to simply as the Tanh  function,
    It is very similar to the sigmoid activation function and even has the same S-shape.
    The function takes any real value as input and outputs values in the range -1 to 1. 
    The larger the input (more positive), the closer the output value will be to 1.0, 
    whereas the smaller the input (more negative), the closer the output will be to -1.0., defined as:

    .. math::
        text{tanh(x)} = frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = tanh(in_array)
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def tanh_prime(x:Tensor) ->Tensor:
    """
    First order derivative of the "tanh" function
    .. math::
        text{tanh'(x)} =1 - text{tanh}^2(x)
    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = tanh_prime(in_array)

    """
    return 1-tanh(x) * tanh (x)

class Tanh(Activation):

    
    def __init__(self):
        super().__init__(tanh,tanh_prime)
    
#sigmoid
def sigmoid(x:Tensor)->Tensor: #need some work
    """
   A sigmoid function is a mathematical function having 
   a characteristic "S"-shaped curve or sigmoid curve. , defined as:

    .. math::
        text{S(x)} = frac{1}{1 + e^{-x}}

    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = sigmoid(in_array)
    """
    return 1 / (1 + np.exp(-x))           
def sigmoid_prime(x:Tensor)->Tensor:
    """
    First order derivative of the "sigmoid" function
    .. math::
        text{S'(x)} =frac{ e^{-x} }{ (1+ e^{-x})^2 }
    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = sigmoid_prime(in_array)

    """
    return sigmoid * (1 - sigmoid)

class Sigmoid(Activation):
    
    def __init__(self):
        super().__init__(sigmoid,sigmoid_prime)

#relu
def relu(x):
    """
    Relu or Rectified Linear Activation Function 
    is the most common choice of activation function in the world of deep learning,
    Relu provides state of the art results ,
    and is computationally very efficient at the same time, defined as:

    .. math::
        text{relu(x)} = \max{(0, x)}

    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = relu(in_array)
    """
    return  np.maximum(np.zeros_like(x), x)

def relu_prime(x):
    """
    First order derivative of the "relu" function
    .. math::
        \text{relu'(x)} =
                        \begin{cases}
                          1, &\quad x \ge 0 \\
                          0, &\quad x < 0.
                        \end{cases}
    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        out_array = relu_prime(in_array)

    """
    return np.where(x >= 0, np.ones_like(x), np.zeros_like(x))
class Relu(Activation):
    
    def __init__(self):
        super().__init__(relu,relu_prime)

#leaky relu

def leaky_relu(x, alpha=0.01):
    """ 
    The Leaky ReLu function is an improvisation of the regular ReLu function.
    To address the problem of zero gradient for negative value.
    note:
    1.problem of zero gradient for negative value:
    the values of x less than zero, the gradient is 0. This means that weights and biases
    for some neurons are not updated. It can be a problem in the training process
    To overcome this problem, we have the Leaky ReLu function. Let’s learn about it nex
    2.alpha:
    small linear component of x to negative inputs.
    .. math::
        text{leaky_relu(x)} = \max{(alpha times x, x)}

    Args:
        alpha (float, optional): slope towards :math:`-\infty`.

    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        alpha = 0.1
        out_array = leaky_relu(in_array, alpha)
    """
    return np.where(x > 0, x, x * alpha)
def leaky_relu_prime(x, alpha=0.01):
  
    """First order derivative of "leaky_relu" function, defined as:

    .. math::
        \text{leaky_relu'(x)} =
                        \begin{cases}
                          1, &\quad x \ge 0 \\
                          \alpha, &\quad x < 0.
                        \end{cases}

    Args:
        alpha (float, optional): slope towards :math:`-\infty`.

    Examples::

        in_array = np.array([-5, 2, 6, -2, 4])
        alpha = 0.1
        out_array = leaky_relu_prime(in_array, alpha)
    """
    return np.where(x > 0, np.ones_like(x), alpha * np.ones_like(x))
class Leaky_Relu(Activation):
    
    def __init__(self):
        super().__init__(leaky_relu,leaky_relu_prime)

#softmax
def softmax(x):
	return np.exp(x) / np.exp(x).sum()
def softmax_prime(softmax):
    s=softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s,s.T)
class Softmax(Activation):
    
    def __init__(self):
        super().__init__(softmax,softmax_prime)