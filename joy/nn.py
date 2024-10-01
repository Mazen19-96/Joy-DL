"""
What is a (Neural Network) NN?
    Single neuron == linear regression without applying activation(perceptron) 
    Basically a single neuron will calculate weighted sum of input(W.T*X) 
    and then we can set a threshold to predict output in a perceptron, 
    If weighted sum of input cross the threshold,
    perceptron fires and if not then perceptron doesn't predict.,
    Perceptron can take real values input or boolean values,
    Actually, when w⋅x+b=0 the perceptron outputs 0.
"""
from typing import Iterator, Sequence, Tuple
from Joy.tensor import Tensor
from Joy.layers import Layer

class Sequential:
    """
    Sequential : its the model to applying neural network ,
    inputs -> Linera layer -> activation layer -> probilty -> outputs
    """
    def __init__(self,layers:Sequence[Layer]) -> None:
        self.layers=layers

    def forward(self,inputs:Tensor)->Tensor:
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs

    def backward(self,grad:Tensor)->Tensor:
        for layer in reversed(self.layers):
            grad=layer.backward(grad)
        return grad

    def param_and_grad(self)-> Iterator[Tuple[Tensor,Tensor]]:
        for layer in self.layers:
            for name , param in layer.params.items():
                grad=layer.grads[name]
                yield param,grad

