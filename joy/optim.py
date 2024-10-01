
import numpy as np
from Joy.nn import Sequential

class Optimizer: 
    def step(self,net: Sequential)->None:        
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self,lr: float = 0.01) -> None:
        """
        Constructor
        :param lr: (float) Learning rate to be utilized
        """
        # Save parameters
        self.lr = lr

    def step(self,net:Sequential)-> None:
        """
        Method performs optimization step
        """
        # Loop over all parameters
        for param,grad in net.param_and_grad():
             # Perform gradient decent
            param-=self.lr * grad
"""
class GDMomentum(Optimizer):#also need some work
    def __init__(self,variables_list: list,lr: float = 0.01,momentum:float = 0) -> None:
       
        # Save parameters
        super().__init__(variables_list)
        self.lr = lr
        self.momentum= momentum
    def initialize_state(state, variable):
        # Velocity for momentum.
        state['velocity'] = np.zeros_like(variable.grad)

    def step(self,net:Sequential  )-> None:
        for i,  param,grad in net.param_and_grad():
            # Store a moving average f the gradients
            velocity = state['velocity'][i]
            # Moving average
            velocity = self.momentum * velocity - self.lr * grad
            # Inplace update
            param += velocity
            # Update the cache
            self ._cache['velocity'][i] = velocity

         

class RMSprop(Optimizer):
   
    def __init__(self, parameters, lr=1e-2, decay=0.99, epsilon=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self._cache = {'velocity': [np.zeros_like(p) for p in self.param_and_grad()]}

    def step(self):
       
        for i, parameter in enumerate(self.parameters):
            # Store a moving average f the gradients
            velocity = self._cache['velocity'][i]
            # Moving average
            velocity = self.decay * velocity + (1 - self.decay) * (parameter.grad ** 2)
            # Inplace update
            parameter -= self.lr * parameter.grad / (np.sqrt (velocity) + self.epsilon)
            # Update the cache
            self._cache['velocity'][i] = velocity

        
"""

    