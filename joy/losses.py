# Creation: 5.1.2021
# Author: Mazen Saleh
# Contact: Rmen1996@gmail.com
import numpy as np
from numpy.core.fromnumeric import mean
from Joy.tensor import Tensor

"""
    Base architecture for LossFunction.
    Every LossFuction Methods  should extends this class to inherits private attributes and
    helper methods.
    """

class LossFunctions:
    def loss(self,predicted:Tensor,actual:Tensor)->float:
        raise NotImplementedError
    
    def grad(self,predicted:Tensor,actual:Tensor)-> Tensor:
        raise NotImplementedError

class MSELoss(LossFunctions):
    """
    Function implements the mean squared error loss:
    measures the average of the squares of the errors,
    the average squared difference between the prdeicted values
    and the actual value. 
    :param predicted: (Tensor) Prediction tensor
    :param actual: (Tensor) One hot encoded label tensor   
    """
    def loss(self,predicted:Tensor,actual:Tensor)->float:
        if actual.shape != predicted.shape:
            raise ValueError("Wrong shapes")
        return np.sum((predicted - actual)**2).mean()
        
    
    def grad(self,predicted:Tensor,actual:Tensor)-> Tensor:
       return 2*(predicted - actual)

class BCELoss(LossFunctions):
    """
    This function implements the binary cross entropy loss:
    is measure from the field of information theroy,
    building upon entropy and generally calculating the 
    difference between two probability distributions.
    :param predicted: (Tensor) Prediction tensor
    :param actual: (Tensor) One hot encoded label tensor
    :return: (Tensor) Loss value
    """
    def loss(self,predicted:Tensor,actual:Tensor)->float:
        if actual.shape != predicted.shape:
            raise ValueError("Wrong shapes")
        return -np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    
    def grad(self,predicted:Tensor,actual:Tensor)-> Tensor:
        return -((actual / predicted) + (1 - actual) / (1 - predicted))

class  CCELoss(LossFunctions):
    """
   This function implements the categorical cross entropy loss:
   is a loss function that is used in multi-class classfication tasks,
   these are tasks where an example can only belong to one out of many possible
   categories,and the model must decide which one,formally,it is designed,
   to quantify the difference between two probability distributions.
   :param predicted: (Tensor) Prediction tensor
   :param actual: (Tensor) One hot encoded label tensor 
    """
    
    def loss(self,predicted:Tensor,actual:Tensor)->float:
        if actual.shape != predicted.shape:
            raise ValueError("Wrong shapes")
            #1e-15 or epsilon its just adding a very small number to avoid np.log(0)
        loses= - np.sum(actual * (np.log(predicted + 1e-15)))
        return loses / (len(actual))
    
    def grad(self,predicted:Tensor,actual:Tensor)-> Tensor:
        raise NotImplementedError

class MAELoss(LossFunctions):
    """
    Function implements the mean absolute error loss:
    measures the average of sum of absolute of the errors,
    the absolute error difference between the prdeicted values
    and the actual value. 
    :param predicted: (Tensor) Prediction tensor
    :param actual: (Tensor) One hot encoded label tensor   
    """
    def loss(self,predicted:Tensor,actual:Tensor)->float:
        if actual.shape != predicted.shape:
            raise ValueError("Wrong shapes")
        diff=np.sum(predicted-actual)
        abso=np.absolute(diff)
        meant=abso.mean()
        return meant
         
    def grad(self,predicted:Tensor,actual:Tensor)-> Tensor:
       return 2*(predicted - actual)
