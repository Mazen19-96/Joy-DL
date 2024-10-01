# Joy:
Joy is a simple deeplearning library written in python to creat and training Neural Networks .
### Includes:

1. Activation functions:
    * ReLU
    * Sigmoid
    * Softmax
    * leaky_relu
    * Tanh
    
2. Losses:
    * MSELoss
    * BCELoss
    * CCELoss
    * MAELoss
    
3. Optimizers:
    * SGD
    * GDMomentum #not complet
    * RMSprop    #not complet 

4. Layers:
    * Linear
    * Conv2d
    * MaxPool2d
    * RNN
    
    
5. nn:
   * Sequential : it is method to creat neural networks

### Examples:

* In examples directory there is a XOR problem.
* Sequential model creation:
import numpy as np
from Joy.train import train
from Joy.losses import *
from Joy.optim import *
from Joy.nn import Sequential
from Joy.layers import Linear 
from Joy.activation import Tanh

inputs=np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
 ])

targets=np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
 ])
net=Sequential([Linear(input_size=2,output_size=2),
                Tanh()
                ,Linear(input_size=2, output_size=2)])

train(net,inputs,targets, num_epochs=10000,loss=MSELoss(),optimizer=SGD(lr=0.01))
for x ,y in zip (inputs,targets):
    pred=net.forward(x)
    print(x, pred,y)
