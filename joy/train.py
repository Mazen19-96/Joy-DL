from Joy.tensor import Tensor
from Joy.nn import Sequential
from Joy.losses import LossFunctions, MSELoss, BCELoss, CCELoss, MAELoss
from Joy.optim import Optimizer, SGD
from Joy.data import  DataIterator, BatchIterator

def train(net: Sequential,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: LossFunctions = MSELoss(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
