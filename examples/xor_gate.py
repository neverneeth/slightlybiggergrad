from src.engine import Tensor
from src.nn import MLP
import numpy as np


# 1. The Dataset (Batch of 4 samples)
X = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], label='X')

Y = Tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], label='Y')

# 2. The Model: 2 inputs -> 4 hidden -> 1 output
# We use a lambda to pass the tanh method we wrote!
model = MLP(nin=2, nout=[4, 1], activation=lambda x: x.tanh())

learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    Y_pred = model(X)
    loss = ((Y_pred - Y) * (Y_pred - Y)).sum()
    model.zero_grad()
    loss.backward_all()
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")