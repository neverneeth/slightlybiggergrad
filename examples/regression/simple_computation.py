import numpy as np
from src.engine import Tensor


# 1. Setup a mini-batch of inputs, weights, and biases
# X: Batch of 2 samples, 3 features each -> Shape (2, 3)
X = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]], label='X')

# W: 3 inputs, 2 output neurons -> Shape (3, 2)
W = Tensor([[0.1,  0.2],
            [-0.1, 0.3],
            [0.5, -0.2]], label='W')

# b: Biases for the 2 output neurons -> Shape (2,)
b = Tensor([0.5, -0.5], label='b')

# 2. Forward Pass: Y = X @ W + b
XW = X @ W
XW.label = 'XW'
Y = XW + b
Y.label = 'Y'
O = Y.tanh()
O.label = 'O'
print(f"Forward Pass Output O:\n{O.data}\nShape: {O.data.shape}\n")

O.backward_all()

print("--- Gradients ---")
print(f"Gradient for b (Notice how it summed the batch!):\n{b.grad}\n")
print(f"Gradient for W:\n{W.grad}\n")
print(f"Gradient for X:\n{X.grad}\n")
print(f"Gradient for Y:\n{Y.grad}\n")