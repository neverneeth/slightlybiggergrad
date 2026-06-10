from src.engine import Tensor
from src.nn import Linear, MLP
import numpy as np
import matplotlib.pyplot as plt

X_data = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
Y_data = np.sin(X_data) # Automatically inherits (100, 1)

X = Tensor(X_data, label='X', device='gpu')
Y = Tensor(Y_data, label='Y', device='gpu')

model = MLP(nin=1, nout=[16, 16, 16, 16, 1], activation=lambda x: x.tanh(), device='gpu')

learning_rate = 0.01
epochs = 20000

for epoch in range(epochs):
    Y_pred = model(X)
    loss = ((Y_pred - Y) * (Y_pred - Y)).mean()
    model.zero_grad()
    loss.backward_all()
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

plt.scatter(X.numpy(), Y.numpy(), color='gray', label='Target Data', alpha=0.5)

predictions = model(X).numpy()
plt.plot(X.numpy(), predictions, color='red', linewidth=3, label='MLP Prediction')

plt.legend()
plt.title("slightlybiggergrad vs. Sine Wave")
plt.show()