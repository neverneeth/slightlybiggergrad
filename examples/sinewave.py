from src.engine import Tensor
from src.nn import Linear, MLP
import numpy as np
import matplotlib.pyplot as plt

xs = [x for x in np.linspace(0, 2 * np.pi, 100)]
X = Tensor(xs, label='X', device='gpu')

ys = [np.sin(x) for x in xs]
Y = Tensor(ys, label='Y', device='gpu')

model = MLP(nin=1, nout=[16, 16, 1], activation=lambda x: x.tanh(), device='gpu')

learning_rate = 0.1
epochs = 5000

for epoch in range(epochs):
    Y_pred = model(X.data.reshape(-1, 1))
    loss = ((Y_pred - Y) * (Y_pred - Y)).mean()
    model.zero_grad()
    loss.backward_all()
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")


plt.scatter(X.data, Y.data, color='gray', label='Noisy Data', alpha=0.5)
plt.plot(X.data, Y_pred.data, color='red', linewidth=3, label='MLP Prediction')
plt.legend()
plt.title("slightlybiggergrad vs. Sine Wave")
plt.show()

