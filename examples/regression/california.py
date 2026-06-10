import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import os

from src.engine import Tensor
from src.nn import MLP
from src.data import load_california_housing, train_test_split, StandardScaler

# Load and preprocess data


X, Y_raw = load_california_housing(path="./datasets/housing.csv")
Y = Y_raw/100000  
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

device = 'gpu'  
X_train_tensor = Tensor(X_train, device=device)
Y_train_tensor = Tensor(Y_train, device=device)

model = MLP(nin=X_train.shape[1], nout=[512, 256, 128, 64, 32, 1], activation=lambda x: x.tanh(), device=device)
EPOCHS = 2000
losses = []
for epoch in range(EPOCHS):
    model.zero_grad()
    Y_pred = model(X_train_tensor)
    loss = ((Y_pred - Y_train_tensor) * (Y_pred - Y_train_tensor)).mean()
    loss.backward_all()
    
    lr = 0.01
    for p in model.parameters():
        p.data -= lr * p.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
        losses.append(loss.numpy())

plt.plot(range(0, EPOCHS, 10), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

X_test_tensor = Tensor(X_test, device=device)
Y_test_tensor = Tensor(Y_test, device=device)

Y_test_pred = model(X_test_tensor)
test_loss = ((Y_test_pred - Y_test_tensor) * (Y_test_pred - Y_test_tensor)).mean()
print(f"Test Loss: {test_loss.data}")



