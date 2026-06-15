import numpy as np
import matplotlib.pyplot as plt
import gc 

from src.engine import Tensor
from src.nn import MLP
from src.data import load_california_housing, train_test_split, StandardScaler
from src.optim import BGD, ADAM

X, Y_raw = load_california_housing()
Y = Y_raw / 100000.0  
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

device = 'gpu'  
X_train_tensor = Tensor(X_train, device=device)
Y_train_tensor = Tensor(Y_train, device=device)

EPOCHS = 2000

# BDG PHASE

print("\n--- Training Model 1 with BGD ---")
model1 = MLP(nin=X_train.shape[1], nout=[512, 256, 128, 64, 32, 1], activation=lambda x: x.tanh(), device=device)
BGD_optimizer = BGD(params=model1.parameters(), lr=0.01)
losses1 = []

for epoch in range(EPOCHS):
    model1.zero_grad()
    Y_pred = model1(X_train_tensor)
    loss = ((Y_pred - Y_train_tensor) * (Y_pred - Y_train_tensor)).mean()
    loss.backward_all()
    BGD_optimizer.step()
    losses1.append(float(loss.data))

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss (BGD): {losses1[-1]:.4f}")
        
    # Nuke everything to avoid CUDA OOM    
    del Y_pred, loss
    gc.collect()
    if device == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()

X_test_tensor = Tensor(X_test, device=device)
Y_test_tensor = Tensor(Y_test, device=device)
Y_test_pred = model1(X_test_tensor)
test_loss_bgd = float(((Y_test_pred - Y_test_tensor) * (Y_test_pred - Y_test_tensor)).mean().data)

# Nuke everything else to avoid CUDA OOM for the next phase
del model1, BGD_optimizer, Y_test_pred
gc.collect()
if device == 'gpu':
    cp.get_default_memory_pool().free_all_blocks()

# ADAM PHASE

print("\n--- Training Model 2 with ADAM ---")
model2 = MLP(nin=X_train.shape[1], nout=[512, 256, 128, 64, 32, 1], activation=lambda x: x.tanh(), device=device)
ADAM_optimizer = ADAM(params=model2.parameters(), lr=0.001)
losses2 = []

for epoch in range(EPOCHS):
    model2.zero_grad()
    Y_pred2 = model2(X_train_tensor)
    loss2 = ((Y_pred2 - Y_train_tensor) * (Y_pred2 - Y_train_tensor)).mean()
    loss2.backward_all()
    ADAM_optimizer.step()
    losses2.append(float(loss2.data))

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss (ADAM): {losses2[-1]:.4f}")
        
    # Once again nuke everything to avoid CUDA OOM
    del Y_pred2, loss2
    gc.collect()
    if device == 'gpu':
        cp.get_default_memory_pool().free_all_blocks()

Y_test_pred2 = model2(X_test_tensor)
test_loss_adam = float(((Y_test_pred2 - Y_test_tensor) * (Y_test_pred2 - Y_test_tensor)).mean().data)


print("\n--- Final Results ---")
print(f"Test Loss (BGD): {test_loss_bgd:.4f}")
print(f"Test Loss (ADAM): {test_loss_adam:.4f}")

plt.plot(range(EPOCHS), losses1, label='BGD', color='blue')
plt.plot(range(EPOCHS), losses2, label='ADAM', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs: BGD vs. ADAM')
plt.legend()
plt.show()