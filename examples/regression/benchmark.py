import time
import numpy as np
from src.engine import Tensor
from src.nn import MLP

BATCH_SIZE = 30000
INPUT_DIM = 200
EPOCHS = 1000

def run_benchmark(device):
    print(f"--- Booting on {device.upper()} ---")
    
    
    # 1. We increased the cargo: 30,000 samples!
    X = Tensor(np.random.randn(BATCH_SIZE, INPUT_DIM), device=device)
    Y = Tensor(np.random.randn(BATCH_SIZE, 10), device=device)
    model = MLP(nin=INPUT_DIM, nout=[150, 100, 50, 10], activation=lambda x: x.tanh(), device=device)
    epochs = EPOCHS
    # 2. WARM UP PASS (Do not time this!)
    print("Warming up hardware...")
    warmup_pred = model(X)
    warmup_loss = ((warmup_pred - Y) * (warmup_pred - Y)).sum()
    warmup_loss.backward_all()
    if device == 'gpu':
        import cupy
        cupy.cuda.Stream.null.synchronize()
        
    # 3. THE REAL TEST
    print("Running timed step...")
    
    start_time = time.time()
    for epoch in range(epochs):
    # Forward
        model.zero_grad()
        Y_pred = model(X)
        loss = ((Y_pred - Y) * (Y_pred - Y)).mean()
        
        # Backward
        loss.backward_all()
        
        # Step
        lr = 0.01
        for p in model.parameters():
            p.data -= lr * p.grad
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

    if device == 'gpu':
        cupy.cuda.Stream.null.synchronize()
        
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.4f} seconds\n")

run_benchmark('cpu')
run_benchmark('gpu')