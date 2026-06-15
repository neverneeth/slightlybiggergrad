# slightlybiggergrad

A PyTorch-style tensor autodiff engine and deep learning library. 

This project was built as an educational journey to bridge the gap between scalar-based automatic differentiation (like Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd){God Bless that man!!}) and modern, hardware-accelerated, array-agnostic frameworks (like PyTorch and JAX).

`slightlybiggergrad` implements a dynamic, tape-based computational graph that operates directly on **N-Dimensional Tensors** rather than single scalars. By pushing mathematical operations down to optimized C and CUDA kernels using CuPY, it achieves massive speedups while maintaining a clean, pure-Python educational architecture.

I have attempted to create a tutorial for those that want to understand how everything works and build their own engine from scratch. Refer to the [Engineering Log](https://neverneeth.github.io/slightlybiggergrad/) for a deep dive into the design and implementation of `slightlybiggergrad`, along with code snippets and explanations.

### Current Features & Status
* **N-Dimensional Tensors:** Full support for batched matrix operations.
* **Dynamic Autograd Engine:** Implements topological sorting (`backward_all()`) across batched memory blocks.
* **Native Broadcasting:** Automatically handles forward broadcasting and backward "unbroadcasting" (gradient summation across batch dimensions).
* **PyTorch-Style `nn` API:** Includes `Linear` layers and a dynamic Multi-Layer Perceptron (`MLP`) class.
* **Device Dispatching (CPU/GPU):** An array-agnostic hardware registry that strictly enforces device locality. Seamlessly swap between `numpy` (CPU) and `cupy` (NVIDIA GPU).
* **Blazing Fast:** Replaced slow Python `for`-loops with vectorized operations, resulting in a **16x+ speedup** when pushing batched datasets to the GPU.

## Installation

To run on the CPU, you only need NumPy:
```bash
pip install numpy

```

To enable GPU acceleration, install the pre-compiled CuPy binary that matches your NVIDIA CUDA version (e.g., for CUDA 12.x):

```bash
pip install cupy-cuda12x

```

## Quickstart

The API is intentionally designed to mirror PyTorch. Here is how to train a 4-layer MLP on a batched dataset using the GPU:

```python
import numpy as np
from src.engine import Tensor
from src.nn import MLP

# 1. Initialize data on the GPU
device = 'gpu'
X = Tensor(np.random.randn(30000, 200), device=device)
Y = Tensor(np.random.randn(30000, 10), device=device)

# 2. Build a Neural Network
model = MLP(
    nin=200, 
    nout=[150, 100, 50, 10], 
    activation=lambda x: x.tanh(), 
    device=device
)

# 3. The Training Loop
epochs = 1000
learning_rate = 0.0001

for epoch in range(epochs):
    # Flush gradients
    model.zero_grad()
    
    # Forward Pass
    Y_pred = model(X)
    
    # Calculate Mean Squared Error Loss
    loss = ((Y_pred - Y) * (Y_pred - Y)).mean() 
    
    # Backward Pass (Compute gradients)
    loss.backward_all()
    
    # Gradient Descent Step
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data}")

```

## Roadmap / Future Work

This engine currently supports regression tasks and Multi-Layer Perceptrons. Future expansions will implement the exact mathematical primitives required to read and replicate classic Deep Learning papers:

* [ ] **Phase 1: Classification.** Implement `ReLU`, `Softmax`, `CrossEntropyLoss`, and `Reshape` to train on image datasets like MNIST.
* [ ] **Phase 2: Computer Vision.** Implement `im2col`-backed `Conv2d` and `MaxPool2d` to build LeNet and AlexNet architectures.
* [ ] **Phase 3: Modern Optimization.** Abstract the training step into an `Optimizer` class and implement momentum tracking (e.g., `optim.Adam`).
* [ ] **Phase 4: Attention.** Implement N-Dimensional transposing, layer normalization, and masking to build minimal Transformer blocks.

## Acknowledgements

Inspired by Andrej Karpathy's incredible [micrograd](https://github.com/karpathy/micrograd) repository, taking the core calculus concepts and scaling them up to modern hardware paradigms.
