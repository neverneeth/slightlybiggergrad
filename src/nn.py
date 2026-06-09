from engine import Tensor
import numpy as np

class Linear:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout), label='W')
        self.b = Tensor(np.random.randn(nout), label='b')
    
    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        return x @ self.W + self.b
    
    def parameters(self):
        return (self.W, self.b)
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)