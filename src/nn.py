from src.engine import Tensor
import numpy as np

class Linear:
    def __init__(self, nin, nout, device='cpu'):
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin), label='W', device=device)
        self.b = Tensor(np.zeros(nout), label='b', device=device)

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x, device=self.W.device)
        return x @ self.W + self.b
    
    def parameters(self):
        return (self.W, self.b)
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = p.xp.zeros_like(p.data)

class MLP:
    def __init__(self, nin, nout, activation=None, device='cpu'):
        self.layers = []
        self.activation = activation
        for ns in nout:
            layer = Linear(nin, ns, device=device)
            self.layers.append(layer)
            nin = ns
            
            

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x, device=self.layers[0].W.device)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.activation and i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = p.xp.zeros_like(p.data)