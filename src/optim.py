class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for param in self.params:
            param.grad = param.xp.zeroes_like(param.data)

    def step(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    

class SGD(Optimizer):
    def __init__(self, params, lr = 0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
    
