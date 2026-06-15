class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for param in self.params:
            param.grad = param.xp.zeros_like(param.data)

    def step(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    

class BGD(Optimizer):
    def __init__(self, params, lr = 0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad
    
class ADAM(Optimizer):
    def __init__(self, params, lr=0.001, beta1 = 0.9, beta2 = 0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, epoch):
        for param in self.params:
            if not hasattr(param, 'm'):
                param.m = param.xp.zeros_like(param.data)
                param.v = param.xp.zeros_like(param.data)
                param.t = 0
            
            param.t += 1
            param.m = self.beta1 * param.m + (1 - self.beta1) * param.grad
            param.v = self.beta2 * param.v + (1 - self.beta2) * (param.grad ** 2)

            m_hat = param.m / (1 - self.beta1 ** param.t)
            v_hat = param.v / (1 - self.beta2 ** param.t)

            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def step(self, epoch):
        self(epoch)
