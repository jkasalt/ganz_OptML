import torch

# Inspiration taken from https://github.com/arthurmensch/Variational-Inequality-GAN/blob/master/optim/extragradient.py
class ExtraSGD(torch.optim.Optimizer):
    def __init__(self, params, lr) -> None:
        default = dict(lr=lr)
        super().__init__(params, default)
        self.params_copy = []

    def __setstate__(self, state: dict) -> None:
        return super().__setstate__(state)

    def update_step(self, p, group):
        return -group["lr"] * p.grad.data

    def extrapolate(self):
        """Moves the current parameter to p_tilde and saves the previous to `self.params_copy`"""
        is_empty = len(self.params_copy) == 0
        for group in self.param_groups:
            for p in group["params"]:
                if is_empty:
                    self.params_copy.append(p.data.clone())
                u = self.update_step(p, group)
                p.data.add_(u)

    def step(self, closure=None):
        """Updates the parameters like so:
            p_tilde <- p + lr * grad_p
            p <- p + lr * p_tilde_grad
            """
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call `self.extrapolate` before calling step.')
        loss = None
        if closure is not None:
            loss = closure()
        
        i = -1
        for group in self.param_groups:
            for p in group["params"]:
                i += 1
                u = self.update_step(p, group)
                p.data = self.params_copy[i].add_(u)



        # Clear old parameters
        self.params_copy = []
        return loss
