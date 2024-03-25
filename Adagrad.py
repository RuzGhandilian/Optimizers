import torch


class AdagradOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(AdagradOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'sum' not in state:
                    state['sum'] = torch.zeros_like(p.data)
                state['sum'] += grad ** 2
                std = state['sum'].sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr)
