import torch


class RMSPropOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, decay_rate=0.9, eps=1e-8):
        defaults = dict(lr=lr, decay_rate=decay_rate, eps=eps)
        super(RMSPropOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            decay_rate = group['decay_rate']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'square_avg' not in state:
                    state['square_avg'] = torch.zeros_like(p.data)
                state['square_avg'].mul_(decay_rate).addcmul_(grad, grad, value=1 - decay_rate)
                std = state['square_avg'].sqrt().add_(eps)
                p.data.addcdiv_(grad, std, value=-lr)
