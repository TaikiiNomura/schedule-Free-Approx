from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

class AdinaSchedulefreeReference(torch.optim.Optimizer):
    def __init__(self,
                params: ParamsT,
                lr: Union[float, torch.Tensor] = 0.001,
                betas: Tuple[float, float] = (0.9, 0.999),
                eps: float = 1e-8,
                a: float = 0.1,
                b: float = 0.9,
                sf: float = 0.9,
                ):
        
        defaults = dict(
            lr = lr,
            betas=betas,
            eps = eps,
            a = a,
            b = b,
            k=0,
            train_mode=False,
            sf=sf,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        p.copy_(state['x'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.copy_(state['y'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group['betas']
            eps = group["eps"]
            a = group["a"]
            b = group["b"]
            k = group['k']
            sf = group['sf']

            bias_correction1 = 1 - beta1 ** (k + 1)
            bias_correction2 = 1 - beta2 ** (k + 1)
            C = (1 - a * b) / a
            ckp1 = 1 / (k + 1)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['x'] = torch.clone(p, memory_format=torch.preserve_format)
                    state['y'] = torch.clone(p, memory_format=torch.preserve_format)

                m = state['m']
                v = state['v']
                z = state['z']
                x = state['x']
                y = state['y']

                # m update
                m.mul_(beta1).add_(grad * C, alpha=(1-beta1))
                m_hat = m.div(bias_correction1)

                # v update
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                v_hat = v.div(bias_correction2).sqrt_().add_(eps)

                # z update
                z.addcdiv_(m_hat + b * grad, v_hat, value=-lr)

                # x update
                x.mul_(1-ckp1).add_(z, alpha=ckp1)

                # y update
                y.copy_(x.mul(sf).add_(z, alpha=1-sf))

                # update parameter
                p.copy_(y)

            group['k'] = k + 1
        return loss
