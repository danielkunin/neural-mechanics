import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        denom = lr * (1 - dampening) * (1 + momentum)
        self.gamma = (1 - momentum) / denom
        self.omega = np.sqrt(4 * weight_decay / denom)

        if self.gamma < self.omega:
            sqrt = np.sqrt(self.omega**2 - self.gamma**2)
            def scale(time):
                scale_1 = np.exp(self.gamma * time) * np.cos(sqrt * time)
                scale_2 = np.exp(self.gamma * time) * np.sin(sqrt * time)
                return (scale_1, scale_2)

        elif self.gamma == self.omega:
            def scale(time):
                scale_1 = np.exp(self.gamma * time)
                scale_2 = np.exp(self.gamma * time) * time
                return (scale_1, scale_2)

        else:
            sqrt = np.sqrt(self.gamma**2 - self.omega**2)
            alpha_p = -self.gamma + sqrt
            alpha_m = -self.gamma - sqrt
            def scale(time):
                scale_1 = np.exp(-alpha_p * time)
                scale_2 = np.exp(-alpha_m * time)
                return (scale_1, scale_2)
        self.scale = scale


    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                        buf.mul_(1 - dampening) # Added to scale buffer appropriately
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)


                # Should we be using d_p or g?
                param_state = self.state[p]
                g = p.grad
                if "step" not in param_state:
                    param_state["step"] = 0
                    scale_1, scale_2 = self.scale(0)
                    param_state["integral_buffer_1"] = scale_1 * d_p ** 2
                    param_state["integral_buffer_2"] = scale_2 * d_p ** 2
                else:
                    param_state["step"] += 1
                    time = lr * (1 - dampening) * param_state["step"]
                    scale_1, scale_2 = self.scale(time)
                    param_state["integral_buffer_1"].add_(scale_1 * d_p ** 2)
                    param_state["integral_buffer_2"].add_(scale_2 * d_p ** 2)

        return loss
