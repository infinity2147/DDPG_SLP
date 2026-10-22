"""Component A: output-layer update clipping, projection (A1 / A1-all) and decoupled weight decay (A2).

All functions act on one critic MLP (an nn.Sequential whose last Linear is the output layer).
A "layer" always means weight and bias together, treated as one vector.
"""
import torch
from .networks import linear_layers


def layer_norm(layer):
    return torch.sqrt(layer.weight.pow(2).sum() + layer.bias.pow(2).sum())


def clip_grad(layer, lam):
    """SGD variant (faithful to the theory): g <- g * min(1, lam/||g||). Returns True if clipped."""
    g = torch.sqrt(layer.weight.grad.pow(2).sum() + layer.bias.grad.pow(2).sum())
    if g > lam:
        layer.weight.grad.mul_(lam / g)
        layer.bias.grad.mul_(lam / g)
        return True
    return False


@torch.no_grad()
def clip_update(layer, old, max_norm):
    """Adam variant. Adam rescales gradients, so clipping the gradient would not bound the step.
    Instead clip the realised parameter delta: Delta = theta_new - theta_old, ||Delta|| <= max_norm (= lr*lam).
    `old` = (weight, bias) copies taken before optimizer.step(). Returns True if clipped."""
    dw, db = layer.weight - old[0], layer.bias - old[1]
    n = torch.sqrt(dw.pow(2).sum() + db.pow(2).sum())
    if n > max_norm:
        s = max_norm / n
        layer.weight.copy_(old[0] + s * dw)
        layer.bias.copy_(old[1] + s * db)
        return True
    return False


@torch.no_grad()
def project(layer, radius):
    """A1: rescale the layer onto the ball of the given radius if it lies outside."""
    n = layer_norm(layer)
    if n > radius:
        layer.weight.mul_(radius / n)
        layer.bias.mul_(radius / n)


@torch.no_grad()
def decay(layer, rate):
    """A2: decoupled l2 weight decay, theta <- (1 - rate) * theta, applied AFTER clipping
    (so the decay itself is never clipped away; see DECISIONS.md)."""
    layer.weight.mul_(1 - rate)
    layer.bias.mul_(1 - rate)


class CriticStabiliser:
    """Wraps one critic optimiser step with Component A.

    clip:       lambda for output-layer clipping (None = off)
    stab:       'none' | 'proj' (A1) | 'proj_all' (A1-all) | 'wd' (A2: output-layer decay) | 'wd_all'
    R_mult:     projection radius as a multiple of each layer's norm at initialisation
    wd:         decay coefficient (per unit learning rate)
    """

    def __init__(self, nets, opt, lr, sgd, clip=None, stab="none", R_mult=10.0, wd=0.0):
        self.nets, self.opt, self.lr, self.sgd = nets, opt, lr, sgd
        self.clip, self.stab, self.wd = clip, stab, wd
        self.radii = [[R_mult * layer_norm(l).item() for l in linear_layers(n)] for n in nets]

    def step(self):
        """Call after loss.backward(). Returns the fraction of heads whose output update was clipped."""
        outs = [linear_layers(n)[-1] for n in self.nets]
        clipped = 0
        if self.clip is not None and self.sgd:
            clipped = sum(clip_grad(o, self.clip) for o in outs)
        old = [(o.weight.detach().clone(), o.bias.detach().clone()) for o in outs]
        self.opt.step()
        if self.clip is not None and not self.sgd:
            clipped = sum(clip_update(o, prev, self.lr * self.clip) for o, prev in zip(outs, old))
        for n, radii in zip(self.nets, self.radii):
            layers = linear_layers(n)
            if self.stab == "proj":
                project(layers[-1], radii[-1])
            elif self.stab == "proj_all":
                for l, r in zip(layers, radii):
                    project(l, r)
            elif self.stab == "wd":
                decay(layers[-1], self.lr * self.wd)
            elif self.stab == "wd_all":
                for l in layers:
                    decay(l, self.lr * self.wd)
        return clipped / len(outs)
