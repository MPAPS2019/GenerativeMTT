import torch
from torch import nn

class Lambda(nn.Module):
    """Wraps a callable in an :class:`nn.Module` without registering it."""

    def __init__(self, func):
        super().__init__()
        object.__setattr__(self, 'forward', func)

    def extra_repr(self):
        return getattr(self.forward, '__name__', type(self.forward).__name__) + '()'


class LossList(nn.ModuleList):
    """A weighted combination of multiple loss functions."""

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is not None:
            assert len(weights)==len(losses), f'The lengths of losses and weights should be the same, \
            but got len(weights)={len(weights)}, len(losses)={len(losses)}.'
            self.weights = weights
        else:
            self.weights=[1]*len(losses)

        for loss in losses:
            self.append(loss if isinstance(loss, nn.Module) else Lambda(loss))

    def forward(self, is_output, *args, **kwargs):
        logits = args[0]
        targets = args[1]
        vmask = args[2]
        losses = {}
        loss_tot = 0
        for loss, weight in zip(self, self.weights):
            loss_class = loss.__class__.__name__
            if loss_class == 'VGGloss' and not is_output:
                continue
            if loss_class in ['vMSEloss', 'vL1loss', 'vDICEloss']:
                loss_value = loss(logits, targets, vmask)
            else:
                loss_value = loss(logits, targets)

            if loss_class in ['vMSEloss', 'vL1loss', 'vDICEloss']:
                if loss.mode == 'lt':
                    loss_class+='_vein'
                elif loss.mode == 'st':
                    loss_class+='_artery'

            losses[loss_class]=loss_value
            loss_tot+=loss_value*weight
        losses['total_loss'] = loss_tot
        return losses

class MSEloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria = torch.nn.MSELoss()

    def forward(self, input, target=None):
        return self.criteria(input, target)

class vMSEloss(nn.Module):
    def __init__(self, thr, mode = 'lt'):
        super().__init__()
        assert 0<=thr<=1, f'thr should be in [0, 1].'
        assert mode in ['lt', 'st'], f'mode should be either \'lt\' or \'st\'.'
        self.thr = thr
        self.mode = mode
    def forward(self, input, target=None, vmask = None):
        if self.mode == 'lt':
            mask = (target >= self.thr) * vmask
        elif self.mode == 'st':
            mask = (target <= self.thr) * vmask
        return torch.sum(((input-target)**2)*mask)/torch.sum(mask)


