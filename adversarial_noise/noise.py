from typing import Optional, Any
import torch
from torch import nn

__all__ = ['FGSM', 'PDG']

#TODO: support GPU

class FGSM:
    def __init__(self, model: nn.Module, targeted: bool = True, loss: nn.Module = nn.CrossEntropyLoss(), clip_min: Optional[float] = None, clip_max: Optional[float] = None):
        self.model = model
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss = loss

    def __call__(self, x: torch.Tensor, target: int, eps: float) -> torch.Tensor:
        input_ = x.clone().cpu().detach()
        input_.requires_grad_()

        logits = self.model(input_)
        target = torch.LongTensor([target]).to(logits.device)
        self.model.zero_grad()
        loss = self.loss(logits, target)
        loss.backward()

        if self.targeted:
            out = input_ - eps * input_.grad.sign()
        else:
            out = input_ + eps * input_.grad.sign()

        if (self.clip_min is not None) or (self.clip_max is not None):
            out.clamp_(min=self.clip_min, max=self.clip_max)
        return out


class PDG(FGSM):
    
    def __init__(self, model: nn.Module, targeted: bool = True, loss: nn.Module = nn.CrossEntropyLoss(), clip_min: Optional[float] = None, clip_max: Optional[float] = None):
        super().__init__(model=model, targeted=targeted, loss=loss, clip_min=clip_min, clip_max=clip_max)
    
    def __call__(self, x: torch.Tensor, target: int, k: int, eps: float) -> Any:
        x=x.cpu()
        x_min = x - eps
        x_max = x + eps

        x_adv = x + eps * (2 * torch.rand_like(x) - 1)

        if (self.clip_min is not None) or (self.clip_max is not None):
            x_adv.clamp_(min=self.clip_min, max=self.clip_max)
        for _ in range(k):
            x_adv = super().__call__(x_adv, target, eps)
            x_adv = torch.min(x_max, torch.max(x_min, x_adv))

        if (self.clip_min is not None) or (self.clip_max is not None):
            x_adv.clamp_(min=self.clip_min, max=self.clip_max)
        return x_adv
