import torch
from typing import List, Callable, Optional
import ttach as tta
import numpy as np

from pytorch_grad_cam.grad_cam import BaseCAM
from .activations_and_gradients import ActivationsAndGradients

class BaseCAM_SemSeg(BaseCAM):
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 tta_transforms: Optional[tta.Compose] = None) -> None:

        super(BaseCAM_SemSeg, self).__init__(model, target_layers, reshape_transform,
                                          compute_input_gradient, uses_gradients, tta_transforms)

        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        # input_tensor = input_tensor.to(self.device)
        input_tensor = input_tensor


        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        self.outputs = outputs = outputs[-1]


        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)


        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

