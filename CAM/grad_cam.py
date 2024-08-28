import numpy as np
from .base_cam import BaseCAM_SemSeg


class GradCAM(BaseCAM_SemSeg):
    def __init__(self, model, target_layers,compute_input_gradient,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
            compute_input_gradient,)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))