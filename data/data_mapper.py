from __future__ import annotations

import cv2
import numpy as np
import matplotlib.image as mpimg
from typing import List
from detectron2.utils import numpy_to_tensor
from detectron2.projects.segmentation.data import ImageSample
from detectron2.projects.segmentation.transforms import Transform, TransformList

class GnrtMTTDataMapper:
    def __init__(self,
                 transforms: List[Transform],):

        self.transforms = TransformList(transforms)

    def __call__(self, data: dict) -> ImageSample:
        RGB = (mpimg.imread(data['RGB_image_path'])/255).astype(np.float32)
        RGB = np.transpose(RGB, axes=(2,0,1))
        try:
            LSI = (mpimg.imread(data['LSI_30_image_path'])[None]/255).astype(np.float32)
        except:
            LSI = (mpimg.imread(data['LSI_MS_30_image_path'])[None]/255).astype(np.float32)

        MTT = (mpimg.imread(data['MTT_image_path'])[None]/255).astype(np.float32)
        vmask = (mpimg.imread(data['vessel_mask_path'])[None] / 255)>0.5
        img_name = data['case_name']

        image = np.concatenate([LSI, RGB], 0)
        # image = RGB
        label = np.concatenate([MTT,vmask],0)

        sample = ImageSample(
            img_name = img_name,
            image = image,
            label = label,
        )

        if len(self.transforms)>0:
            self.transforms(sample)


        return ImageSample(
            img_name=sample.img_name,
            image=numpy_to_tensor(sample.image),
            label=numpy_to_tensor(sample.label)
        )




