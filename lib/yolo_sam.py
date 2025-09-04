import torch
import numpy as np
from ultralytics import SAM
from typing import Dict, Any

from lib.base import ImageT
from lib.yolo import YoloModel

class SamYoloModel(YoloModel):
    """ Segment Anything model from YOLO (unfinished) """

    DEFAULT_ARGS: Dict[str, Any] = {
        'model': 'weights/sam2.1_l.pt',
        'augment': True,
        'conf': 0.10,
        'iou': 0.1,
        'agnostic_nms': True,
        'imgsz': 640,
        'retina_masks': False,
    }

    def __init__(self, device: torch.device | str | None = 'cuda', args: Dict[str, Any] | None = None):
        args = self.DEFAULT_ARGS.copy() | (args or {})
        model = args.pop('model', self.DEFAULT_ARGS['model'])
        args['model'] = None

        super().__init__(device, args)

        self._model = SAM(model)
        self._model.to(self._device)

    def predict(self, image: ImageT):
        if not self._model.predictor:
            self._model.predict(np.array(image), verbose=False)

        assert self._model.predictor
        self._model.predictor.segment_all = True

        anns = self._model.predict(
            source=np.array(image),
            verbose=False,
            **self._args,
        )
        return anns
