import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results as YoloResults
from typing import List, Dict, Any, Set

from lib.base import Model, ImageT

class YoloModel(Model, YoloResults):
    """ YOLO-base recognition model """

    DEFAULT_ARGS: Dict[str, Any] = {
        'model': "weights/yolo11n.pt",
        'augment': True,
        'conf': 0.10,
        'iou': 0.1,
        'agnostic_nms': True,
        'imgsz': 640,
        'retina_masks': False,
        'classes': '39,42',
    }
    EXCLUDE_ARGS: Set[str] = set(['classes'])

    def __init__(self, device: torch.device | str | None = 'cuda', args: Dict[str, Any] | None = None):
        args = self.DEFAULT_ARGS.copy() | (args or {})
        model = args.pop('model', self.DEFAULT_ARGS['model'])

        super().__init__(device, args)

        if (classes := self._args.get('classes', None)):
            self._classes = [int(x.strip()) for x in classes.split(',')]
        else:
            self._classes = None

        if model:
            self._model = YOLO(model)
            self._model.to(self._device)

    def get_labels(self) -> Dict[int, str]:
        return self._model.names

    def predict(self, image: ImageT):
        return self._model.predict(
            source=np.array(image),
            classes=self._classes,
            verbose=False,
            **self._model_args,
        )

    def draw(self, image: ImageT, ann: YoloResults) -> ImageT:
        return ann.plot(
            img=np.array(image),
            labels=False,
            conf=False,
            probs=False,
            masks=False,
        )

    def count_objects(self, image: ImageT, anns: List[YoloResults]) -> int:
        return len(anns[0]) if anns else 0  #type:ignore
