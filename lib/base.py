import cv2
import torch
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from abc import ABC, abstractmethod
from typing import Set, List, TypeVar, Generic, Dict, Any

from lib.put_text import put_text_aligned

ImageT = npt.NDArray[np.uint8] | Image.Image
ResultsT = TypeVar('ResultsT')

class Model(ABC, Generic[ResultsT]):
    """ Base recognition model """

    DEFAULT_ARGS: Dict[str, Any] = {}
    """ Default arguments for actual model or models """

    INCLUDE_ARGS: Set[str] = set()
    """ Keys to include when model_args are prepared (empty - all keys) """

    EXCLUDE_ARGS: Set[str] = set()
    """ Keys to exclude when model_args are prepared """

    NEEDS_PILLOW: bool = False
    """ True if model expects PIL Image, False - numpy arrays """

    def __init__(self, device: torch.device | str | None = 'cuda', args: Dict[str, Any] | None = None):
        self._device: torch.device = torch.device(device or 'cuda')
        self._args: Dict[str, Any] = self.DEFAULT_ARGS.copy()
        self._model_args: Dict[str, Any] = {}

        include_args = self.INCLUDE_ARGS.copy() or set(self._args.keys())
        exclude_args = self.EXCLUDE_ARGS.copy() or set()

        for k, v in (args or {}).items():
            assert k in self._args, f'Unknown model argument: {k}'
            self._args[k] = type(self._args[k])(v)

            if k in include_args and not k in exclude_args:
                self._model_args[k] = self._args[k]

    @abstractmethod
    def predict(self, image: ImageT) -> List[ResultsT]:
        """ Handle the image """
        ...

    @abstractmethod
    def draw(self, image: ImageT, ann: ResultsT) -> ImageT:
        """ Draw single recognition result """
        ...

    @abstractmethod
    def count_objects(self, image: ImageT, anns: List[ResultsT]) -> int:
        """" Count objects from recognition results """
        ...

    def _show_object_count(self, image: ImageT, count: int) -> ImageT:
        """ Draw object count on the image """
        match(image):
            case np.ndarray():
                put_text_aligned(
                    image,
                    f'{count} objects detected',
                    (10, 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0,255,0),
                    # bg_color=(255,255,255),
                    thickness=2,
                    pad=4,
                )
                return image
            
            case Image.Image():
                draw = ImageDraw.Draw(image)
                draw.text(
                    (10,20), 
                    f'{count} objects detected',
                    fill=(255,0,0),
                    font_size=20,
                )
                return image
            

    def draw_all(self, image: ImageT, anns: List[ResultsT]) -> ImageT:
        """ Draw all recognition results on image """
        image_copy = image.copy()
        for a in anns:
            image_copy = self.draw(image_copy, a)

        count = self.count_objects(image, anns)
        image_copy = self._show_object_count(image_copy, count)

        return image_copy
