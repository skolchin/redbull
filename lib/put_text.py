""" Draws text on image with specified alignment """

import cv2
import numpy as np
from typing import Tuple

def put_text_aligned(
        image: np.ndarray, 
        text: str, 
        pt: Tuple[int, int],
        align: str = 'left',
        origin: str = 'bottom',
        width: int = 0,
        pad: int = 0,
        bg_color: Tuple[int,int,int] | None = None,
        *args,
        **kwargs
    ):
    """ Draws text on image with specified alignment and other extras """

    font = args[0] if len(args) > 0 else kwargs.get('fontFace', cv2.FONT_HERSHEY_SIMPLEX)
    scale = args[1] if len(args) > 1 else kwargs.get('fontScale', 1.0)
    thickness = args[2] if len(args) > 2 else kwargs.get('thickness', 1)
    sz = cv2.getTextSize(text, font, scale, thickness)[0]

    match origin:
        case 'bottom':
            pass

        case 'top':
            pad += sz[1]

        case _:
            raise ValueError('Invalid origin', origin)

    match align:
        case 'left':
            tx, ty = pt[0], pt[1] - pad

        case 'center':
            tx, ty = pt[0] + (width or 0) // 2 - sz[0] // 2, pt[1] - pad

        case 'right':
            tx, ty = pt[0] - (width or 0) - sz[0], pt[1] - pad

        case _:
            raise ValueError('Invalid align', align)
        
    if bg_color is not None:
        rx, ry = min(pt[0], tx), min(pt[1], ty)
        rw, rh = max(width or 0, sz[0]), sz[1] + pad
        cv2.rectangle(image, (rx, ry - rh), (rx + rw, ry + pad -1), bg_color, -1)

    return cv2.putText(image, text, (tx, ty), *args, **kwargs)
