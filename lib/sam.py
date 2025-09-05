import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List, Set, Dict, Any, Tuple

from lib.base import Model, ImageT

def segment_image(image: Image.Image, segmentation_mask) -> Image.Image:
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def show_anns(image, anns):
    original_image = image.copy()
    overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = tuple((np.concatenate([np.random.random(3), [0.35]]) * 255).astype('int').tolist())
        segmentation_mask_image = Image.fromarray(m.astype('uint8') * 255)
        draw.bitmap((0, 0), segmentation_mask_image, fill=color_mask)

    return Image.alpha_composite(original_image.convert('RGBA'), overlay_image)


def convert_box_xywh_to_xyxy(
        box: List[int], 
        max_sz: Tuple[int,int],
        cap: int = 0) -> Tuple[int, int, int, int]:

    x1 = max(box[0]-cap, 0)
    y1 = min(box[1]-cap, 0)
    x2 = min(box[0] + box[2] + cap, max_sz[0])
    y2 = min(box[1] + box[3] + cap, max_sz[1])
    return (x1, y1, x2, y2)

SamResultT = Dict[str, Any]

class SamModel(Model, SamResultT):
    """ Meta Segment Anything model (unfinished) """

    DEFAULT_ARGS: Dict[str, Any] = {
        'sam_checkpoint': 'weights/sam_vit_h_4b8939.pth',
        'model_type': "vit_h",
        'conf': 0.2,
        'cap': 10,
        'iou': 0.7,
        'points_per_side': 32,
        'points_per_batch': 64,
        'pred_iou_thresh': 0.99,
        'stability_score_thresh': 0.95,
        'stability_score_offset': 1,
        'box_nms_thresh': 0.8,
        'crop_n_layers': 1,
        'crop_nms_thresh': 0.8,
        'crop_overlap_ratio': 512 / 1500,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 500,
    }

    EXCLUDE_ARGS: Set[str] = set([
        'sam_checkpoint',
        'model_type',
        'conf',
        'cap',
        'iou',
    ])

    NEEDS_PILLOW: bool = True

    def __init__(self, device: torch.device | str | None = 'cuda', args: Dict[str, Any] | None = None):
        args = self.DEFAULT_ARGS.copy() | (args or {})
        sam_checkpoint = args['sam_checkpoint']
        model_type  = args['model_type']

        super().__init__(device, args)

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self._mask_generator = SamAutomaticMaskGenerator(sam, **self._model_args)

        self._yolo_model = YOLO('weights/yolo11n.pt')

    def predict(self, image: ImageT):
        masks = self._mask_generator.generate(np.array(image))
        return masks

    def draw(self, image: ImageT, ann: SamResultT) -> ImageT:
        return image

    def draw_all(self, image: ImageT, anns: List[SamResultT]) -> ImageT:

        assert isinstance(image, Image.Image)

        object_anns = []
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for n, ann in enumerate(sorted_anns):
            m = ann['segmentation']
            bbox = convert_box_xywh_to_xyxy(ann["bbox"], image.size, self._args['cap'])

            found = False
            cropped_image = segment_image(image, m).crop(bbox)
            imgsz = max(32 * ((max(cropped_image.size) - 1) // 32 + 1), 640)
            if imgsz > 64:
                # cv2.imshow(str(n), np.array(cropped_image))
                results = self._yolo_model.predict(
                    source=cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR),
                    classes=[39],
                    verbose=False,
                    augment=False,
                    retina_masks=False,
                    imgsz=imgsz,
                    conf=self._args['conf'],
                )
                for r in results:
                    # print(r.summary())
                    if r.summary():
                        # cv2.imshow(str(n), np.array(cropped_image))
                        object_anns.append(ann)
                        found = True
                        break
            # if found:
            #     break

        # print('Waiting for any key')
        # cv2.waitKey(0)

        self._last_object_count = len(object_anns)
        segmentation_masks = []

        for ann in object_anns:
            segmentation_mask_image = Image.fromarray(ann["segmentation"].astype('uint8') * 255)
            segmentation_masks.append(segmentation_mask_image)

        original_image = image.copy()
        overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_color = (255, 0, 0, 200)

        draw = ImageDraw.Draw(overlay_image)
        for segmentation_mask_image in segmentation_masks:
            draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

        drawn_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)
        drawn_image = self._show_object_count(drawn_image, self._last_object_count)
        return drawn_image

    def count_objects(self, image: ImageT, anns: List[SamResultT]) -> int:
        return self._last_object_count

