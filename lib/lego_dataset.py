# source: https://www.kaggle.com/datasets/martinellis/synthetic-lego-images/

import cv2
import json
import torch
import logging
import numpy as np
import pandas as pd
import albumentations as A
from pathlib import Path
from numpy.typing import NDArray
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from typing import Dict, Literal, Tuple, Annotated, TypedDict, Callable, cast

_logger = logging.getLogger(__name__)

FloatImageTensor = Annotated[ torch.Tensor, torch.float32, Literal[3, 'H', 'W'] ]
IntImageTensor = Annotated[ torch.Tensor, torch.int64, Literal[3, 'H', 'W'] ]
IntMaskTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 'H', 'W'] ]
BoxesTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 4] ]
AreaTensor = Annotated[ torch.Tensor, torch.float32, Literal['N'] ]
IsCrowdTensor = Annotated[ torch.Tensor, torch.int64, Literal['N'] ]
LabelsArray = Annotated[ NDArray[np.int_], Literal['N'] ]
ColorArray = Annotated[ NDArray[np.uint8], Literal[3] ]

ImagePreprocessCallable = Callable[..., FloatImageTensor]
""" Will be called with image=... kwarg """

ImageWithMasksPreprocessCallable = Callable[..., Tuple[FloatImageTensor, IntMaskTensor]]
""" Will be called with image=... mask=... kwargs """

class TargetDict(TypedDict):
    """ Target description for images """
    image_id: int
    boxes: BoxesTensor
    labels: LabelsArray
    area: AreaTensor
    iscrowd: IsCrowdTensor
    image_fn: str
    mask_fn: str | None

class SynthentcLegoImagesDatasetBase(Dataset):
    """ Base class """

    CLASSES_FILE: str = 'classes.csv'
    ANNOTATION_FILE: str = 'annotation-data.csv'

    _splits: Dict[str, NDArray[np.int_]] | None = None
    _annotations: pd.DataFrame | None

    def __new__(cls, *args, **kwargs):

        if cls._splits is None:
            data_root = Path(kwargs.get('data_root', args[0]))
            cls._annotations = pd.read_csv(data_root / cls.ANNOTATION_FILE)
            ann_ids_df = cls._annotations.drop_duplicates('img_id')

            if 'frac' in kwargs:
                valid_ids_df = ann_ids_df.sample(frac=kwargs['frac'], replace=False)
            elif 'n_samples' in kwargs:
                valid_ids_df = ann_ids_df.sample(n=kwargs['n_samples'], replace=False)
            else:
                valid_ids_df = ann_ids_df.sample(n=100, replace=False)

            valid_ids = valid_ids_df['img_id'].unique()
            train_ids = ann_ids_df[ ~ann_ids_df['img_id'].isin(valid_ids) ]['img_id'].unique()
            cls._splits = {
                'train': train_ids,
                'valid': valid_ids
            }
        return super().__new__(cls)

    def __init__(self,
                 data_root: Path | str,
                 split: Literal['train', 'valid'] | None = None):

        super().__init__()

        assert self._splits is not None
        assert self._annotations is not None

        self.data_root = Path(data_root)
        self.annotations = self._annotations
        self.split = split
        self.image_ids = self._splits[split] if split else self.annotations['img_id'].unique()

        classes_df = pd.read_csv(self.data_root / self.CLASSES_FILE)
        self.classes = pd.Series(classes_df['label_name'].astype(str), index=classes_df['label_id'].astype(int)).to_dict()
        self.class_names = pd.Series(classes_df['name'].astype(str), index=classes_df['label_id'].astype(int)).to_dict()

        self.transforms: A.Compose | None = None
        match split:
            case 'train':
                self.transforms = self.get_transform_train()
            case 'valid':
                self.transforms = self.get_transform_valid()

    def __len__(self) -> int:
        """ Length """
        return self.image_ids.shape[0]

    @classmethod
    def get_transform_train(cls) -> A.Compose:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0)
        ], bbox_params={'format':'pascal_voc', 'label_fields': ['labels']})

    @classmethod
    def get_transform_valid(cls) -> A.Compose:
        return A.Compose([
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})


class SynthentcLegoImagesDataset(SynthentcLegoImagesDatasetBase):
    """ Dataset object for Synthetic Lego images """

    def __init__(self,
                 data_root: Path | str,
                 split: Literal['train', 'valid'] | None = None,
                 preprocess_fn: ImagePreprocessCallable | None = None):

        super().__init__(data_root, split)
        self.preprocess_fn = preprocess_fn

    def _get(self, index: int | str) -> Tuple[FloatImageTensor, TargetDict]:
        """ Get implementation """
        image_id = self.image_ids[index] if isinstance(index, int) else index
        records = self.annotations[self.annotations['img_id'] == image_id]
        if records.empty:
            raise KeyError(index)

        image_fn = self.data_root / image_id
        image = cv2.imread(str(image_fn), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_fn)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mine are different, heithg & width not xmax & ymax
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.int64)

        labels = cast(LabelsArray, records['labels'].values)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = TargetDict(
            image_id=int(records.index.min()),
            boxes=boxes,
            labels=labels,
            area=area,
            iscrowd=iscrowd,
            image_fn=str(image_fn),
            mask_fn=None,
        )

        if not self.transforms:
            # simply cast from uint to float
            image = torch.as_tensor(image).to(torch.float32) / 255.0
        else:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)

        if self.preprocess_fn:
            image = self.preprocess_fn(image=image)

        return image, target

    def get(self, index: int | str) -> Tuple[FloatImageTensor, TargetDict] | None:
        """ Get an entry """
        try:
            return self._get(index)
        except (KeyError, FileNotFoundError):
            return None

    def __getitem__(self, index: int | str) -> Tuple[FloatImageTensor, TargetDict]:
        """ Indexing """
        return self._get(index)

    def __iter__(self):
        """ Iteration """
        for i in range(len(self)):
            yield self[i]

class SynthentcLegoImagesWithMasksDataset(SynthentcLegoImagesDatasetBase):
    """ Dataset object for Synthetic Lego images with segmentation masks """

    ANNOTATION_DEFS_FILE: str = 'annotation-jsons/annotation_definitions.json'
    SEMANTIC_SEGMENTATION_DIR: str = 'semantic-segmentation'

    def __init__(self,
                 data_root: Path | str,
                 split: Literal['train', 'valid'] | None = None,
                 preprocess_fn: ImageWithMasksPreprocessCallable | None = None,
                 remove_missing_labels: bool = False):
        
        super().__init__(data_root, split)
        self.preprocess_fn = preprocess_fn
        self.remove_missing_labels = remove_missing_labels

        ann_defs = json.loads((self.data_root / self.ANNOTATION_DEFS_FILE).read_text())
        segm_defs = [x for x in ann_defs['annotation_definitions'] if x['name'] == 'semantic segmentation']
        assert segm_defs, f'Cannot find semantic segmentation definition in {self.data_root / self.ANNOTATION_DEFS_FILE} file'

        def to_clr(pixel_value: Dict) -> ColorArray:
            return (np.array([pixel_value['r'], pixel_value['g'], pixel_value['b']], dtype=np.float32) * 255).round(0).astype(np.uint8)

        label_to_id_mapping = {v: k for k, v in self.classes.items()}
        self.class_colors = {label_to_id_mapping[spec['label_name']]: to_clr(spec['pixel_value']) for spec in segm_defs[0]['spec']}

    def _get(self, index: int | str) -> Tuple[FloatImageTensor, IntMaskTensor, TargetDict]:
        """ Get implementation """
        image_id = self.image_ids[index] if isinstance(index, int) else index
        records = self.annotations[self.annotations['img_id'] == image_id]
        if records.empty:
            raise KeyError(index)

        image_fn = self.data_root / image_id
        image = cv2.imread(str(image_fn), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_fn)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_fn = image_fn.parent.parent / self.SEMANTIC_SEGMENTATION_DIR / image_id.split('/')[-1].replace('rgb', 'segmentation')
        mask_image = cv2.imread(str(mask_fn), cv2.IMREAD_COLOR)
        if mask_image is None:
            raise FileNotFoundError(mask_fn)

        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        # mine are different, heigth & width not xmax & ymax
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.as_tensor(boxes, dtype=torch.int64)

        labels = cast(LabelsArray, records['labels'].values)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        # parse mask by colors
        masks = []
        for lbl, clr in self.class_colors.items():
            empty = torch.zeros(mask_image.shape[:2], dtype=torch.int64)
            if lbl in labels:
                clr_mask = (mask_image == clr)
                object_mask = clr_mask.all(axis=-1).reshape(mask_image.shape[:2])
                if object_mask.any():
                    empty[object_mask] = 1
                elif self.remove_missing_labels:
                    _logger.warning(f'Cannot find segmentation mask for label {lbl} on {mask_fn}, removing label from target')
                    boxes = boxes[ labels != lbl ]
                    labels = labels[ labels != lbl ]
                else:
                    _logger.warning(f'Cannot find segmentation mask for label {lbl} on {mask_fn}')
            
            masks.append(empty)

        masks = np.stack(masks, axis=0)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = TargetDict(
            image_id=int(records.index.min()),
            boxes=boxes,
            labels=labels,
            area=area,
            iscrowd=iscrowd,
            image_fn=str(image_fn),
            mask_fn=str(mask_fn),
        )

        if not self.transforms:
            # simply cast from uint to float
            image = torch.as_tensor(image).to(torch.float32) / 255.0
            masks = torch.as_tensor(masks)
        else:
            sample = {
                'image': image,
                'masks': masks,
                'bboxes': target['boxes'],
                'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            masks = sample['masks']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)

        if self.preprocess_fn:
            image, masks = self.preprocess_fn(image=image, mask=masks)

        return image, masks, target

    def get(self, index: int | str) -> Tuple[FloatImageTensor, IntMaskTensor, TargetDict] | None:
        """ Get an entry """
        try:
            return self._get(index)
        except (KeyError, FileNotFoundError):
            return None

    def __getitem__(self, index: int | str) -> Tuple[FloatImageTensor, IntMaskTensor, TargetDict]:
        """ Indexing """
        return self._get(index)

    def __iter__(self):
        """ Iteration """
        for i in range(len(self)):
            yield self[i]
