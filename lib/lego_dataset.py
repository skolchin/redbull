# source: https://www.kaggle.com/datasets/martinellis/synthetic-lego-images/

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from pathlib import Path
from numpy.typing import NDArray
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from typing import Dict, Literal, Tuple, Annotated, TypedDict, Union, cast

FloatImageTensor = Annotated[ torch.Tensor, torch.float32, Literal[3, 'H', 'W'] ]
IntImageTensor = Annotated[ torch.Tensor, torch.int64, Literal[3, 'H', 'W'] ]
IntMaskTensor = Annotated[ torch.Tensor, torch.int64, Literal['H', 'W'] ]
BoxesTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 4] ]
AreaTensor = Annotated[ torch.Tensor, torch.float32, Literal['N'] ]
IsCrowdTensor = Annotated[ torch.Tensor, torch.int64, Literal['N'] ]
LabelsArray = Annotated[ NDArray[np.int_], Literal['N'] ]
ColorArray = Annotated[ NDArray[np.int_], Literal[4] ]
ColorListTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 3] ]

class TargetDict(TypedDict):
    image_id: int
    boxes: BoxesTensor
    labels: LabelsArray
    mask: FloatImageTensor | None
    area: AreaTensor
    iscrowd: IsCrowdTensor

class SynthentcLegoImagesDataset(Dataset):
    """ Dataset object for Synthetic Lego images.

    Supports indexing, both sequental and by image name (e.g. [`images/rgb_81.png`])
    """

    CLASSES_FILE: str = 'classes.csv'
    ANNOTATION_FILE: str = 'annotation-data.csv'
    SEMANTIC_SEGMENTATION_DIR: str = 'semantic-segmentation'

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
                 split: Literal['train', 'valid'] | None = None,
                 transforms: A.Compose | None = None,
                 with_masks: bool = False,
                 cache_size: int = 5):

        super().__init__()

        assert self._splits is not None
        assert self._annotations is not None

        self.data_root = Path(data_root)
        self.annotations = self._annotations
        self.split = split
        self.image_ids = self._splits[split] if split else self.annotations['img_id'].unique()

        classes_df = pd.read_csv(self.data_root / self.CLASSES_FILE)
        self.classes = pd.Series(classes_df.name.astype(str),index=classes_df.label_id.astype(int)).to_dict()

        self.transforms = transforms
        self.with_masks = with_masks
        self.cache_size = 5

    def get(self, index: int | str) -> Tuple[FloatImageTensor, TargetDict, Union['SegmentationMask',None]] | None:
        """ Get an entry. 
        
        Returns image tensor, targed definition and mask tensor (if `with_masks` was set to `True) or None` 
        """
        try:
            return self[index]
        except (KeyError, FileNotFoundError):
            return None

    def __getitem__(self, index: int | str) -> Tuple[FloatImageTensor, TargetDict, 'SegmentationMask']:
        """ Get implementatio"""
        image_id = self.image_ids[index] if isinstance(index, int) else index
        records = self.annotations[self.annotations['img_id'] == image_id]
        if records.empty:
            raise KeyError(index)

        image_fn = self.data_root / image_id
        image = cv2.imread(str(image_fn), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_fn)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_image = None
        if self.with_masks:
            mask_fn = image_fn.parent.parent / self.SEMANTIC_SEGMENTATION_DIR / image_id.split('/')[-1].replace('rgb', 'segmentation')
            mask_image = cv2.imread(str(mask_fn), cv2.IMREAD_COLOR)
            if mask_image is None:
                raise FileNotFoundError(mask_fn)

            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

        # mine are different, heithg & width not xmax & ymax
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = records['labels'].values
        # labels = torch.as_tensor(label, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target: TargetDict = TargetDict(
            image_id=int(records.index.min()),
            boxes=cast(BoxesTensor, torch.as_tensor(boxes)),
            labels=cast(LabelsArray, labels),
            mask=None,
            area=cast(AreaTensor, area),
            iscrowd=cast(IsCrowdTensor, iscrowd),
        )

        if not self.transforms:
            # simply cast from uint to float
            image = torch.as_tensor(image).to(torch.float32) / 255.0
            if mask_image is not None:
                mask_image = torch.as_tensor(mask_image)
        else:
            sample = {
                'image': image,
                'mask': mask_image,
                'bboxes': target['boxes'],
                'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            mask_image = sample['mask']

            target['boxes'] = cast(BoxesTensor, torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0))

        return cast(FloatImageTensor, image), target, SegmentationMask(mask_image)

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class SegmentationMask:
    def __init__(self, mask: IntImageTensor | None):
        if mask is None:
            self._orig_mask = None
            self._mask = None
            self._objects = torch.as_tensor([], dtype=torch.int64)
        else:
            assert len(mask.shape) == 3
            self._orig_mask = mask
            self._mask = mask.permute(1, 2, 0) if mask.shape[0] == 3 else mask
            self._objects = torch.unique(self._mask.reshape(-1, 3), dim=0)[1:]

    @property
    def mask(self) -> IntImageTensor | None:
        return self._orig_mask

    @property
    def colors(self) -> ColorListTensor:
        return self._objects

    def __len__(self) -> int:
        return len(self.colors)

    def __getitem__(self, index: int) -> IntMaskTensor:
        if self._mask is None:
            raise KeyError(index)
        
        object_color = self._objects[index]

        object_mask = (self._mask == object_color)
        object_mask = object_mask.all(dim=-1).reshape(self._mask.shape[:2])

        empty = torch.zeros(self._mask.shape[:2], dtype=torch.int64)
        empty[object_mask] = 1

        return empty

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __array__(self, copy: bool = False) -> np.ndarray | None:
        return self._orig_mask.detach().cpu().numpy() if self._orig_mask is not None else None

def get_transform_train():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToFloat(255.0, p=1.0),
        ToTensorV2(transpose_mask=True, p=1.0)
    ], bbox_params={'format':'pascal_voc', 'label_fields': ['labels']})

def get_transform_valid():
    return A.Compose([
        A.ToFloat(255.0, p=1.0),
        ToTensorV2(transpose_mask=True, p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})
