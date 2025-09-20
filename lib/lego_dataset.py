# source: https://www.kaggle.com/datasets/martinellis/synthetic-lego-images/

import cv2
import json
import torch
import logging
import numpy as np
import pandas as pd
import albumentations as A

from pathlib import Path
from joblib import Memory
from numpy.typing import NDArray
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2
from typing import Dict, Literal, Tuple, Annotated, TypedDict, Callable, cast, List, Any

_logger = logging.getLogger(__name__)
_cache = Memory('.cache/', verbose=0)

FloatImageTensor = Annotated[ torch.Tensor, torch.float32, Literal[3, 'H', 'W'] ]
IntImageTensor = Annotated[ torch.Tensor, torch.int64, Literal[3, 'H', 'W'] ]
IntMaskTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 'H', 'W'] ]
BoxesTensor = Annotated[ torch.Tensor, torch.int64, Literal['N', 4] ]
AreaTensor = Annotated[ torch.Tensor, torch.float32, Literal['N'] ]
IsCrowdTensor = Annotated[ torch.Tensor, torch.int64, Literal['N'] ]
LabelsArray = Annotated[ NDArray[np.int_], Literal['N'] ]
LabelsTensor = Annotated[ torch.Tensor, torch.float32, Literal['N'] ]
ColorArray = Annotated[ NDArray[np.uint8], Literal[3] ]

PreprocessCallable = Callable[[torch.Tensor], torch.Tensor]

class ImageTargetDict(TypedDict):
    """ Target description for images """
    image_id: int
    image_fn: str
    boxes: BoxesTensor
    labels: LabelsTensor
    area: AreaTensor
    iscrowd: IsCrowdTensor

class MaskTargetDict(TypedDict):
    """ Target description for segmentation masks """
    image_id: int
    image_fn: str
    mask_id: int
    mask_fn: str
    labels: List[int]

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

            if (max_size := kwargs.get('max_size')):
                if max_size > 0:
                    ann_ids_df = ann_ids_df.sample(n=max_size, replace=False)
                elif max_size < 0:
                    ann_ids_df = ann_ids_df.head(-max_size)

            match (val_size := kwargs.get('val_size')):
                case int() if val_size > 0:
                    valid_ids_df = ann_ids_df.sample(n=int(val_size), replace=False)
                case int() if val_size < 0:
                    valid_ids_df = ann_ids_df.head(-val_size)
                case int() if val_size == 0:
                    raise ValueError(f'Invalid val_size={val_size}')
                case float():
                    valid_ids_df = ann_ids_df.sample(frac=float(val_size), replace=False)
                case _:
                    valid_ids_df = ann_ids_df.sample(frac=0.2, replace=False)

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
                 preprocess_fn: PreprocessCallable | None = None,
                 *,
                 max_size: int = 0,
                 val_size: int | float = 0):

        super().__init__()

        assert self._splits is not None
        assert self._annotations is not None

        self.data_root = Path(data_root)
        self.annotations = self._annotations
        self.preprocess_fn = preprocess_fn

        self.split = split
        self.image_ids = self._splits[split] if split else self.annotations['img_id'].unique()

        classes_df = pd.read_csv(self.data_root / self.CLASSES_FILE)
        self.classes = pd.Series(classes_df['label_name'].astype(str), index=classes_df['label_id'].astype(int)).to_dict()
        self.class_names = pd.Series(classes_df['name'].astype(str), index=classes_df['label_id'].astype(int)).to_dict()

        self.transforms: A.Compose | None = None
        match split:
            case 'train':
                self.transforms = self.get_transform_train(self.preprocess_fn)
            case 'valid':
                self.transforms = self.get_transform_valid(self.preprocess_fn)

    def get_transform_train(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:
        raise NotImplementedError('Not implemented')

    def get_transform_valid(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:
        raise NotImplementedError('Not implemented')

ImageDatasetReturnT = FloatImageTensor | Tuple[FloatImageTensor, ImageTargetDict]

class SynthentcLegoImagesDataset(SynthentcLegoImagesDatasetBase):
    """ Dataset object for Synthetic Lego images """

    def __init__(self,
                 data_root: Path | str,
                 split: Literal['train', 'valid'] | None = None,
                 preprocess_fn: PreprocessCallable | None = None,
                 with_target: bool = True,
                 *,
                 max_size: int = 0,
                 val_size: int | float = 0):

        super().__init__(data_root, split, preprocess_fn, max_size=max_size, val_size=val_size)
        self.with_target = with_target

    def _get(self, index: int | str) -> ImageDatasetReturnT:
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

        labels = torch.as_tensor(records['labels'].values, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        if not self.transforms:
            # simply cast from uint to float
            image = torch.as_tensor(image).to(torch.float32) / 255.0
        else:
            sample = {
                'image': image,
                'bboxes': boxes,
                # 'labels': labels,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)

        if self.preprocess_fn:
            image = self.preprocess_fn(image)

        if self.with_target:
            return image, ImageTargetDict(
                image_id=int(records.index.min()),
                image_fn=str(image_fn),
                boxes=boxes,
                labels=labels,
                area=area,
                iscrowd=iscrowd,
            )
       
        return image

    def get(self, index: int | str) -> ImageDatasetReturnT | None:
        """ Get an entry """
        try:
            return self._get(index)
        except (KeyError, FileNotFoundError):
            return None

    def __getitem__(self, index: int | str) -> ImageDatasetReturnT:
        """ Indexing """
        return self._get(index)

    def __iter__(self):
        """ Iteration """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """ Length """
        return self.image_ids.shape[0]

    def get_transform_train(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:

        transforms = []

        if preprocess_fn:
            transforms.append(A.Lambda(image=preprocess_fn, p=1))

        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0),
        ])
        return A.Compose(
            transforms, 
            bbox_params={
                'format':'pascal_voc', 
                'label_fields': ['labels'],}
            )

    def get_transform_valid(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:

        transforms = []

        if preprocess_fn:
            transforms.append(A.Lambda(image=preprocess_fn, p=1))

        transforms.extend([
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0),
        ])
        return A.Compose(
            transforms, 
            bbox_params={
                'format':'pascal_voc', 
                'label_fields': ['labels']}
            )

MaskDatasetReturnT =  Tuple[FloatImageTensor, IntMaskTensor] | Tuple[FloatImageTensor, IntMaskTensor, MaskTargetDict]

class SynthentcLegoImagesWithMasksDataset(SynthentcLegoImagesDatasetBase):
    """ Dataset object for Synthetic Lego images with segmentation masks """

    ANNOTATION_DEFS_FILE: str = 'annotation-jsons/annotation_definitions.json'
    SEMANTIC_SEGMENTATION_DIR: str = 'semantic-segmentation'

    def __init__(self,
                 data_root: Path | str,
                 split: Literal['train', 'valid'] | None = None,
                 preprocess_fn: PreprocessCallable | None = None,
                 with_cache: bool = True,
                 *,
                 max_size: int = 0,
                 val_size: int | float = 0):
        
        super().__init__(data_root, split, preprocess_fn, max_size=max_size, val_size=val_size)
        self.with_cache = with_cache

        if split:
            assert self._annotations is not None
            self.annotations = self._annotations[ self._annotations['img_id'].isin(self.image_ids) ]

        # self.annotations = self.annotations.explode('labels', ignore_index=True)
        self.annotations = self.annotations.groupby(['img_id','width','height']) \
            .agg(labels=pd.NamedAgg('labels', lambda x: set(x.tolist()))) \
            .reset_index()

        ann_defs = json.loads((self.data_root / self.ANNOTATION_DEFS_FILE).read_text())
        segm_defs = [x for x in ann_defs['annotation_definitions'] if x['name'] == 'semantic segmentation']
        assert segm_defs, f'Cannot find semantic segmentation definition in {self.data_root / self.ANNOTATION_DEFS_FILE} file'

        def to_clr(pixel_value: Dict) -> ColorArray:
            return (np.array([pixel_value['r'], pixel_value['g'], pixel_value['b']], dtype=np.float32) * 255).round(0).astype(np.uint8)

        label_to_id_mapping = {v: k for k, v in self.classes.items()}
        self.class_colors = {label_to_id_mapping[spec['label_name']]: to_clr(spec['pixel_value']) for spec in segm_defs[0]['spec']}

        if self.with_cache:
            self._get = _cache.cache(self._get) #type:ignore

    def _get(self, index: int) -> Tuple[FloatImageTensor, IntMaskTensor, MaskTargetDict]:
        """ Get implementation """

        records = self.annotations.iloc[index]
        if records.empty:
            raise KeyError(index)

        # print(f'cache miss with index {index}')

        image_id = records['img_id']
        labels = cast(List[int], list(records['labels']))

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

        # parse mask by colors
        masks = np.empty((len(self.classes), mask_image.shape[0], mask_image.shape[1]), dtype=np.uint8)
        for lbl, clr in self.class_colors.items():
            if lbl in labels:
                clr_mask = (mask_image == clr)
                object_mask = clr_mask.all(axis=-1).reshape(mask_image.shape[:2])
                if not object_mask.any():
                    _logger.warning(f'Cannot find segmentation mask for label {lbl} on {mask_fn}')
            else:
                object_mask = np.zeros(mask_image.shape[:2], dtype=masks.dtype)
            
            masks[lbl, :] = object_mask

        if not self.transforms:
            # simply cast to tensors
            image = torch.as_tensor(image).to(torch.float32) / 255.0
            masks = torch.as_tensor(masks).to(torch.float32)
        else:
            sample = {
                'image': image,
                'masks': masks,
            }
            sample = self.transforms(**sample)
            image = sample['image']
            masks = sample['masks'].to(torch.float32)

        return image, masks, MaskTargetDict(
            image_id=image_id,
            image_fn=str(image_fn),
            mask_id=index,
            mask_fn=str(mask_fn),
            labels=labels,
        )

    def get(self, index: int) -> Tuple[FloatImageTensor, IntMaskTensor] | None:
        """ Get an entry """
        try:
            return self._get(index)[:2]
        except (KeyError, FileNotFoundError):
            return None

    def get_with_target(self, index: int) -> Tuple[FloatImageTensor, IntMaskTensor, MaskTargetDict] | None:
        """ Get an entry """
        try:
            return self._get(index)
        except (KeyError, FileNotFoundError):
            return None

    def __getitem__(self, index: int) -> Tuple[FloatImageTensor, IntMaskTensor]:
        """ Indexing """
        return self._get(index)[:2]

    # def __iter__(self):
    #     """ Iteration """
    #     for row in self.annotations.itertuples(index=True):
    #         yield self._get(row.Index) # type:ignore

    # def __len__(self) -> int:
    #     """ Length """
    #     return len(self.annotations)
    
    def __iter__(self):
        """ Iteration """
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """ Length """
        return self.image_ids.shape[0]
    
    def get_transform_train(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:

        transforms: List[Any] = [
            A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_CUBIC, p=1),
            A.PadIfNeeded(min_height=320, min_width=320, p=1),
        ]

        if preprocess_fn:
            transforms.append(A.Lambda(image=preprocess_fn, p=1))

        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0),
        ])

        return A.Compose(transforms)

    def get_transform_valid(self, preprocess_fn: PreprocessCallable | None) -> A.Compose:

        transforms: List[Any] = [
            A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_CUBIC, p=1),
            A.PadIfNeeded(min_height=320, min_width=320, p=1),
        ]

        if preprocess_fn:
            transforms.append(A.Lambda(image=preprocess_fn, p=1))

        transforms.extend([
            A.ToFloat(255.0, p=1.0),
            ToTensorV2(transpose_mask=True, p=1.0),
        ])

        return A.Compose(transforms)
