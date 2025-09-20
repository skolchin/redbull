# PyTorch model testing script

import re
import cv2
import torch
import click
import warnings
import numpy as np
from pathlib import Path
import albumentations as A
import segmentation_models_pytorch as smp
from typing import List

warnings.simplefilter('ignore', DeprecationWarning)

from lib.videoreader import VideoReader
from lib.lego_dataset import SynthentcLegoImagesWithMasksDataset

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

def get_vis_augmentation():
    transform = [
        A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_CUBIC, p=1),
        A.PadIfNeeded(min_height=320, min_width=320, p=1),
        A.ToFloat(255.0, p=1.0),
        A.ToTensorV2(transpose_mask=True, p=1.0),
    ]
    return A.Compose(transform)

def process_frame(
        image, 
        model, 
        dataset: SynthentcLegoImagesWithMasksDataset, 
        display_labels: List[int] | None, 
        device: torch.device,
        threshold: float):
    
    if not dataset.transforms:
        aug_image = torch.as_tensor(image)
    else:
        aug_image = dataset.transforms(image=image)['image']

    x_tensor = aug_image.to(device).unsqueeze(0)
    pred_mask = model.predict(x_tensor)
    pred_mask = pred_mask.cpu().numpy().squeeze().transpose(1, 2, 0)

    output_aug_image = get_vis_augmentation()(image=image)['image'] \
        .cpu().numpy().transpose(1, 2, 0)
    output_mask_image = output_aug_image.copy()

    for label in (display_labels or list(dataset.classes.keys())):
        if label > 0:
            color = dataset.class_colors[label]
            # color = (0,0,1)
            m = pred_mask[:,:, label].squeeze()
            m_idx = m >= threshold
            output_mask_image[m_idx] = output_mask_image[m_idx]*0.6 + np.array(color)*0.4
        if label > 10:
            break

    return output_aug_image, output_mask_image

@click.command
@click.option('-i', '--image', 'image_path', 
              type=click.Path(exists=True, dir_okay=False), 
              default='./input/bricks.jpeg', show_default=True,
              help='Image file path')
@click.option('-v', '--video', 'video_path', 
              type=click.Path(exists=True, dir_okay=False), 
              default=None,
              help='Video file path')
@click.option('-d', '--data_dir', 
              default='./datasets/synthetic-lego-images/versions/4/', show_default=True,
              help='Dataset root')
@click.option('-o', '--output', 
              type=click.Path(exists=False, file_okay=True, writable=True),
              help='Save result to file')
@click.option('-t', '--threshold', type=click.FloatRange(0.1, 1.0), 
              default=0.8, show_default=True,
              help='Confidence threshold')
@click.option('-f', '--filter', 'class_filter',
              help='Class filter (list of regex expressions separated by comma)')
@click.option('--use_cpu', is_flag=True, help='Run inference on CPU')
@click.option('-s', '--save_dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              default='./weights/', show_default=True,
              help='Weights directory')
def main(
    image_path: str, 
    video_path: str, 
    data_dir: str, 
    output: str, 
    threshold: float, 
    class_filter: str, 
    use_cpu: bool,
    save_dir: str):
    """ Model test """

    device = torch.device(DEVICE_CUDA if not use_cpu else DEVICE_CPU)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    dataset = SynthentcLegoImagesWithMasksDataset(
        data_dir,
        'valid',
        preprocess_fn=preprocessing_fn,
    )
    
    best_name = str(Path(save_dir).joinpath('deeplabv3plus_best.pth'))
    best_model = torch.load(best_name, weights_only=False)

    if not class_filter:
        display_labels = None
    else:
        patterns = [s.strip() for s in class_filter.split(',')]
        display_labels = []
        for cls in dataset.classes:
            for p in patterns:
                if re.search(p, cls):
                    display_labels.append(cls)
                    break

    if not video_path:
        if (image := cv2.imread(image_path)) is None:
            raise Exception(f'Cannot open image file {image_path}')

        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_image, output_image = process_frame(
            orig_image, 
            best_model, 
            dataset, 
            display_labels, 
            device, 
            threshold
        )

        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        stacked_image = np.hstack((aug_image, output_image))

        cv2.imshow(f'{image_path}', stacked_image)
        cv2.waitKey(0)

        if output:
            cv2.imwrite(output, stacked_image)
            print(f'Image saved to {output}')
    # else:
        # stream = cv2.VideoCapture(video_path)
        # output_stream = None

        # while (buf := stream.read()) is not None and buf[0]:
        #     _, frame = buf
        #     aug_frame, output_frame = process_frame(frame, best_model, dataset, display_labels, device, threshold)
        #     aug_frame = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2BGR)

        #     x, y, w, h = 0, 0, legend.shape[1], legend.shape[0]
        #     output_frame[y:y+h, x:x+w] = cv2.addWeighted(output_frame[y:y+h, x:x+w], 0.2, legend, 0.8, 0.0)
        #     stacked_frame = np.hstack((aug_frame, output_frame))

        #     cv2.imshow(f'{video_path}: press q to quit, space to pause', stacked_frame)
        #     if output:
        #         if output_stream is None:
        #             output_fps = round(stream.get(cv2.CAP_PROP_FPS), 0)
        #             output_stream = cv2.VideoWriter(output, FOURCC, output_fps,
        #                                             tuple(reversed(stacked_frame.shape[:2])))
        #         output_stream.write(stacked_frame)
            
        #     match cv2.waitKey(1) & 0xFF:
        #         case 113:   # q
        #             break

        #         case 32:    # space
        #             while cv2.waitKey(10) & 0xFF != 32:
        #                 continue

        # if output_stream is not None:
        #     output_stream.release()
        #     print(f'Video saved to {output}')

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
