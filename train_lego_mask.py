# PyTorch model training script

import json
import click
import torch
import logging
# import json_numpy
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from lib.lego_dataset import LegoWithMasksDataset

# json_numpy.patch()

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
PATIENCE = 5
LR_INITIAL = 1e-3
LR_DECAY = 0.9
LR_ON_PLATEAU_DECAY = 0.1
LR_MINIMAL = 1e-5
MAX_PLATEAU_PERIODS = 3
MIN_SCORE_CHANGE = 1e-3
APPLY_LR_DECAY_EPOCH = 30

logging.basicConfig(level=logging.INFO)
logging.getLogger('lib.lego_dataset').setLevel(logging.ERROR)

def numpy_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

@click.command
@click.option('-n', '--epochs', type=click.IntRange(1, 10000),
              default=10, show_default=True,
              help='Number of epoch to train')
@click.option('-i', '--image_count',type=int,
              default=10, show_default=True,
              help='Maximum number of images to be processed (=0 - all)')
@click.option('--use_cpu', is_flag=True,
              help='Run training on CPU (much slower)')
@click.option('-d', '--data_dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='./datasets/synthetic-lego-images/versions/4/', show_default=True,
              help='Dataset root')
@click.option('-v', '--val-size',type=float,
              default=0.2, show_default=True,
              help='Validation-to-train split ratio')
@click.option('-s', '--save_dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='./weights/', show_default=True,
              help='Weights directory')
def main(
    epochs: int,
    image_count: int,
    use_cpu: bool,
    data_dir: str,
    val_size: float,
    save_dir: str):
    """ Training script """

    device = DEVICE_CUDA if not use_cpu else DEVICE_CPU
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = LegoWithMasksDataset(
        data_dir,
        split='train',
        preprocess_fn=preprocessing_fn,
        max_size=image_count,
        val_size=val_size,
        with_cache=False,
    )
    val_dataset = LegoWithMasksDataset(
        data_dir,
        split='valid',
        preprocess_fn=preprocessing_fn,
        with_cache=False,
    )
    print(f'Train / val length: {len(train_dataset)} / {len(val_dataset)}')
    # print(f'Validation images (10 max):\n{"\n".join([f"\t{f}" for f in val_dataset.annotations["img_id"].unique()][:10])}')

    # sample = train_dataset[0]
    # plt.subplot(1, 2, 1)
    # plt.imshow(sample[0].cpu().numpy().transpose(1, 2, 0))
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(sample[1].cpu().numpy().sum(axis=0))
    # plt.axis('off')
    # plt.show()

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=ACTIVATION,
        classes=len(train_dataset.classes),
        aux_params={
            'classes': len(train_dataset.classes),
            'activation': ACTIVATION,
            'dropout': 0.2,
        }
    ).to(device)

    loss = smp_utils.losses.CrossEntropyLoss().to(device)

    # loss = smp.losses.DiceLoss(
    #     mode='multilabel',
    #     classes=list(train_dataset.classes.keys()),
    # ).to(device)
    # setattr(loss, '__name__', 'dice_loss')

    metrics = [
        smp_utils.metrics.IoU(threshold=0.5).to(device),
        smp_utils.metrics.Accuracy(threshold=0.5).to(device),
    ]

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.AdamW([
        {'params' : model.decoder.parameters(), 'lr': LR_INITIAL},
    ])

    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    val_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    Path(save_dir).mkdir(exist_ok=True)
    save_name = str(Path(save_dir).joinpath('lego_masked.pth'))
    save_name_best = str(Path(save_dir).joinpath('lego_masked_best.pth'))

    min_score = 100
    min_score_epoch = epochs
    plateau_periods = 0
    print(f'Training for {epochs} epochs')

    train_logs_list = []
    valid_logs_list = []

    epoch = 0
    lr = LR_INITIAL
    score = 0.0

    for epoch in range(0, epochs):
        lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch #{epoch+1} (learning rate - {lr:.2e})')

        train_logs = train_epoch.run(train_loader)
        valid_logs = val_epoch.run(val_loader)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        score = valid_logs['dice_loss']
        if score < min_score - MIN_SCORE_CHANGE:
            # score decreased
            min_score = score
            min_score_epoch = epoch
            plateau_periods = 0
            torch.save(model, save_name_best)

            if epoch and epoch % APPLY_LR_DECAY_EPOCH == 0:
                lr *= LR_DECAY

        elif min_score_epoch < epoch - PATIENCE:
            # score didn't decrease in PATIENCE interval, decrease LR or stop at once
            lr *= LR_ON_PLATEAU_DECAY
            min_score_epoch = epoch
            plateau_periods += 1
            if plateau_periods > MAX_PLATEAU_PERIODS:
                # Early stopping
                print(f'Early termination!')
                break

        else:
            if epoch and epoch % APPLY_LR_DECAY_EPOCH == 0:
                lr *= LR_DECAY

        optimizer.param_groups[0]['lr'] = max(lr, LR_MINIMAL)
        torch.save(model, save_name)

    if epoch > 0:
        Path(save_dir).joinpath('lego_masked_log.json').write_text(
            json.dumps({
                    'train_files': train_dataset.annotations["img_id"].unique().tolist(),
                    'val_files': val_dataset.annotations["img_id"].unique().tolist(),
                    'stats': {
                        'epoch': epoch+1,
                        'initial_lr': LR_INITIAL,
                        'final_lr': lr,
                        'final_score': score,
                    },
                    'model': save_name_best,
                    'train_logs': train_logs_list,
                    'val_logs': valid_logs_list,
                },
                default=numpy_serializer,
                indent=2,
                ensure_ascii=False,
            )
        )

if __name__ == '__main__':
    main()
