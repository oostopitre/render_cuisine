from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fastai.vision import (
    ImageDataBunch, cnn_learner, error_rate, get_transforms, imagenet_stats, models
)
from fastai.vision.data import download_images, verify_images

DOWNLOAD_IMAGES_FLAG = False
SHOW_IMAGES_FLAG = False
TRAIN_MODEL_FLAG = True
INFERENCE_MODEL_FLAG = False

data_root = Path('data')
img_root = data_root / 'images'

cuisines = ['chinese', 'indian', 'italian', 'japanese', 'mexican']

# Download data
if DOWNLOAD_IMAGES_FLAG:
    for cuisine in cuisines:
        url_fpath = data_root / f'urls_{cuisine}.csv'
        dest_folder = img_root / cuisine

        print(f'{url_fpath} >> {dest_folder}')
        dest_folder.mkdir(parents=True, exist_ok=True)

        download_images(url_fpath, dest_folder, max_pics=20, max_workers=0)
        verify_images(dest_folder, delete=True, max_size=500)

# Setup dataloader
np.random.seed(42)
data = ImageDataBunch.from_folder(
    img_root, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=0
).normalize(imagenet_stats)
print(f'Classes: {data.classes}, C: {data.c}')
print(f'No. Train Images: {len(data.train_ds)}')
print(f'No. Valid Images: {len(data.valid_ds)}')

# Show Images
if SHOW_IMAGES_FLAG:
    data.show_batch(rows=4, figsize=(7, 8))
    plt.show()

# Train model
if TRAIN_MODEL_FLAG:
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(1)
    learn.save('stage-1')

    # learn.unfreeze()
    # learn.lr_find()

    # learn.fit_one_cycle(2, max_lr=slice(3e-5, 3e-4))
    # learn.save('stage-2')
    # learn.load('stage-2')

    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # plt.show()

    # exported to: data/images/export.pkl, can't change that AFAIK :(
    learn.export()

# Inference
if INFERENCE_MODEL_FLAG:
    import torch
    from fastai import defaults
    from fastai.vision import open_image, load_learner

    defaults.device = torch.device('cpu')

    img = open_image(img_root / 'indian' / '00000001.jpg')
    learn = load_learner(data_root)

    pred_class, pred_idx, outputs = learn.predict(img)
    print(pred_class)
