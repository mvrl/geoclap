#Taken from https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def build_transform(is_train, input_size):
    """
    Builds train/eval data transforms for the dataset class.
    :param is_train: Whether to yield train or eval data transform/augmentation.
    :param input_size: Image input size (assumed square image).
    :param mean: Per-channel pixel mean value, shape (c,) for c channels
    :param std: Per-channel pixel std. value, shape (c,)
    :return: Torch data transform for the input image before passing to model
    """
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD

    # train transform
    interpol_mode = transforms.InterpolationMode.BICUBIC

    t = []
    if is_train:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
        )
        t.append(transforms.RandomHorizontalFlip())
        return transforms.Compose(t)

    # eval transform
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(
        transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    # t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    import os
    from torchvision.io import read_image
    import numpy as np
    import torch
    sat_image_path = '/storage1/fs1/jacobsn/Active/user_k.subash/data/aporee/images/sentinel/'
    sat_transform = build_transform(is_train=True, input_size=224)
    demo_image  =  os.listdir(sat_image_path)[0]
    sat_img = read_image(os.path.join(sat_image_path,demo_image))
    sat_img = np.array(torch.permute(sat_img,[1,2,0]))
    print(sat_transform(sat_img).shape)
   

