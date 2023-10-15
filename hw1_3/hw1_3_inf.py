# -*- coding: utf-8 -*-
"""2023dlcv-hw1-3-inf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d_r_y1F_JTrIxsjUF2CKl8aeDVX7C_Ti

]# Set up packages for HW1
"""

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import imageio
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image #PIL包含在pillow這個函式庫
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm import tqdm
import random
import argparse
parser = argparse.ArgumentParser(description='Description of my script.')

parser.add_argument('--arg1', type=str, help='Help message for arg1')
parser.add_argument('--arg2', type=str, help='Help message for arg2')

# 解析命令行参数
args = parser.parse_args()

# storing the arguments
test_img_dir = args.arg1
out_masks_dir = args.arg2
check_path = './hw1_3.ckpt'

""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
mv3_backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, dilated=True)
mv3_backbone = mv3_backbone.features
stage_indices = [0] + [i for i, b in enumerate(mv3_backbone) if getattr(b, "_is_cn", False)] + [len(mv3_backbone) - 1]
out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
out_inplanes = mv3_backbone[out_pos].out_channels

aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
aux_inplanes = mv3_backbone[aux_pos].out_channels

def createDeepLabv3_mv3(outputchannels):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
                                                    progress=True)

    model.aux_classifier = FCNHead(aux_inplanes, outputchannels)
    model.classifier = DeepLabHead(out_inplanes, outputchannels)

    return model


def createDeepLabv3_r50(outputchannels):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
                                                    progress=True)

    model.classifier = DeepLabHead(2048, outputchannels)

    return model

# set a random seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def read_masks(seg):
    masks = np.empty((512, 512)) # Return an array of zeros with the same shape and type as a given array
    mask = (seg >= 128).astype(int)  # 將mask中像素值大於等於128的元素轉為整數1，否則數值是原來的整數0
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks[mask == 3] = 0  # (Cyan: 011) Urban land
    masks[mask == 6] = 1 # (Yellow: 110) Agriculture land
    masks[mask == 5] = 2  # (Purple: 101) Rangeland
    masks[mask == 2] = 3  # (Green: 010) Forest land
    masks[mask == 1] = 4  # (Blue: 001) Water
    masks[mask == 7] = 5  # (White: 111) Barren land
    masks[mask == 0] = 6  # (Black: 000) Unknown
    return masks

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for this recognition of this task.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.RandomHorizontalFlip(), #影像 RandomHorizontalFlip 和 RandomVerticalFlip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.RandomHorizontalFlip(),
])

train_target_tfm = transforms.Compose([
    # transforms.ToPILImage(),
    #影像 RandomHorizontalFlip 和 RandomVerticalFlip
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_target_tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

"""# Datasets"""

import imageio

class hw1_3_Dataset(Dataset):

    def __init__(self,path,tfm=test_tfm, target_tfm=test_target_tfm):
        super(hw1_3_Dataset).__init__()
        self.path = path
        self.files = sorted([x.split(".")[0] for x in os.listdir(path) if x.endswith(".jpg")])
        self.imgs = [os.path.join(path,(x)+".jpg") for x in self.files]
        self.masks = [os.path.join(path,(x)+"_mask.png") for x in self.files]

        self.transform = tfm
        self.target_tfm = target_tfm
    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        img_name = self.imgs[idx] # fname = 31_327
        mask_name = self.masks[idx]
        im = Image.open(img_name)
        im = self.transform(im)

        try:
            mask = imageio.imread(mask_name)
            mask = read_masks(mask)
            mask = torch.from_numpy(mask).to(dtype=torch.long)

        except:
            mask = im

        return im,mask

"""# Dataloader(Mini-ImageNet)"""

batch_size = 16
test_set = hw1_3_Dataset(path=test_img_dir, tfm=test_tfm, target_tfm=test_target_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
prediction_names = [f'{i[0]+i[1]+i[2]+i[3]}.png' for i in test_set.files]

"""# Create Model and Configurations"""

num_classes = 7
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

model = createDeepLabv3_mv3(outputchannels=num_classes)
model.to(device)
model.eval()
model.load_state_dict(torch.load(check_path))

def write_masks_batch(masks):
    # Return an array of zeros with the same shape as masks, but with an extra channel dimension
    seg = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    # print('Shape of seg: ', seg.shape)
    # Loop over the batch dimension and call the write_masks function for each mask
    for b in range(masks.shape[0]):
        seg[b] = write_masks(masks[b])
    return seg

def write_masks(masks):
    # Return an array of zeros with the same shape as masks, but with three channels
    seg = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
    # print('Shape of seg: ', seg.shape)
    # Loop over the mask values and assign the corresponding pixel values to seg
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if masks[i, j] == 0: # Urban land (Cyan: 011)
                seg[i, j, 0] = 0
                seg[i, j, 1] = 255
                seg[i, j, 2] = 255
            elif masks[i, j] == 1: # Agriculture land (Yellow: 110)
                seg[i, j, 0] = 255
                seg[i, j, 1] = 255
                seg[i, j, 2] = 0
            elif masks[i, j] == 2: # Rangeland (Purple: 101)
                seg[i, j, 0] = 255
                seg[i, j, 1] = 0
                seg[i, j, 2] = 255
            elif masks[i, j] == 3: # Forest land (Green: 010)
                seg[i, j, 0] = 0
                seg[i, j, 1] = 128
                seg[i, j, 2] = 0
            elif masks[i, j] == 4: # Water (Blue: 001)
                seg[i, j, 0] = 0
                seg[i, j, 1] = 0
                seg[i, j, 2] = 255
            elif masks[i, j] == 5: # Barren land (White: 111)
                seg[i, j, 0] = 255
                seg[i, j, 1] = 255
                seg[i, j, 2] = 255
            elif masks[i, j] == 6: # Unknown (Black: 000)
                seg[i, j, 0] = 0
                seg[i, j, 1] = 0
                seg[i, j, 2] = 0
    return seg

from PIL import Image
from torchvision import transforms

# Import the necessary libraries
import os
import matplotlib.pyplot as plt

count = 0
with torch.no_grad():
    # Loop over the test loader and get the output masks
    for data,_ in tqdm(test_loader):
        output = model(data.to(device))['out']
        output_predictions = output.argmax(1).detach().cpu().numpy()
        #print(output_predictions.shape)
        output_masks = write_masks_batch(output_predictions)
        #print(output_masks.shape)
        # Loop over the output masks and save them as png files
        for i in range(output_masks.shape[0]):
            # Get the corresponding prediction name and file name
            prediction_name = prediction_names[count]
            file_name = test_set.files[count]
            count+=1
            # Save the output mask as a png file in the out_masks_dir
            plt.imsave(os.path.join(out_masks_dir, prediction_name), output_masks[i])
            # print(f'Saved {prediction_name} for {file_name}')