# -*- coding: utf-8 -*-
"""2023dlcvhw1-2 (3).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1npo2LWc26_7uD9CO-SRGm6JV3PkQo-bq

# Set up packages for HW1
"""

!pip install --upgrade gdown tqdm transformers==4.21.3 timm==0.6.7 scikit-learn==1.1.2 pandas einops imageio==2.21.2 matplotlib==3.5.3 numpy==1.23.1 Pillow==9.2.0 scipy==1.9.1 torch==1.12.1 torchvision==0.13.1 byol-pytorch

"""# About the Dataset"""

# Download dataset
!gdown 1owVOM3V94bC8v2N8s7n56ciVzVbSzYtd -O hw1_data.zip

# Unzip the downloaded zip file
# This may take some time.
!unzip -q ./hw1_data.zip

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image #PIL包含在pillow這個函式庫
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm import tqdm
import random

# set a random seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for this recognition of this task.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(), #影像 RandomHorizontalFlip 和 RandomVerticalFlip
    # transforms.RandomResizedCrop(32,scale=(0.08, 1.0)),#影像 RandomCrop:torchvision.transforms.RandomCrop(size(寬度和高度，可以為一個值), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
    #影像 GaussianBlur:torchvision.transforms.GaussianBlur(kernel_size(模糊半徑，必須是奇數), sigma（高斯kernel生成的標準差，須為固定值在設定的float值或是在（min,max）內）=(0.1,.2))(img)，半徑和標準差越大照片越模糊
    transforms.CenterCrop(128),
    # transforms.RandomRotation(30),#影像 RandomRotation:torchvision.transforms.RandomRotation(degrees（如 30，则表示在（-30，+30）之间随机旋转
    # 若为sequence，如(30，60)，则表示在30-60度之间随机旋转）, resample=False(resample- 重采样方法选择，可选 PIL.Image.NEAREST, PIL.Image.BILINEAR), expand=False, center=None:以圖的正中心旋轉)，
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    # You may add some transforms here!!!!!!!!!!!! https://blog.csdn.net/qq_42951560/article/details/109852790
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

mini_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

"""# Datasets"""

class hw1_2Dataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(hw1_2Dataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx] # fname = ./hw1_data/p1_data/train_50/31_327.png
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label

"""# Create Model and Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model_A = models.resnet50(pretrained=False)
model_A.load_state_dict(torch.load('/kaggle/input/2023dlcvhw1-2-exp0-ssl-ckpt/exp0_SSL_best.ckpt'))
in_features = model_A.fc.in_features
# model_A.fc = torch.nn.Linear(in_features, num_classes)

model_SSL = model_A
model_SSL.to(device)

# The number of batch size.
batch_size = 128

bk_ckpt_name = "exp1_SSL_best.ckpt"

"""# Dataloader(Mini-ImageNet)"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
mini_train_set = hw1_2Dataset("./hw1_data/p2_data/mini/train", tfm=mini_tfm)
mini_train_loader = DataLoader(mini_train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

"""# Pretraining on Mini-ImageNet(without loading default pretrained weights)"""

import torch
from byol_pytorch import BYOL
import math

learner = BYOL(
    model_SSL,
    image_size = 128,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
ues_lrscheduling = True
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10)
best_loss, early_stop, early_stop_count = math.inf, 10, 0
n_epochs = 200
for epoch in range(n_epochs):
    model_SSL.train()

    # These are used to record information in training.
    train_loss = []

    for batch in tqdm(mini_train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        loss = learner(imgs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

        train_loss.append(loss.item())
    mean_train_loss = sum(train_loss) / len(train_loss)
    if ues_lrscheduling:
        scheduler.step(mean_train_loss)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {mean_train_loss:.5f}")
    if mean_train_loss < best_loss:
        best_loss = mean_train_loss
        # save your improved network
        torch.save(model_SSL.state_dict(), bk_ckpt_name)
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= early_stop:
        print(f'\nModel is not improving at {epoch+1} epoch, so we halt the training session.')
        break

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 65

# Initialize a model, and put it on the device specified.
model_A = models.resnet50(pretrained=False)
model_A.load_state_dict(torch.load(bk_ckpt_name))
#for param in model_A.parameters():
#    param.requires_grad = False
in_features = model_A.fc.in_features
model_A.fc = torch.nn.Linear(in_features, num_classes)


model = model_A
model.to(device)
# The number of batch size.
batch_size = 256

# The number of training epochs.
n_epochs = 150

# weight decay
wd_num = 0.00001

# MixUp alpha(α \alphaα在0.2 ~ 2之間效果都差不多，表示mixup對α \alphaα參數並不是很敏感。但如果α \alphaα過小，等於沒有進行mixup的原始數據，如果α \alphaα過大，等於所有輸入都是各取一半混合)
alpha = 0

# If no improvement in 'patience' epochs, early stop.
patience = 15

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=wd_num)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)

ckpt_name = "exp1_SSL_trainall_best.ckpt"

"""# Dataloader(Office-Home)"""

office_train_set = hw1_2Dataset("./hw1_data/p2_data/office/train", tfm=train_tfm)
office_train_loader = DataLoader(office_train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
office_valid_set = hw1_2Dataset("./hw1_data/p2_data/office/val", tfm=test_tfm)
office_valid_loader = DataLoader(office_valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""# Downstream task on Office-Home dataset"""

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(office_train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, alpha)
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        # loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(office_valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # update logs
    if valid_acc > best_acc:
        with open("./sample_best_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open("./sample_best_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        best_epoch = epoch+1
        print(f"Best model found at epoch {best_epoch}, saving model")
        torch.save(model.state_dict(), ckpt_name) # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping at no.{epoch+1} epoch, best epoch at {best_epoch}.")
            break