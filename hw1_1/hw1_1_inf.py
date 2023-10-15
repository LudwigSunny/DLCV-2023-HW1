# Import Packages
import sys
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

# storing the arguments
test_img_dir = sys.argv[1]
output_pred_path = sys.argv[2]
check_path = './hw1_1/hw1_1_best.ckpt'

"""# Transforms
Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
"""

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for this recognition of this task.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), #影像 RandomHorizontalFlip 和 RandomVerticalFlip
    # transforms.RandomResizedCrop(32,scale=(0.08, 1.0)),#影像 RandomCrop:torchvision.transforms.RandomCrop(size(寬度和高度，可以為一個值), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
    #影像 GaussianBlur:torchvision.transforms.GaussianBlur(kernel_size(模糊半徑，必須是奇數), sigma（高斯kernel生成的標準差，須為固定值在設定的float值或是在（min,max）內）=(0.1,.2))(img)，半徑和標準差越大照片越模糊
    transforms.RandomRotation(30),#影像 RandomRotation:torchvision.transforms.RandomRotation(degrees（如 30，则表示在（-30，+30）之间随机旋转
    # 若为sequence，如(30，60)，则表示在30-60度之间随机旋转）, resample=False(resample- 重采样方法选择，可选 PIL.Image.NEAREST, PIL.Image.BILINEAR), expand=False, center=None:以圖的正中心旋轉)，
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    # You may add some transforms here!!!!!!!!!!!! https://blog.csdn.net/qq_42951560/article/details/109852790
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.ToTensor(),
])

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class hw1_1Dataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(hw1_1Dataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".png")])
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

"""# Create Model"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 50

# Initialize a model, and put it on the device specified.

model_E = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
in_features = model_E.classifier[1].in_features
model_E.classifier[1] = torch.nn.Linear(in_features, num_classes)


model = model_E
model.load_state_dict(torch.load(check_path))
model.to(device)

# The number of batch size.
batch_size = 32

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

"""### Dataloader for test"""

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = hw1_1Dataset(test_img_dir, tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model.eval()

# These are used to record information in validation.
valid_loss = []
valid_accs = []

# Iterate the validation set by batches.
for batch in tqdm(test_loader):

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
print(f"[ Valid ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

"""### Testing and generate prediction CSV"""

model_best = model_E
model_best.to(device)
model_best.load_state_dict(torch.load(check_path))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_pred_np = test_pred.cpu().data.numpy()

        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

#create test csv
def get_filename(test_path):
    return sorted([x for x in os.listdir(test_path) if x.endswith(".png")])
filename = get_filename(test_img_dir)
df = pd.DataFrame()
df["filename"] = filename
df["label"] = prediction
df.to_csv(output_pred_path, index = False)