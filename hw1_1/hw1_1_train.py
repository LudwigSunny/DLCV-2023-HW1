# -*- coding: utf-8 -*-
"""2023dlcvhw1-1 (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZWvU9sXBUrv2yCjyIzJKCIuU9J_jnBsF

## HW1-1 Image Classification：

# Set up environment
"""

# Import Packages

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
import timm
# packages for inceptionnext model
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

# set a random seed for reproducibility
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

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

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 50

# Initialize a model, and put it on the device specified.
model_A = models.resnet50(pretrained=False)
in_features = model_A.fc.in_features
model_A.fc = torch.nn.Linear(in_features, num_classes)

model_B = inceptionnext_tiny(pretrained=True) # can change different model name

model_C = models.resnet50(pretrained=True)
in_features = model_C.fc.in_features
model_C.fc = torch.nn.Linear(in_features, num_classes)


model_D = timm.create_model('convnext_base_in22ft1k', pretrained=True, num_classes=50)

model_E = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_mobilenetv2_x1_4", pretrained=True)
in_features = model_E.classifier[1].in_features
model_E.classifier[1] = torch.nn.Linear(in_features, num_classes)


model = model_E
model.to(device)

# The number of batch size.
batch_size = 256

# The number of training epochs.
n_epochs = 200

# weight decay
wd_num = 0.00001

# MixUp alpha(α \alphaα在0.2 ~ 2之間效果都差不多，表示mixup對α \alphaα參數並不是很敏感。但如果α \alphaα過小，等於沒有進行mixup的原始數據，如果α \alphaα過大，等於所有輸入都是各取一半混合)
alpha = 0

# If no improvement in 'patience' epochs, early stop.
patience = 20

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=wd_num)

ckpt_name = "cifar100_mobilenetv2_x1_4_exp2_best.ckpt"
ckpt_first_name = "cifar100_mobilenetv2_x1_4_exp2_first.ckpt"
ckpt_middle_name = "cifar100_mobilenetv2_x1_4_exp2_middle.ckpt"
ckpt_last_name = "cifar100_mobilenetv2_x1_4_exp2_last.ckpt"

"""### Dataloader(Use MixUp and CutMix)
Where to use MixUp and CutMix: After the DataLoader
The simplest way to do this is right "after the DataLoader": the Dataloader has already batched the images and labels for us, and this is exactly what these transforms expect as input:
"""

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = hw1_1Dataset("./hw1_data/p1_data/train_50", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = hw1_1Dataset("./hw1_data/p1_data/val_50", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""### Start Training"""

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

    for batch in tqdm(train_loader):

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
    for batch in tqdm(valid_loader):

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

    if (epoch+1)==1:
        print(f"middle model found at epoch {epoch+1}, val acc is {valid_acc}, saving model")
        torch.save(model.state_dict(), ckpt_first_name)

    if (epoch+1)==10:
        print(f"middle model found at epoch {epoch+1}, val acc is {valid_acc}, saving model")
        torch.save(model.state_dict(), ckpt_middle_name)

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
            torch.save(model.state_dict(), ckpt_last_name)
            break
    if (epoch+1) == n_epochs:
        torch.save(model.state_dict(), ckpt_last_name)

"""### Dataloader for test"""

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = hw1_1Dataset("./hw1_data/p1_data/val_50", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""### Testing and generate prediction CSV"""

model_best = model_E
model_best.to(device)
model_best.load_state_dict(torch.load(f'/kaggle/working/{ckpt_name}'))
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
filename = get_filename('./hw1_data/p1_data/val_50')
df = pd.DataFrame()
df["filename"] = filename
df["label"] = prediction
df.to_csv("submission.csv",index = False)

"""# Q2. Visual Representations Implementation
## Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE (t-distributed Stochastic Neighbor Embedding) on the output on 3 different epochs(first, middle, last).
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = model_E
model.load_state_dict(torch.load('/kaggle/input/2023dlcv-hw1-1-a/2023dlcv_hw1_1_A/cifar100_mobilenetv2_x1_4_exp2_first.ckpt'))
model.eval()

# Load the vaildation set defined by TA
valid_set = hw1_1Dataset("./hw1_data/p1_data/val_50", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# Extract the representations for the specific layer of model
index = ... # You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model.features[:18](imgs.to(device))
        logits = logits.view(logits.size()[0], -1)
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)

features = np.array(features)
print(features.shape)
print(len(labels))
colors_per_class = cm.tab20(np.linspace(0, 1, 50))

# Apply t-SNE to the features
features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5, color=colors_per_class[label])

# 移动图例到右上角并分列显示
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # 可根据需要更改ncol的值


# 保存图像
plt.savefig('tsne_plot.png')  # 指定图像文件名

plt.show()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

# ... 之前的代码 ...

# Apply PCA to the features
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Plot the PCA visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_pca[labels == label, 0], features_pca[labels == label, 1], label=label, s=5)

# 移动图例到右上角并分列显示
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)  # 可根据需要更改ncol的值

plt.show()

'''
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = Classifier().to(device)
state_dict = torch.load('/kaggle/input/2023mlhw3-exp1/sample_best.ckpt')
model.load_state_dict(state_dict)
model.eval()

print(model)
'''

'''

# Load the vaildation set defined by TA
valid_set = FoodDataset("/kaggle/input/ml2023spring-hw3/valid", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

# Extract the representations for the specific layer of model
index = 20
# You should find out the index of layer which is defined as "top" or 'mid' layer of your model.
# 12(middel，可以看得到部分圖片),16（Top）,20（Top，可以看到整張圖片），取「MaxPool2d」的輸出，不然參數會太多，RAM不夠用
features = []
labels = []
for batch in tqdm(valid_loader):
    imgs, lbls = batch
    with torch.no_grad():
        logits = model.cnn[:index](imgs.to(device))
        logits = logits.view(logits.size()[0], -1)
    labels.extend(lbls.cpu().numpy())
    logits = np.squeeze(logits.cpu().numpy())
    features.extend(logits)
print(len(features))
# array 和 asarray 都可以將 結構資料 轉化為 ndarray，但是主要區別就是當資料來源是ndarray時，array仍然會copy出一個副本，佔用新的記憶體，但asarray不會。
features = np.asarray(features)
colors_per_class = cm.rainbow(np.linspace(0, 1, 11))

# Apply t-SNE to the features
features_tsne = TSNE(n_components=2, init='pca', random_state=42).fit_transform(features)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], label=label, s=5)
plt.legend()
plt.savefig('layer_16.png')

'''