import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import torchvision.io as io
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tempfile import TemporaryDirectory
import os
import time
from pathlib import Path
import sys
repo_path = Path(__file__).parent / "external/yolov5"
sys.path.insert(0, str(repo_path.resolve()))
from utils.loss import ComputeLoss
from models.yolo import Model

IM_WIDTH = 640
IM_HEIGHT = 640
batchSize = 32
gEpochs = 20

# Download latest version
path = kagglehub.dataset_download("fareselmenshawii/license-plate-dataset")

#load model
#Data Prep
transform = transforms.Compose([
    transforms.Resize((IM_HEIGHT, IM_WIDTH)),
    transforms.ToTensor()
])


class ImageDataset(Dataset):
    def __init__(self, imageFolder, labelFolder, im_width, im_height):
        self.image_paths = list(Path(imageFolder).glob("*"))
        
        self.label_paths = [
            Path(labelFolder) / (p.stem + ".txt")
            for p in self.image_paths
        ]
        
        self.im_width = im_width
        self.im_height = im_height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #images
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.im_width, self.im_height))
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.permute(2, 0, 1)  # HWC → CHW
        
        #labels
        boxes = []

        with open(self.label_paths[idx], "r") as f:
            for line in f.readlines():
                values = list(map(float, line.strip().split()))
                boxes.append(values)

        boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, boxes

def collate_fn(batch):
    images = []
    targets = []

    for i, (img, boxes) in enumerate(batch):
        images.append(img)

        if boxes.numel() > 0:
            img_idx = torch.full((boxes.shape[0], 1), i)
            boxes = torch.cat([img_idx, boxes], dim=1)  # [N, 6]
            targets.append(boxes)

    images = torch.stack(images, dim=0)

    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6))

    return images, targets

def train(dataloader, valloader, device, model, compute_loss, optimizer, epochs):
    for epoch in range(epochs):
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            preds = model(imgs)
            loss, loss_items = compute_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss = {loss.item()}")

if __name__ == "__main__":
    trainDataset = ImageDataset(
        imageFolder = path + "\\images\\train",
        labelFolder = path + "\\labels\\train",
        im_width = IM_WIDTH,
        im_height = IM_HEIGHT
    )
    
    trainLoader = DataLoader(
        trainDataset,
        batch_size = batchSize,
        shuffle = True,
        num_workers = 4,
        collate_fn=collate_fn
    )
    
    valDataset = ImageDataset(
        imageFolder = path + "\\images\\val",
        labelFolder = path + "\\labels\\val",
        im_width = IM_WIDTH,
        im_height = IM_HEIGHT
    )
    
    valLoader = DataLoader(
        valDataset,
        batch_size = batchSize,
        shuffle = True,
        num_workers = 4,
        collate_fn=collate_fn
    )
    
    #setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #The below comments are for training from scratch
    model = Model(
        cfg="external/yolov5/models/yolov5s.yaml",
        ch=3,
        nc=1
    ).to(device)
    
    #this starts training from pretrained weights
    ckpt = torch.load("external/yolov5/yolov5s.pt", map_location=device, weights_only=False)
    model_dict = model.state_dict()
    pretrained_dict = ckpt["model"].state_dict()

    # Filter out mismatched keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # Update current model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    #Additionally hyperparameters needed for training yolov5
    model.hyp = {
        'lr0': 0.01, # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.01, # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937, # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # optimizer weight decay 5e-4
        'warmup_epochs': 3.0, # warmup epochs (fractions ok)
        'warmup_momentum': 0.8, # warmup initial momentum
        'warmup_bias_lr': 0.1, # warmup initial bias lr
        'box': 0.05, # box loss gain
        'cls': 0.5, # cls loss gain
        'cls_pw': 1.0, # cls BCELoss positive_weight
        'obj': 1.0, # obj loss gain (scale with pixels)
        'obj_pw': 1.0, # obj BCELoss positive_weight
        'iou_t': 0.20, # IoU training threshold
        'anchor_t': 4.0, # anchor-multiple threshold
        # anchors: 3  # anchors per output layer (0 to ignore)
        'fl_gamma': 0.0, # focal loss gamma (efficientDet default gamma=1.5)
        'hsv_h': 0.015, # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7, # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4, # image HSV-Value augmentation (fraction)
        'degrees': 0.0, # image rotation (+/- deg)d
        'translate': 0.1, # image translation (+/- fraction)
        'scale': 0.5, # image scale (+/- gain)
        'shear': 0.0, # image shear (+/- deg)
        'perspective': 0.0, # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0, # image flip up-down (probability)
        'fliplr': 0.5, # image flip left-right (probability)
        'mosaic': 1.0, # image mosaic (probability)
        'mixup': 0.0, # image mixup (probability)
        'copy_paste': 0.0 # segment copy-paste (probability)
    }
    
    compute_loss = ComputeLoss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train(trainLoader, valLoader, device, model, compute_loss, optimizer, epochs = gEpochs)
    
    save_path = "yolov5_retrained.pt"
    torch.save(model.state_dict(), save_path)
    
    pass