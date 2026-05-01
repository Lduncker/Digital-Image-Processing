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
from utils.general import non_max_suppression
from models.yolo import Model

IM_WIDTH = 640
IM_HEIGHT = 640
batchSize = 32
gEpochs = 20

path = kagglehub.dataset_download("fareselmenshawii/license-plate-dataset")

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

#shows the images versus their prediction
def show_results(imgs, preds, examples, conf_thres=0.1, iou_thres=0.45):
    imgs = imgs.cpu()
    
    #YOLOv5 returns (pred, train_out)
    preds = preds[0] if isinstance(preds, tuple) else preds
    preds = preds.cpu()
    
    print("Raw preds shape:", preds.shape)
    print("Max conf:", preds[..., 4].max().item())
    print("Mean conf:", preds[..., 4].mean().item())
    
    preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

    imgs_np = imgs.numpy()

    for i in range(min(examples, len(imgs_np))):
        print("Displaying image: ", i)
        img = imgs_np[i].transpose(1, 2, 0).copy()  # CHW → HWC

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis("off")

        detections = preds[i]

        if detections is not None and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                ax.add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        color="red",
                        linewidth=2
                    )
                )

                ax.text(
                    x1,
                    y1 - 5,
                    f"{conf:.2f}",
                    color="red",
                    fontsize=8,
                    backgroundcolor="white"
                )
        else:
            print("detections is none")

        fig.savefig(f"output_{i}.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

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

if __name__ == "__main__":
    valDataset = ImageDataset(
        imageFolder = os.path.join(path, "images", "val"),
        labelFolder = os.path.join(path, "labels", "val"),
        im_width = IM_WIDTH,
        im_height = IM_HEIGHT
    )
    
    valLoader = DataLoader(
        valDataset,
        batch_size = batchSize,
        shuffle = True,
        num_workers = 0,
        collate_fn=collate_fn
    )
    
    #setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(
        cfg=str((Path(__file__).parent / "external/yolov5/models/yolov5s.yaml").resolve()),
        ch=3,
        nc=1
    ).to(device)
    
    #load the retrained weights
    weights = torch.load(str((Path(__file__).parent / "yolov5_retrained.pt").resolve()), map_location=device, weights_only=False)
    model.load_state_dict(weights)
    
    model.eval()
    
    #num examples
    examples = 5
    
    with torch.inference_mode():
        for imgs, targets in valLoader:
            imgs = imgs[:examples].to(device)
            
            preds = model(imgs)
            
            show_results(imgs, preds, examples)
            
            break
    
    