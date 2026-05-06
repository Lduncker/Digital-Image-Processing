import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import cv2
from pathlib import Path
import sys
import random
import kagglehub
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, models
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import torchvision.io as io
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tempfile import TemporaryDirectory
from PIL import Image

repo_path = Path(__file__).parent / "external/yolov5"
sys.path.insert(0, str(repo_path.resolve()))
from utils.general import non_max_suppression
from models.yolo import Model

IM_WIDTH = 640
IM_HEIGHT = 640
BATCH_SIZE = 16
CONF_THRES = 0.18
IOU_THRES = 0.45
batchSize = 32


RESIZE_WIDTH = 120
RESIZE_HEIGHT = 120
target_height = 64

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


def run_inference_on_tensor(model, images, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    images = images.to(device)

    annotated_images = []

    with torch.inference_mode():
        raw = model(images)

    preds = raw[0] if isinstance(raw, tuple) else raw
    detections = non_max_suppression(
        preds.cpu(),
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES
    )

    for img_idx, (img_tensor, det) in enumerate(zip(images, detections)):
        rgb = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        annotated = bgr.copy()

        if det is not None and len(det) > 0:
            for det_idx, d in enumerate(det):
                x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                crop = bgr[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

                crop_vis = processPlate(crop)

                if len(crop_vis.shape) == 2:
                    crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)

                ch, cw = crop_vis.shape[:2]
                y_top = y1 - ch - 5

                if not (y_top < 0 or x1 + cw > annotated.shape[1]):
                    annotated[y_top:y_top + ch, x1:x1 + cw] = crop_vis

                    filename = f"img{img_idx}_det{det_idx}.png"
                    save_path = os.path.join(save_dir, filename)
                    cv2.imwrite(save_path, annotated)

        annotated_images.append(annotated)

    return annotated_images

#runs processing to make a plate more human readable
def processPlate(img):

    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img

    scale = target_height / h
    new_w = int(w * scale)
    if new_w <= 0:
        return img

    img = cv2.resize(img, (new_w, target_height))

    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #normalize lighting
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    #contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    #denoise but preserve edges
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    #sharpen
    sharp = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharp)

    #adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 4
    )

    #morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh
    
#collate function for the dataloader
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
    script_dir = Path(__file__).parent
    
    crop_dir = script_dir / "licensePlateEnhancementTests"
    crop_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Model(
        cfg=str(script_dir / "external/yolov5/models/yolov5s.yaml"),
        ch=3,
        nc=1
    ).to(device)

    weights = torch.load(script_dir / "yolov5_retrained.pt", map_location=device, weights_only=False)
    model.load_state_dict(weights)
    model.eval()
    
    #make dataLoader
    valDataset = ImageDataset(
        imageFolder = os.path.join(path, "images", "val"),
        labelFolder = os.path.join(path, "labels", "val"),
        im_width = IM_WIDTH,
        im_height = IM_HEIGHT
    )
    
    numExamples = 50
    subset, _ = random_split(valDataset, [numExamples, len(valDataset) - numExamples])

    images = torch.stack([subset[i][0] for i in range(len(subset))])

    
    run_inference_on_tensor(model, images, device, crop_dir)
