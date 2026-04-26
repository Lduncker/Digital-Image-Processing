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
import random
import time
from pathlib import Path
import cv2
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

#shows the images versus their prediction
def show_results(title, imgs, preds, examples, conf_thres=0.18, iou_thres=0.45):
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

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Prediction {i}")
        plt.axis("off")

        detections = preds[i]

        if detections is not None and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                plt.gca().add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        color="red",
                        linewidth=2
                    )
                )

                plt.text(
                    x1,
                    y1 - 5,
                    f"{conf:.2f}",
                    color="red",
                    fontsize=8,
                    backgroundcolor="white"
                )
        else:
            print("detections is none")

        #path here
        plt.savefig(f"outputFrames/{title}_output_{i}.png")
        plt.close()

def recreateVideo(numVids, titles):
    fps = 30
    
    imageFolder = Path('outputFrames/')
    outFolder = Path('outputVideo/')
    
    for i in range(numVids):
        videoName = titles[i] + ".mp4"
        
        #get all the frames
        images = sorted([img for img in os.listdir(imageFolder) if img.find(titles[i]) != -1])
        frame = cv2.imread(os.path.join(imageFolder, images[0]))
        height, width, layers = frame.shape
        
        #setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(videoName, fourcc, fps, (width, height))
        
        for image in images:
            video.write(cv2.imread(os.path.join(imageFolder, image)))
        
        video.release()
        cv2.destroyAllWindows()
        
    

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
    #setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(
        cfg="external/yolov5/models/yolov5s.yaml",
        ch=3,
        nc=1
    ).to(device)
    
    #load the retrained weights
    weights = torch.load("yolov5_retrained.pt", map_location=device, weights_only=False)
    model.load_state_dict(weights)
    
    model.eval()
    
    #load frames
    sampleRate = 1
    
    videoPath = Path('dashcam_videos/')
    videoSet = []
    videos = os.listdir(videoPath)
    
    #for each video
    for v in videos:
        vPath = os.path.join(videoPath, v)
        cap = cv2.VideoCapture(vPath)
        
        #go through each frame and add them to a list
        frames = []
        frameCounter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frameCounter % sampleRate == 0:
                frame = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))
                frames.append(frame)
            frameCounter += 1
        
        cap.release()
        
        #stack all the frames into one numpy array then add them to the set
        if frames:
            arr = np.stack(frames)
            videoSet.append(arr)
    
    #videoTensor = [torch.tensor(v) for v in videoSet]
    
    titles = []
    
    with torch.inference_mode():
        for video in videoSet:
            #generate a random number for labeling video frames
            randTitleNum = random.randint(0, 10000)
            titles.append(str(randTitleNum))
            batch = []
            batchSize = 16
            frameCounter = 0
            
            for frame in video:
                frameCounter += 1
                
                if frameCounter % batchSize == 0:
                    batch = torch.stack(batch).to(device)
                    preds = model(batch)
                    
                    show_results(str(randTitleNum) + str(frameCounter), batch, preds, batchSize)   
                    batch = []
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0 #HWC -> CHW
                
                batch.append(frame) 
    
    
    recreateVideo(5, titles)
    