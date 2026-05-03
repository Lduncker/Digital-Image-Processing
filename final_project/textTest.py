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

repo_path = Path(__file__).parent / "external/yolov5"
sys.path.insert(0, str(repo_path.resolve()))
from utils.general import non_max_suppression
from models.yolo import Model

IM_WIDTH = 640
IM_HEIGHT = 640
BATCH_SIZE = 16
CONF_THRES = 0.18
IOU_THRES = 0.45

RESIZE_WIDTH = 120
RESIZE_HEIGHT = 120
target_height = 64


def run_inference_on_video(model, video_path, video_idx, device, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {total} frames @ {fps:.1f} fps")

    out_path = str(out_dir / f'dashcam_{video_idx}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    best_conf = -1.0
    best_frame_data = None  # (rgb float32, detections)

    batch_tensors = []
    batch_bgr = []
    total_written = 0

    originalFrames = []
    cropCount = 0

    def flush_batch():
        nonlocal writer, best_conf, best_frame_data, total_written, cropCount
        if not batch_tensors:
            return

        tensor_batch = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            raw = model(tensor_batch)
        preds = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(preds.cpu(), conf_thres=CONF_THRES, iou_thres=IOU_THRES)

        for bgr, det, frame in zip(batch_bgr, detections, originalFrames):
            annotated = bgr.copy()
            max_conf = 0.0
            if det is not None and len(det) > 0:
                for d in det:
                    x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    conf = float(conf)
                    max_conf = max(max_conf, conf)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    #save a cropped version for text detection
                    scale_x = frame.shape[1] / IM_WIDTH
                    scale_y = frame.shape[0] / IM_HEIGHT

                    x1o = int(x1 * scale_x)
                    x2o = int(x2 * scale_x)
                    y1o = int(y1 * scale_y)
                    y2o = int(y2 * scale_y)

                    crop = frame[y1o:y2o, x1o:x2o]
                    
                    if crop.size == 0:
                        continue

                    #crop = cv2.resize(crop, (RESIZE_WIDTH, RESIZE_HEIGHT))

                    crop_filename = crop_dir / f"video{video_idx}_crop{cropCount}.jpg"
                    cv2.imwrite(str(crop_filename), crop)

                    cropCount += 1

            if max_conf > best_conf:
                best_conf = max_conf
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                best_frame_data = (rgb, det)

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            writer.write(annotated)
            total_written += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (IM_WIDTH, IM_HEIGHT))
        batch_bgr.append(resized)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb).permute(2, 0, 1).float() / 255.0
        batch_tensors.append(tensor)
        originalFrames.append(frame)

        if len(batch_tensors) == BATCH_SIZE:
            flush_batch()
            originalFrames = []
            batch_tensors = []
            batch_bgr = []

    flush_batch()
    cap.release()
    if writer:
        writer.release()

    print(f"  Written {total_written} annotated frames -> {out_path}")

    # Save best-detection frame as a report figure
    if best_frame_data is not None:
        img, det = best_frame_data
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        if det is not None and len(det) > 0:
            for d in det:
                x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, color='red', linewidth=2))
                ax.text(x1, y1 - 5, f"{conf:.2f}", color='red', fontsize=8,
                        backgroundcolor='white')
        out_fig = out_dir / f"dashcam_{video_idx}.png"
        fig.savefig(out_fig, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  Report figure: {out_fig}  (best conf={best_conf:.2f})")
    else:
        print(f"  No detections in {video_path.name}")

#checks the size of a plate image and cuts it if its too large
#this is done to remove obvious errors, since due to the perspective of the dashcam plates are almost never above a certain sizex
thresholdX = 200
thresholdY = 100
def blindCut(img):
    #get size
    h, w = img.shape[:2]
    
    if h > thresholdY and w > thresholdX:
        return 0
    else:
        return 1

def processPlate(img):
    #resize
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, target_height))

    #grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #utilize some edge detection to try and recover the text as much as possible
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
        
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9, 5
    )
    
    return thresh
    

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.makedirs(script_dir / 'outputVideo', exist_ok=True)
    
    crop_dir = script_dir / "outputFrames"
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

    #for testing you can skip the generation of videos
    justRead = 1
    if not justRead:
    
        video_dir = script_dir / 'dashcam_videos'
        videos = sorted(v for v in os.listdir(video_dir) if not v.startswith('.'))
        print(f"Found {len(videos)} dashcam videos\n")

        out_dir = script_dir / 'outputVideo'
        for idx, vfile in enumerate(videos):
            print(f"[{idx + 1}/{len(videos)}] {vfile}")
            run_inference_on_video(model, video_dir / Path(vfile), idx, device, out_dir)

        print("\nAll done.")
        print(f"Annotated videos : {out_dir}/dashcam_0.mp4 ... dashcam_{len(videos) - 1}.mp4")
        print(f"Report figures   : {out_dir}/dashcam_0.png ... dashcam_{len(videos) - 1}.png")
    
    #whether to do my cropping tests
    cropTest = 1
    if cropTest:
        # Load crop filenames
        croppedFrames = [c for c in os.listdir(crop_dir) if not c.startswith('.')]
        croppedFrames = sorted(croppedFrames)

        # Randomly pick 10 (or fewer if not enough images)
        numExamples = min(20, len(croppedFrames))
        sampled = random.sample(croppedFrames, numExamples)

        for fname in sampled:
            path = os.path.join(crop_dir, fname)

            #read
            img = cv2.imread(path)
            if img is None:
                continue
            
            #try a blind cut just for testing
            isValid = blindCut(img)
            if isValid:
                print("Pretty sure this is actually a plate")
            else:
                print("I don't think this is a plate")
                continue
            
            thresh = processPlate(img)

            cv2.imshow("Original", img)
            cv2.imshow("Processed", thresh)
            
            
            key = cv2.waitKey(0)
            if key == 27:  # press ESC to quit early
                break
            elif key == 13:
                continue

        cv2.destroyAllWindows()
