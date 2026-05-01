import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import cv2
from pathlib import Path
import sys

repo_path = Path(__file__).parent / "external/yolov5"
sys.path.insert(0, str(repo_path.resolve()))
from utils.general import non_max_suppression
from models.yolo import Model

IM_WIDTH = 640
IM_HEIGHT = 640
BATCH_SIZE = 16
CONF_THRES = 0.18
IOU_THRES = 0.45


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

    def flush_batch():
        nonlocal writer, best_conf, best_frame_data, total_written
        if not batch_tensors:
            return

        tensor_batch = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            raw = model(tensor_batch)
        preds = raw[0] if isinstance(raw, tuple) else raw
        detections = non_max_suppression(preds.cpu(), conf_thres=CONF_THRES, iou_thres=IOU_THRES)

        for bgr, det in zip(batch_bgr, detections):
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

        if len(batch_tensors) == BATCH_SIZE:
            flush_batch()
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


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    os.makedirs(script_dir / 'outputVideo', exist_ok=True)

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
