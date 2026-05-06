# License Plate Detection with YOLOv5

Final project for Digital Image Processing (Worcester Polytechnic
Institute). Authors: Lucian Duncker, Joseph Pasino.

The goal is to detect license plates in road-scene images and draw a
bounding box around each one. We fine-tune Ultralytics YOLOv5s (COCO
pretrained) into a single-class detector on the public Kaggle
`fareselmenshawii/license-plate-dataset` and evaluate it qualitatively
on held-out images.

Full write-up: [`final_report.tex`](final_report.tex) (compiles to
`final_report.pdf`, CVPR 2024 style).

## Getting started

Clone with submodules (or initialize them afterward):

```sh
git clone --recurse-submodules <this repo>
# or, if already cloned:
git submodule update --init --recursive
```

Install Python dependencies (Python 3.10+ is fine):

```sh
pip install torch torchvision kagglehub pillow matplotlib numpy
# plus whatever external/yolov5 needs: see external/yolov5/requirements.txt
pip install -r external/yolov5/requirements.txt
```

`train.py` and `test.py` download the dataset at runtime via
`kagglehub.dataset_download("fareselmenshawii/license-plate-dataset")`,
which needs a Kaggle API token in `~/.kaggle/kaggle.json` (or in the
`KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## Running

Train (produces `yolov5_retrained.pt`):

```sh
python train.py
```

Inference on the validation split (reads `yolov5_retrained.pt`,
writes `output_{0..4}.png`):

```sh
python test.py
```


## Data

- Kaggle: <https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset>
- YOLOv5 reference: <https://github.com/ultralytics/yolov5>
