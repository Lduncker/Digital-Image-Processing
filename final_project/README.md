# License Plate Detection with YOLOv5

Final project for Digital Image Processing (Worcester Polytechnic
Institute). Authors: Lucian Duncker, Joshua Pasino.

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

## Results

Qualitative predictions on the validation split are saved as
`output_0.png` through `output_4.png`. See the report for full
discussion of the results.

## Known issues

- `train.py` and `test.py` currently build dataset paths with Windows
  separators (`path + "\\images\\train"`). On macOS/Linux these need to
  become `"/images/train"` (or, better, `os.path.join(path, "images",
  "train")`). Fix at the two `ImageDataset(...)` construction sites in
  each script before running outside Windows.
- 20 epochs is short for YOLOv5; accuracy would improve with a longer
  schedule (100 epochs, cosine LR, SGD + momentum 0.937), which is the
  standard Ultralytics recipe.
- No quantitative metrics (mAP@0.5, mAP@0.5:0.95) are reported. Adding
  `external/yolov5/val.py` against `yolov5_retrained.pt` would cover
  that.

## Data

- Kaggle: <https://www.kaggle.com/datasets/fareselmenshawii/license-plate-dataset>
- YOLOv5 reference: <https://github.com/ultralytics/yolov5>
