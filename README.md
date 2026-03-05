<p align="center">
  <h1 align="center">♟️ Chess Piece Detection with DINO + OOD Detection</h1>
  <p align="center">
    <b>End-to-end chess board recognition from images using DINO object detection and out-of-distribution detection</b>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/PyTorch-1.10.1-red?logo=pytorch" alt="PyTorch">
    <img src="https://img.shields.io/badge/Detectron2-0.6-blue" alt="Detectron2">
    <img src="https://img.shields.io/badge/Python-3.7-green?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/CUDA-11.3-76B900?logo=nvidia" alt="CUDA">
  </p>
</p>

---

## 📋 Overview

A two-stage system that takes a chessboard image and outputs a complete board state (FEN notation):

1. **DINO Piece Detector** — Swin-Large backbone fine-tuned to detect and classify all 12 chess piece types
2. **OOD Detector** — MobileNetV3-Small binary classifier that identifies occluded/unknown squares (e.g., player's hand)

Built on top of [detrex](https://github.com/IDEA-Research/detrex) (DINO: DETR with Improved DeNoising Anchor Boxes).

---

## 🏗️ Project Structure

```
├── projects/dino/                   # DINO training (detrex framework)
│   ├── train_net.py                 #   Training script
│   ├── inference_chess.py           #   Single-image inference with visualization
│   ├── cls_accuracy_hook.py         #   Classification accuracy hook
│   ├── configs/
│   │   └── dino_swin_large_chess_finetune.py
│   └── modeling/
│       ├── dino.py                  #   DINO model
│       ├── weighted_criterion.py    #   Weighted loss for class imbalance
│       └── ood_detector.py          #   Mahalanobis OOD detector
│
├── projects/ood/                    # OOD detector training
│   └── train_ood.py                 #   OOD model training script
│
├── inference/                       # Inference & evaluation pipeline
│   ├── predict_board.py             #   Full board prediction (DINO + OOD)
│   ├── pieces_detection.py          #   DINO piece detection module
│   ├── ood_detection.py             #   OOD square detection module
│   ├── ood_model.py                 #   MobileNetV3 OOD model definition
│   ├── cfg.py                       #   Configuration constants
│   └── eval_fen_tests.py            #   FEN accuracy evaluation
│
├── data/
│   ├── dino/                        #   DINO piece detection data
│   │   ├── train/train/             #     Training (COCO format)
│   │   └── val/                     #     Validation (COCO format)
│   ├── ood/                         #   OOD training (binary: 0=ID, 2=OOD)
│   └── eval/                        #   End-to-end evaluation
│       ├── game13/                  #     images/ + FEN.txt
│       ├── game2/                   #     images/ + game2.csv
│       └── game5/                   #     images/ + game5.csv
│
├── output/                          # Training outputs
│   ├── dino_chess/                  #   DINO training checkpoints
│   └── ood/                         #   OOD training checkpoints
│
├── weights/                         # Pretrained model weights
│   ├── dino_chess_model.pth         #   Fine-tuned chess detector
│   └── mobilenet_v3_small_weights.pth  #   OOD detector
│
├── report/                          # Report figures & PDFs
├── detectron2/                      # Detectron2 framework (submodule)
├── detrex/                          # Detrex framework (included)
├── requirements.txt
└── README.md
```

---

## 🎯 Classes

14 detection classes — 12 piece types + background + OOD (occluded):

| ID | Class | ID | Class |
|:--:|:------|:--:|:------|
| 1 | ♝ black-bishop | 7 | ♗ white-bishop |
| 2 | ♚ black-king | 8 | ♔ white-king |
| 3 | ♞ black-knight | 9 | ♘ white-knight |
| 4 | ♟ black-pawn | 10 | ♙ white-pawn |
| 5 | ♛ black-queen | 11 | ♕ white-queen |
| 6 | ♜ black-rook | 12 | ♖ white-rook |
| 0 | background | **13** | **OOD (occluded)** |

---

## 🚀 Environment Setup

**Prerequisites:** A machine with an NVIDIA GPU and CUDA drivers installed. Verify with `nvidia-smi`.

### 1. Clone the repository

```bash
git clone --recursive https://github.com/lidorBrami/chess-Dino.git
cd chess-Dino
```

### 2. Create conda environment

```bash
conda create -n chess python=3.7 -y
conda activate chess
```

### 3. Install PyTorch (CUDA 11.3)

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
```

### 4. Install C++ compiler

A C++ compiler (`g++`) is required to build the detectron2 and detrex CUDA/C++ extensions. Install GCC 9.3 via conda (compatible with CUDA 11.3):

```bash
conda install -y -c conda-forge gcc_linux-64=9.3.0 gxx_linux-64=9.3.0 libxcrypt
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++
```

> **Note:** GCC 9.3 is recommended. GCC 11+ headers are incompatible with CUDA 11.3's nvcc parser.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Install detectron2

```bash
cd detectron2
pip install -e .
cd ..
```

### 7. Install detrex (with CUDA extensions)

Detrex requires CUDA extensions (MultiScaleDeformableAttention, DCNv3) compiled with GPU support. This step must be run on a machine with a GPU and CUDA toolkit installed.

**If your system GCC is <= 10 and CUDA 11.3 is installed:**

```bash
FORCE_CUDA=1 pip install -e .
```

**If your system GCC is 11+ (common on modern Linux):**

CUDA 11.3's nvcc cannot parse GCC 11+ headers. Use CUDA 11.8+ for compilation instead:

```bash
# If using environment modules (e.g., SLURM cluster):
module load cuda/11.8

# Or set CUDA_HOME manually to a CUDA >= 11.8 installation:
# export CUDA_HOME=/usr/local/cuda-11.8

rm -rf build/ detrex/_C*.so detrex.egg-info/
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include"
CC=/usr/bin/gcc CXX=/usr/bin/g++ FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation --no-deps
```

> **Note:** Set `TORCH_CUDA_ARCH_LIST` to match your GPU architecture (e.g., `8.6` for RTX 3090/4090, `7.5` for RTX 2080, `7.0` for V100). The compiled extensions are compatible with PyTorch's CUDA 11.3 runtime.

### 8. Patch detectron2 for Python 3.7

The bundled detectron2 uses `functools.cached_property` which requires Python 3.8+. Apply this patch:

In `detectron2/detectron2/utils/events.py`, replace:
```python
from functools import cached_property
```
with:
```python
try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache
    def cached_property(func):
        return property(lru_cache(maxsize=None)(func))
```


### 7. Download data and weights

The dataset and model weights are not included in this repository due to their size. Download them from our Google Drive:

> **[Download weights from Google Drive](https://drive.google.com/drive/folders/171O4a8FFRloupBf_Rth5bTv1XzCwL3J-)** 
>
> **[Download data from Google Drive](https://drive.google.com/drive/folders/132CT931JP1SleatsDDebEdmQ41fQaiIR)** 

After downloading, place them in the repository root:

```bash
mkdir -p weights data
# Place the following files:
# weights/dino_chess_model.pth                  (fine-tuned chess detector)
# weights/mobilenet_v3_small_weights.pth        (OOD detector)
# data/dino/                                    (training & validation data)
# data/eval/                                    (evaluation data)
```

Alternatively, download only the COCO pretrained weights for training from scratch:

```bash
mkdir -p weights
wget -P weights/ https://github.com/IDEA-Research/detrex-storage/releases/download/v0.2.0/dino_swin_large_384_4scale_36ep.pth
```

---

## 📁 Data Preparation

### DINO training data (COCO format)

```
data/dino/
├── train/
│   └── train/
│       ├── _annotations.coco.json
│       ├── *.jpg
│       ├── game6/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg
│       └── ...
└── val/
    └── game2/
        ├── _annotations_merged.coco.json
        └── *.jpg
```

### OOD training data

```
data/ood/
├── train/
│   ├── 0/   ← in-distribution square crops (*.jpg)
│   └── 2/   ← OOD square crops: hands, occluded (*.jpg)
├── val/
│   ├── 0/
│   └── 2/
└── test/
    ├── 0/
    └── 2/
```

### End-to-end evaluation data

```
data/eval/
├── game13/
│   ├── images/
│   └── FEN.txt
├── game2/
│   ├── images/
│   └── game2.csv
└── game5/
    ├── images/
    └── game5.csv
```

---

## 🏋️ Training

> **Important:** Training requires a machine with an NVIDIA GPU. If you are on a SLURM cluster, allocate a GPU node first:
> ```bash
> srun --partition=<gpu_partition> --gpus=1 --mem=32G --cpus-per-task=4 --time=04:00:00 --pty bash
> conda activate chess
> ```

### DINO — Piece Detection

Before training, update the following paths in `projects/dino/configs/dino_swin_large_chess_finetune.py`:

- **`train.init_checkpoint`** — path to the starting weights:
  - For fine-tuning from COCO: `weights/dino_swin_large_384_4scale_36ep.pth`
  - For continued training from a chess model: `weights/dino_chess_model.pth`
- **`train.output_dir`** — where to save training outputs
- **`CLASS_WEIGHTS`** — per-class weights for the weighted focal loss

```bash
# Single GPU
python projects/dino/train_net.py \
    --config-file projects/dino/configs/dino_swin_large_chess_finetune.py \
    --num-gpus 1

# Multi-GPU
python projects/dino/train_net.py \
    --config-file projects/dino/configs/dino_swin_large_chess_finetune.py \
    --num-gpus 2
```

<details>
<summary><b>Training hyperparameters</b></summary>

| Parameter | Value |
|-----------|-------|
| Backbone | Swin-Large 384, COCO pretrained |
| Iterations | 10,000 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-5 (backbone: 1e-6) |
| Weighted loss | Class weights for imbalance (e.g., black-rook 14x, white-bishop 14x) |
| Eval period | Every 200 iterations |
| Gradient clip | max_norm=0.1 |

</details>

Output: `./output/dino_chess/model_final.pth`

### OOD Detector

```bash
python projects/ood/train_ood.py
```

Trains MobileNetV3-Small binary classifier (ID vs OOD) for 30 epochs with balanced sampling.

Output: `./output/ood/`

---

## 🔍 Inference

Before running inference, ensure the model weights are placed in the `weights/` directory. The checkpoint path is configured in `inference/pieces_detection.py` (`CHECKPOINT_PATH` variable).

### Single image — bounding box visualization

```bash
python projects/dino/inference_chess.py \
    --image path/to/chessboard.jpg \
    --output path/to/output.jpg \
    --checkpoint weights/dino_chess_model.pth \
    --threshold 0.3
```

### Full board prediction — DINO + OOD → 8x8 board state
In order to predict a full board, as instructed in the assignament requirements, use predict_board(image: np.ndarray) -> torch.Tensor function implemented in 
inference/predict_board.py file (predict_board.py file inside inference folder)

Pipeline:
1. **OOD detection** — crops each of 64 squares, classifies ID vs OOD
2. **Piece detection** — runs DINO on the full image, assigns detections to grid squares
3. **Merge** — empty squares flagged as OOD get class 13; detected pieces are preserved

---

## 📈 Evaluation

### FEN accuracy evaluation

```bash
cd inference
python eval_fen_tests.py
```

Evaluates board prediction accuracy against ground-truth FEN annotations:

| Game | Format | Description |
|------|--------|-------------|
| game13 | TXT | FEN.txt + image folder |
| game2 | CSV | CSV with frame numbers |
| game5 | CSV | CSV with frame numbers |

**Output:** per-image accuracy, overall accuracy per game, per-class recall & precision.

> **Note:** All inference and evaluation require a CUDA GPU.

---

## 📦 Model Weights

| Model | File | Description |
|:------|:-----|:------------|
| DINO Swin-Large (pretrained) | `weights/dino_swin_large_384_4scale_36ep.pth` | COCO pretrained — download for training |
| DINO Chess (fine-tuned) | `weights/dino_chess_model.pth` | Fine-tuned chess piece detector |
| OOD Detector | `weights/mobilenet_v3_small_weights.pth` | MobileNetV3-Small binary classifier |
