from __future__ import annotations

import sys
import os
import numpy as np
import torch
from PIL import Image

# Add detrex to path so we can import the model
DETREX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, DETREX_ROOT)
sys.path.insert(0, os.path.join(DETREX_ROOT, "projects", "dino"))

from detectron2.config import LazyConfig, instantiate

CONFIG_PATH = os.path.join(DETREX_ROOT, "projects", "dino", "configs", "dino_swin_large_chess_finetune.py")
CHECKPOINT_PATH = os.path.join(DETREX_ROOT, "output", "dino_chess_v25", "model_best.pth")
CONFIDENCE_THRESHOLD = 0.3

DINO_TO_ENCODING = {
    10: 0,   # white-pawn   -> 0
    12: 1,   # white-rook   -> 1
    9:  2,   # white-knight -> 2
    7:  3,   # white-bishop -> 3
    11: 4,   # white-queen  -> 4
    8:  5,   # white-king   -> 5
    4:  6,   # black-pawn   -> 6
    6:  7,   # black-rook   -> 7
    3:  8,   # black-knight -> 8
    1:  9,   # black-bishop -> 9
    5:  10,  # black-queen  -> 10
    2:  11,  # black-king   -> 11
}

EMPTY = 12

# Singleton model cache
_model = None


def _load_model():
    """Load the DINO model once and cache it."""
    global _model
    if _model is not None:
        return _model

    cfg = LazyConfig.load(CONFIG_PATH)
    _model = instantiate(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device)
    _model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    _model.load_state_dict(state_dict, strict=False)
    return _model


def predict_pieces(img: Image.Image) -> list[list[int]]:
    """
    Predict the chessboard state from a single RGB image, using the DINO model.
    Returns an 8x8 list of integers, where each integer represents the piece type,
    as written in CLASS_ENCODING in cfg.py.

    input: PIL Image
    output: 8x8 list of integers, where out[0][0] is the top-left corner,
            out[0][7] is the top-right corner, out[7][0] is the bottom-left corner,
            and out[7][7] is the bottom-right corner.
    """
    model = _load_model()
    device = next(model.parameters()).device

    # Convert PIL -> numpy RGB -> tensor
    image_rgb = np.array(img.convert("RGB"))
    H, W = image_rgb.shape[:2]
    image_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float()

    # Run DINO inference
    with torch.no_grad():
        outputs = model([{"image": image_tensor.to(device), "height": H, "width": W}])

    inst = outputs[0]["instances"]
    boxes = inst.pred_boxes.tensor.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    classes = inst.pred_classes.cpu().numpy()

    # Filter by confidence
    mask = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    # Initialize 8x8 board as all empty (12)
    board = [[EMPTY] * 8 for _ in range(8)]
    best_score = [[0.0] * 8 for _ in range(8)]

    # Square sizes
    sq_w = W / 8.0
    sq_h = H / 8.0

    # Assign each detection to the closest square by center point
    for box, score, cls in zip(boxes, scores, classes):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0

        col = int(cx / sq_w)
        row = int(cy / sq_h)
        col = max(0, min(col, 7))
        row = max(0, min(row, 7))

        encoding = DINO_TO_ENCODING.get(cls, -1)
        if encoding < 0:
            continue

        # Keep highest confidence detection per square
        if score > best_score[row][col]:
            board[row][col] = encoding
            best_score[row][col] = score

    return board