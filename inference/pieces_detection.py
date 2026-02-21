from __future__ import annotations

import sys
import os
import numpy as np
import torch
from PIL import Image

DETREX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, DETREX_ROOT)
sys.path.insert(0, os.path.join(DETREX_ROOT, "projects", "dino"))

from detectron2.config import LazyConfig, instantiate

CONFIG_PATH = os.path.join(DETREX_ROOT, "projects", "dino", "configs", "dino_swin_large_chess_finetune.py")
CHECKPOINT_PATH = os.path.join(DETREX_ROOT, "weights", "dino_chess_model.pth")
CONFIDENCE_THRESHOLD = 0.3

DINO_TO_ENCODING = {
    10: 0, 12: 1, 9: 2, 7: 3, 11: 4, 8: 5,
    4: 6, 6: 7, 3: 8, 1: 9, 5: 10, 2: 11,
}

EMPTY = 12

_model = None


def _load_model():
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
    model = _load_model()
    device = next(model.parameters()).device

    image_rgb = np.array(img.convert("RGB"))
    H, W = image_rgb.shape[:2]
    image_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float()

    with torch.no_grad():
        outputs = model([{"image": image_tensor.to(device), "height": H, "width": W}])

    inst = outputs[0]["instances"]
    boxes = inst.pred_boxes.tensor.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    classes = inst.pred_classes.cpu().numpy()

    mask = scores >= CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    board = [[EMPTY] * 8 for _ in range(8)]
    best_score = [[0.0] * 8 for _ in range(8)]

    sq_w = W / 8.0
    sq_h = H / 8.0

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

        if score > best_score[row][col]:
            board[row][col] = encoding
            best_score[row][col] = score

    return board
