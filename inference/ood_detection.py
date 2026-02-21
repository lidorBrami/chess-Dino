from __future__ import annotations
import csv
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from ood_model import OOD_DETECTOR, load_checkpoint, compute_metrics_from_probs
from cfg import *


def grid_edges(total: int) -> List[int]:
    return [round(i * total / 8) for i in range(9)]

def get_single_square(img: Image.Image, row: int, column: int, x_edges: List[int], y_edges: List[int]) -> Image.Image:
    width, height = img.size
    x0, x1 = x_edges[column], x_edges[column + 1]
    y0, y1 = y_edges[row], y_edges[row + 1]
    bx0 = max(0, x0 - OUTER_BORDER_PX)
    by0 = max(0, y0 - OUTER_BORDER_PX)
    bx1 = min(width, x1 + OUTER_BORDER_PX)
    by1 = min(height, y1 + OUTER_BORDER_PX)
    return img.crop((bx0, by0, bx1, by1))

def corp_and_Iterate_squares(img: Image.Image) -> list[list[bool]]:
    model = OOD_DETECTOR(pretrained=PRETRAINED)
    load_checkpoint(model, CKPT_PATH, DEVICE)

    width, height = img.size
    x_edges = grid_edges(width)
    y_edges = grid_edges(height)
    output = [[False]*8 for _ in range(8)]

    for row in range(8):
        for column in range(8):
            crop = get_single_square(img, row, column, x_edges, y_edges)
            _, is_ood = detect_ood(crop, model)
            output[row][column] = is_ood
    return output


@torch.no_grad()
def detect_ood(image: Image.Image, model: OOD_DETECTOR):
    model.eval()
    device = next(model.parameters()).device

    tfm = T.Compose([
        T.Resize((RESIZE_IMG_SIZE, RESIZE_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = image.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    logits, _ = model(x)
    prob = float(torch.sigmoid(logits)[0].item())
    return prob, (prob > THRESHOLD_PROB)


def main():
    model = OOD_DETECTOR(pretrained=PRETRAINED)
    load_checkpoint(model, CKPT_PATH, DEVICE)

    id_folder = Path(PARENT_FOLDER) / EVAL_MODEL_ON / ID_CLASS_DIR
    ood_folder = Path(PARENT_FOLDER) / EVAL_MODEL_ON / OOD_CLASS_DIR

    rows, y_true, p_ood = [], [], []

    for photo_path in id_folder.glob("*.jpg"):
        img = Image.open(photo_path).convert("RGB")
        prob, _ = detect_ood(img, model)
        rows.append((str(photo_path), 0, prob))
        y_true.append(0)
        p_ood.append(prob)

    for photo_path in ood_folder.glob("*.jpg"):
        img = Image.open(photo_path).convert("RGB")
        prob, _ = detect_ood(img, model)
        rows.append((str(photo_path), 1, prob))
        y_true.append(1)
        p_ood.append(prob)

    metrics = compute_metrics_from_probs(np.array(y_true), np.array(p_ood), threshold=THRESHOLD_PROB)

    out_csv = Path(OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "p_ood"])
        w.writeheader()
        for (path_str, label, prob) in rows:
            w.writerow({"path": path_str, "label": int(label), "p_ood": float(prob)})

    return metrics


if __name__ == "__main__":
    main()
