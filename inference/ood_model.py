import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cfg import *


class OOD_DETECTOR(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        
        in_feats = self.backbone.classifier[0].in_features 
        self.backbone.classifier = nn.Identity()

        self.head = nn.Linear(in_feats, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        logits = self.head(feats).squeeze(1)
        return logits, feats

def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC for binary labels (0/1) without sklearn.

    Uses the rank-based Mann–Whitney U interpretation.
    Returns NaN if one class is missing.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i:j]].mean())
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics_from_probs(y_true: np.ndarray, p_ood: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(np.int64)
    p_ood = np.asarray(p_ood).astype(np.float64)

    y_pred = (p_ood > threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    auc = binary_auroc(y_true, p_ood)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float(acc),
        "recall": float(recall),
        "auroc": float(auc),
    }

def load_checkpoint(model: nn.Module, ckpt_path: Union[str, Path], device: str) -> dict:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return ckpt
