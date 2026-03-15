import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "inference")))
from ood_model import OOD_DETECTOR, binary_auroc
from cfg import *


class SquaresDataset(Dataset):
    """Binary dataset for OOD detection.

    Expected folders:
        {PARENT_FOLDER}/{split}/0  (ID)
        {PARENT_FOLDER}/{split}/2  (OOD)

    Returns:
        x: Tensor (3,H,W)
        y: int {0,1} where 1=OOD
        path: str
    """

    def __init__(self, root: str, split: str):
        self.root = Path(root) / split

        self.class_map = {
            ID_CLASS_DIR: 0,
            OOD_CLASS_DIR: 1,
        }
        self.items: List[Tuple[Path, int]] = []
        for folder_name, y in self.class_map.items():
            d = self.root / folder_name
            if not d.exists():
                continue
            for p in d.glob("*.jpg"):
                self.items.append((p, y))

        if split == "train":
            self.tfm = T.Compose([
                T.Resize((RESIZE_IMG_SIZE, RESIZE_IMG_SIZE)),
                T.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST,
                              saturation=AUG_SATURATION, hue=AUG_HUE),
                T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.90, 1.10)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                # keep ImageNet normalization for pretrained ResNet
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.RandomErasing(p=0.20, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
            ])
        else:
            self.tfm = T.Compose([
                T.Resize((RESIZE_IMG_SIZE, RESIZE_IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), int(y), str(p)


def _make_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    labels_t = torch.tensor(labels, dtype=torch.long)
    n0 = int((labels_t == 0).sum())
    n1 = int((labels_t == 1).sum())
    if n0 == 0 or n1 == 0:
        raise RuntimeError(f"Need both classes in training set, got n0={n0}, n1={n1}")

    w0 = 1.0 / n0
    w1 = 1.0 / n1
    weights = torch.where(labels_t == 0, torch.tensor(w0), torch.tensor(w1)).double()

    # sample with replacement so each epoch has stable size
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def main():
    out_dir = Path(CLS_WEIGHTS_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SquaresDataset(PARENT_FOLDER, "train")
    val_ds = SquaresDataset(PARENT_FOLDER, "val")

    train_labels = [y for _, y in train_ds.items]
    train_sampler = _make_balanced_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = OOD_DETECTOR(pretrained=PRETRAINED).to(DEVICE)

    n0 = sum(1 for y in train_labels if y == 0)
    n1 = sum(1 for y in train_labels if y == 1)
    pos_weight = torch.tensor([max(1.0, n0 / max(1, n1))], device=DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log_path = out_dir / CLS_LOG_CSV
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_mean", "train_min", "train_max",
            "val_mean", "val_min", "val_max",
            "train_auroc", "val_auroc",
        ])

    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(1, EPOCH + 1):
        model.train()
        train_losses = []
        y_true = []
        y_score = []

        for x, y, _ in tqdm.tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x = x.to(DEVICE)
            y_t = y.to(device=DEVICE, dtype=torch.float32)

            logits, _ = model(x)
            loss = criterion(logits, y_t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(float(loss.item()))

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                y_true.extend([int(v) for v in y])
                y_score.extend(probs.detach().cpu().numpy().tolist())

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        train_min = float(np.min(train_losses)) if train_losses else float("nan")
        train_max = float(np.max(train_losses)) if train_losses else float("nan")
        train_auc = binary_auroc(np.array(y_true), np.array(y_score)) if y_true else float("nan")

        val_losses = []
        val_scores = []
        val_labels = []
        model.eval()
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y_t = y.to(device=DEVICE, dtype=torch.float32)
                logits, _ = model(x)
                vloss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
                val_losses.append(float(vloss.item()))

                probs = torch.sigmoid(logits)
                val_labels.extend([int(v) for v in y])
                val_scores.extend(probs.detach().cpu().numpy().tolist())

        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        val_min = float(np.min(val_losses)) if val_losses else float("nan")
        val_max = float(np.max(val_losses)) if val_losses else float("nan")
        val_auc = binary_auroc(np.array(val_labels), np.array(val_scores)) if val_labels else float("nan")

        saved_msg = ""
        if val_mean < best_val_loss:
            if best_ckpt_path and best_ckpt_path.exists():
                best_ckpt_path.unlink()
            best_val_loss = val_mean
            best_ckpt_path = out_dir / f"model_best.pth"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "resize_img_size": RESIZE_IMG_SIZE,
                    "pretrained": PRETRAINED,
                    "pos_weight": float(pos_weight.item()),
                    "epoch": epoch,
                    "val_loss": val_mean,
                },
                best_ckpt_path,
            )
            saved_msg = " -> saved model_best.pth"

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_mean, train_min, train_max,
                val_mean, val_min, val_max,
                train_auc, val_auc,
            ])

        print(
            f"[{epoch:03d}] train_loss={train_mean:.4f} val_loss={val_mean:.4f} "
            f"train_AUROC={train_auc:.4f} val_AUROC={val_auc:.4f}{saved_msg}"
        )

    print(f"Done. Best val_loss={best_val_loss:.4f} | Weights: {best_ckpt_path}")


if __name__ == "__main__":
    main()
