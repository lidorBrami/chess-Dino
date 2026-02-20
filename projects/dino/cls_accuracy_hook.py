"""
Custom hook to compute pure classification accuracy (no bbox) on train and val sets.
Runs every eval_period iterations. Logs to metrics via EventStorage.
"""
import os
import sys
import json
import numpy as np
import torch
import cv2
from collections import defaultdict

sys.path.insert(0, "/home/lidorbr/detrex")
sys.path.insert(0, "/home/lidorbr/detrex/projects/dino")

from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

CLASS_NAMES = [
    "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen",
    "black-rook", "white-bishop", "white-king", "white-knight", "white-pawn",
    "white-queen", "white-rook"
]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def load_coco_annotations(ann_path, img_dir):
    """Load COCO annotations, return (image_annotations, image_info)."""
    with open(ann_path) as f:
        coco = json.load(f)
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    image_annotations = defaultdict(list)
    for ann in coco['annotations']:
        image_annotations[ann['image_id']].append({
            'bbox': ann['bbox'],
            'category_name': cat_id_to_name[ann['category_id']]
        })
    image_info = {img['id']: os.path.join(img_dir, img['file_name']) for img in coco['images']}
    return image_annotations, image_info


def compute_cls_accuracy(model, image_annotations, image_info, max_images=None):
    """Run model on images, match by IoU, return pure classification accuracy and loss."""
    correct = 0
    total = 0
    all_pred_scores = []
    all_gt_labels = []

    items = list(image_annotations.items())
    if max_images and len(items) > max_images:
        indices = np.random.choice(len(items), max_images, replace=False)
        items = [items[i] for i in indices]

    for img_id, annotations in items:
        if img_id not in image_info:
            continue
        image_path = image_info[img_id]
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = image[:, :, ::-1]
        image_tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float()

        with torch.no_grad():
            outputs = model([{
                "image": image_tensor.to("cuda"),
                "height": image.shape[0],
                "width": image.shape[1],
            }])

        inst = outputs[0]["instances"]
        pred_boxes = inst.pred_boxes.tensor.cpu().numpy()
        pred_scores = inst.scores.cpu().numpy()
        pred_classes = inst.pred_classes.cpu().numpy()

        mask = pred_scores >= CONFIDENCE_THRESHOLD
        pred_boxes = pred_boxes[mask]
        pred_classes = pred_classes[mask]
        pred_scores_filtered = pred_scores[mask]

        gt_boxes = []
        gt_classes = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x+w, y+h])
            name = ann['category_name']
            if name in CLASS_NAMES:
                gt_classes.append(CLASS_NAMES.index(name))
            else:
                gt_classes.append(-1)

        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
        gt_classes = np.array(gt_classes)

        valid = gt_classes >= 0
        gt_boxes = gt_boxes[valid]
        gt_classes = gt_classes[valid]

        gt_matched = [False] * len(gt_classes)

        for pi, (pbox, pcls) in enumerate(zip(pred_boxes, pred_classes)):
            best_iou = 0
            best_gi = -1
            for gi, gbox in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou and iou >= IOU_THRESHOLD:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0:
                gt_matched[best_gi] = True
                pred_cls_mapped = pcls - 1
                total += 1
                if pred_cls_mapped == gt_classes[best_gi]:
                    correct += 1
                    all_pred_scores.append(pred_scores_filtered[pi])
                    all_gt_labels.append(1)  # correct
                else:
                    all_pred_scores.append(pred_scores_filtered[pi])
                    all_gt_labels.append(0)  # wrong

    accuracy = correct / total * 100 if total > 0 else 0
    # Compute cross-entropy-like loss: -log(score) for correct, -log(1-score) for wrong
    cls_loss = 0.0
    if all_pred_scores:
        for score, label in zip(all_pred_scores, all_gt_labels):
            s = max(min(score, 0.999), 0.001)
            if label == 1:
                cls_loss += -np.log(s)
            else:
                cls_loss += -np.log(1.0 - s)
        cls_loss /= len(all_pred_scores)

    return accuracy, correct, total, cls_loss


class ClsAccuracyHook(HookBase):
    """Hook that computes classification accuracy on train and val subsets."""

    def __init__(self, eval_period, train_ann_path, train_img_dir,
                 val_ann_path, val_img_dir, train_sample_size=200):
        self._period = eval_period
        self._train_sample_size = train_sample_size

        self._train_anns, self._train_info = load_coco_annotations(train_ann_path, train_img_dir)
        self._val_anns, self._val_info = load_coco_annotations(val_ann_path, val_img_dir)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return
        self._do_eval()

    def _do_eval(self):
        model = self.trainer.model
        model.eval()

        # Validation accuracy + loss (all images)
        val_acc, val_c, val_t, val_loss = compute_cls_accuracy(model, self._val_anns, self._val_info)

        # Training accuracy + loss (sample to save time)
        train_acc, train_c, train_t, train_loss = compute_cls_accuracy(
            model, self._train_anns, self._train_info, max_images=self._train_sample_size)

        storage = get_event_storage()
        storage.put_scalar("cls_acc/train", train_acc)
        storage.put_scalar("cls_acc/val", val_acc)
        storage.put_scalar("cls_loss/train", train_loss)
        storage.put_scalar("cls_loss/val", val_loss)

        print(f"\n[ClsAccuracy] Train: {train_c}/{train_t} = {train_acc:.1f}% (loss={train_loss:.4f}) | "
              f"Val: {val_c}/{val_t} = {val_acc:.1f}% (loss={val_loss:.4f})\n")

        model.train()
