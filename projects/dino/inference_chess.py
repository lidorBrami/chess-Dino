#!/usr/bin/env python
import os
import sys
import cv2
import torch
import argparse
import numpy as np

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

CLASS_NAMES = [
    "1", "black-bishop", "black-king", "black-knight", "black-pawn",
    "black-queen", "black-rook", "white-bishop", "white-king",
    "white-knight", "white-pawn", "white-queen", "white-rook",
]

# Colors for each class (BGR format)
CLASS_COLORS = {
    0: (128, 128, 128),  # gray - unused
    1: (0, 0, 0),        # black-bishop
    2: (0, 0, 0),        # black-king
    3: (0, 0, 0),        # black-knight
    4: (0, 0, 0),        # black-pawn
    5: (0, 0, 0),        # black-queen
    6: (0, 0, 0),        # black-rook
    7: (255, 255, 255),  # white-bishop
    8: (255, 255, 255),  # white-king
    9: (255, 255, 255),  # white-knight
    10: (255, 255, 255), # white-pawn
    11: (255, 255, 255), # white-queen
    12: (255, 255, 255), # white-rook
}

def register_datasets():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dino"))
    if "chess_train" not in DatasetCatalog.list():
        register_coco_instances("chess_train", {},
            os.path.join(data_root, "train/train/_annotations.coco.json"),
            os.path.join(data_root, "train/train"))
    if "chess_val" not in DatasetCatalog.list():
        register_coco_instances("chess_val", {},
            os.path.join(data_root, "val/valid/_annotations.coco.json"),
            os.path.join(data_root, "val/valid"))


def load_model(config_path, checkpoint_path, device="cuda"):
    cfg = LazyConfig.load(config_path)
    model = instantiate(cfg.model)
    model.to(device)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint_path)
    return model, cfg


def preprocess_image(image_path, device="cuda"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image_rgb.transpose(2, 0, 1).astype("float32"))
    inputs = [{"image": image_tensor, "height": height, "width": width}]
    return inputs, image


def visualize_detections(image, instances, score_threshold=0.5):
    vis_image = image.copy()
    if len(instances) == 0:
        return vis_image

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    print(f"\nDetected {len(boxes)} pieces (threshold={score_threshold}):")

    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class_{cls}"

        if "black" in class_name:
            box_color = (50, 50, 50)
            text_color = (255, 255, 255)
        else:
            box_color = (200, 200, 200)
            text_color = (0, 0, 0)

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, 2)
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), box_color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        print(f"  {i+1}. {class_name}: {score:.3f} at [{x1}, {y1}, {x2}, {y2}]")

    return vis_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="weights/dino_chess_model.pth")
    parser.add_argument("--config", type=str, default="projects/dino/configs/dino_swin_large_chess_finetune.py")
    args = parser.parse_args()

    register_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.config, args.checkpoint, device)

    inputs, original_image = preprocess_image(args.image, device)

    with torch.no_grad():
        outputs = model(inputs)

    instances = outputs[0]["instances"].to("cpu")
    vis_image = visualize_detections(original_image, instances, args.threshold)

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"output/{base_name}_detected.jpg"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, vis_image)
    print(f"\nSaved result to: {args.output}")


if __name__ == "__main__":
    main()
