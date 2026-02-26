import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data.datasets import register_coco_instances
from detrex.config import get_config
from detrex.modeling.matcher import HungarianMatcher
from .models.dino_swin_large_384 import model
from projects.dino.modeling.weighted_criterion import WeightedDINOCriterion

NUM_CLASSES = 13

CLASS_WEIGHTS = [
    1.0,    # background
    3.0,    # black-bishop
    1.0,    # black-king
    1.0,    # black-knight
    1.0,    # black-pawn
    3.0,    # black-queen
    18.0,   # black-rook
    10.0,   # white-bishop
    1.0,    # white-king
    1.0,    # white-knight
    1.0,    # white-pawn
    3.0,    # white-queen
    2.0,    # white-rook
]

import os as _os
from detectron2.data import DatasetCatalog as _DC
DATA_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "data", "dino"))

def _safe_register(name, meta, json_file, image_root):
    if name not in _DC.list():
        register_coco_instances(name, meta, json_file, image_root)

_safe_register("chess_train", {}, f"{DATA_ROOT}/train/train/_annotations.coco.json", f"{DATA_ROOT}/train/train")
_safe_register("chess_game7", {}, f"{DATA_ROOT}/train/train/game7/_annotations.coco.json", f"{DATA_ROOT}/train/train/game7")
_safe_register("chess_game6", {}, f"{DATA_ROOT}/train/train/game6/_annotations.coco.json", f"{DATA_ROOT}/train/train/game6")

for game_name in ["game4.3", "game4.4", "game6.3", "game6.4", "game7.3", "game7.4"]:
    _safe_register(f"chess_{game_name.replace('.', '_')}", {}, f"{DATA_ROOT}/train/train/{game_name}/_annotations.coco.json", f"{DATA_ROOT}/train/train/{game_name}")

for train_name in ["train.7"]:
    _safe_register(f"chess_{train_name.replace('.', '_')}", {}, f"{DATA_ROOT}/train/train/{train_name}/_annotations.coco.json", f"{DATA_ROOT}/train/train/{train_name}")

for folder in ["more_chess", "toAdd", "n1", "n2"]:
    _safe_register(f"chess_{folder}", {}, f"{DATA_ROOT}/train/train/{folder}/_annotations.coco.json", f"{DATA_ROOT}/train/train/{folder}")

for gname in ["game9", "game10.2", "game11.1", "game11.2", "game11.3", "game12"]:
    _safe_register(f"chess_{gname.replace('.', '_')}", {}, f"{DATA_ROOT}/train/train/{gname}/_annotations.coco.json", f"{DATA_ROOT}/train/train/{gname}")

_safe_register("chess_val_merged", {}, f"{DATA_ROOT}/val/game2/_annotations_merged.coco.json", f"{DATA_ROOT}/val/game2")

dataloader = get_config("common/data/coco_detr.py").dataloader

dataloader.train.mapper.augmentation = [
    L(T.RandomFlip)(),
    L(T.ResizeShortestEdge)(short_edge_length=(384, 416, 448, 480, 512), max_size=640, sample_style="choice"),
]
dataloader.train.mapper.augmentation_with_crop = [
    L(T.RandomFlip)(),
    L(T.ResizeShortestEdge)(short_edge_length=(256, 320, 384), sample_style="choice"),
    L(T.RandomCrop)(crop_type="absolute_range", crop_size=(256, 400)),
    L(T.ResizeShortestEdge)(short_edge_length=(384, 416, 448, 480, 512), max_size=640, sample_style="choice"),
]
dataloader.test.mapper.augmentation = [
    L(T.ResizeShortestEdge)(short_edge_length=512, max_size=640),
]

optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

model.num_classes = NUM_CLASSES

model.criterion = L(WeightedDINOCriterion)(
    num_classes=NUM_CLASSES,
    matcher=L(HungarianMatcher)(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
        cost_class_type="focal_loss_cost", alpha=0.25, gamma=2.0,
    ),
    weight_dict={
        "loss_class": 5.0, "loss_bbox": 2.0, "loss_giou": 2.0,
        "loss_class_dn": 5.0, "loss_bbox_dn": 2.0, "loss_giou_dn": 2.0,
    },
    loss_class_type="focal_loss",
    alpha=0.25, gamma=2.0,
    two_stage_binary_cls=False,
    class_weights=CLASS_WEIGHTS,
)

train.init_checkpoint = _os.path.join(DATA_ROOT, "..", "..", "weights", "dino_chess_model.pth")
train.output_dir = "./output/dino_chess"

import copy
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

train.max_iter = 5000
train.eval_period = 200
train.log_period = 20
train.checkpointer.period = 200

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

optimizer.lr = 1e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 2

dataloader.train.dataset.names = (
    "chess_train", "chess_game7", "chess_game6",
    "chess_game4_3", "chess_game4_4",
    "chess_game6_3", "chess_game6_4",
    "chess_game7_3", "chess_game7_4",
    "chess_train_7",
    "chess_more_chess", "chess_toAdd", "chess_n1", "chess_n2",
    "chess_game9",
    "chess_game10_2", "chess_game11_1", "chess_game11_2", "chess_game11_3", "chess_game12",
)
dataloader.test.dataset.names = "chess_val_merged"
