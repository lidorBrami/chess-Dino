import os
import torch

AUG_BRIGHTNESS = 0.25
AUG_CONTRAST   = 0.25
AUG_SATURATION = 0.15
AUG_HUE        = 0.02

ID_CLASS_DIR = "0"
OOD_CLASS_DIR = "2"
CLASS_ENCODING = {
    "White Pawn": 0, "White Rook": 1, "White Knight": 2, "White Bishop": 3,
    "White Queen": 4, "White King": 5, "Black Pawn": 6, "Black Rook": 7,
    "Black Knight": 8, "Black Bishop": 9, "Black Queen": 10, "Black King": 11,
    "Empty Square": 12, "OOD": 13
}

CLASS_TO_FEN = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
    12: "", 13: ""
}

OUTPUT_IMAGE_SIZE = 480
RESIZE_IMG_SIZE = 96
OUTER_BORDER_PX = 10

PRETRAINED = True
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
EPOCH = 30
NUM_WORKERS = 4
LR = 1e-3
THRESHOLD_PROB = 0.94

_CFG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_CFG_DIR, ".."))

LOCATION_TO_SAVE_IMAGE = os.path.join(_ROOT_DIR, "results")
PARENT_FOLDER = os.path.join(_ROOT_DIR, "data", "ood")
CKPT_PATH = os.path.join(_ROOT_DIR, "weights", "mobilenet_v3_small_weights.pth")
EVAL_MODEL_ON = "test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLS_WEIGHTS_OUT_DIR = os.path.join(_CFG_DIR, "weights_ood_mobilenet_v3_small_bin")
WEIGHTS_NAME = "mobilenet_v3_small_bin"
CLS_LOG_CSV = "train_log.csv"

OUT_CSV = f"./{CLS_WEIGHTS_OUT_DIR}/{EVAL_MODEL_ON}_scores_threashhold_{THRESHOLD_PROB}.csv"
