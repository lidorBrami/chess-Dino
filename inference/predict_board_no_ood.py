from __future__ import annotations

from PIL import Image
import torch
import numpy as np
from pieces_detection import predict_pieces
from cfg import *


def predict_board(image: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(image)
    pieces = predict_pieces(img)
    return torch.tensor(pieces, dtype=torch.long).cpu()
