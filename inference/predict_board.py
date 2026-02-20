from __future__ import annotations

from PIL import Image, ImageDraw
import torch
import numpy as np
from pathlib import Path
from ood_detection import corp_and_Iterate_squares
from pieces_detection import predict_pieces
from cfg import *



def ood_x_marks(board_array: list[list[int]], ood_mask: list[list[bool]]) -> list[tuple[int, int]]:
    marks = []
    for row in range(8):
        for col in range(8):
            if board_array[row][col] == CLASS_ENCODING["Empty Square"] and ood_mask[row][col]:
                marks.append((row, col))
    return marks


def print_and_save_board(board_array, ood_mask, name_of_file: str):
    Path(LOCATION_TO_SAVE_IMAGE).mkdir(parents=True, exist_ok=True)
    sq = OUTPUT_IMAGE_SIZE // 8
    img = Image.new("RGB", (OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), "white")
    d = ImageDraw.Draw(img)

    # draw squares + pieces + red X
    for r in range(8):
        for c in range(8):
            x0, y0 = c*sq, r*sq
            x1, y1 = x0+sq, y0+sq
            d.rectangle([x0,y0,x1,y1], fill=(240,217,181) if (r+c)%2==0 else (181,136,99))

            # red X on OOD empty squares only
            if board_array[r][c] == CLASS_ENCODING["Empty Square"] and ood_mask[r][c]:
                pad = int(sq*0.18)
                w = max(2, int(sq*0.06))
                d.line([x0+pad,y0+pad,x1-pad,y1-pad], fill="red", width=w)
                d.line([x0+pad,y1-pad,x1-pad,y0+pad], fill="red", width=w)

    img.save(f"{LOCATION_TO_SAVE_IMAGE}/{name_of_file}.png")

    
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    """
    img = Image.fromarray(image)
    ood_mask = corp_and_Iterate_squares(img)
    pieces = predict_pieces(img)
    print_and_save_board(pieces, ood_mask, "predicted_board")

    for row in range(8):
        for column in range(8):
            if ood_mask[row][column]:
                # Only mark as OOD if DINO didn't detect a piece (empty square)
                # This prevents OOD false positives from overriding real piece detections
                if pieces[row][column] == CLASS_ENCODING["Empty Square"]:
                    pieces[row][column] = CLASS_ENCODING["OOD"]
    return torch.tensor(pieces, dtype=torch.long).cpu()


def main():
    img = np.array(Image.open("./frame_020024.jpg").convert("RGB"))
    print(predict_board(img))


if __name__ == "__main__":
    main()