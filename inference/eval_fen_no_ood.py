"""Eval FEN without OOD detection — ablation study."""
import csv
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from predict_board_no_ood import predict_board

FEN_TO_ENC = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11,
}
EMPTY = 12
OOD = 13

ENC_TO_CH = {0:'P',1:'R',2:'N',3:'B',4:'Q',5:'K',
             6:'p',7:'r',8:'n',9:'b',10:'q',11:'k',12:'.',13:'X'}


def fen_to_board(fen):
    board = [[EMPTY] * 8 for _ in range(8)]
    rows = fen.split('/')
    for r, row_str in enumerate(rows):
        if r >= 8:
            break
        c = 0
        for ch in row_str:
            if c >= 8:
                break
            if ch.isdigit():
                c += int(ch)
            elif ch == 'x':
                board[r][c] = -1
                c += 1
            else:
                board[r][c] = FEN_TO_ENC.get(ch, EMPTY)
                c += 1
    return board


def compare(pred, gt):
    correct = 0
    total = 64
    for r in range(8):
        for c in range(8):
            p = pred[r][c].item()
            g = gt[r][c]
            if g == -1:
                g = OOD
            # Without OOD: model never predicts 13, so GT OOD squares will be wrong
            # unless model predicts empty (12) for them
            if p == g:
                correct += 1
    return correct, total


def load_csv(csv_path, img_dir):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    pairs = []
    for row in rows:
        frame_num = int(row['from_frame'])
        fen = row['fen']
        fname = f'frame_{frame_num:06d}.jpg'
        img_path = os.path.join(img_dir, fname)
        if os.path.exists(img_path):
            pairs.append((img_path, fen))
    return pairs


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    EVAL_DATA = os.path.join(ROOT, "data", "eval")

    all_correct = 0
    all_total = 0

    for game_name in ["game13"]:
        game_dir = os.path.join(EVAL_DATA, game_name)
        fen_path = os.path.join(game_dir, "FEN.txt")
        img_dir = os.path.join(game_dir, "images")
        if not os.path.exists(fen_path):
            continue
        with open(fen_path) as f:
            fens = [line.strip() for line in f if line.strip()]
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        pairs = [(os.path.join(img_dir, images[i]), fens[i]) for i in range(min(len(images), len(fens)))]

        game_correct = 0
        game_total = 0
        for img_path, gt_fen in pairs:
            img = np.array(Image.open(img_path).convert('RGB'))
            pred = predict_board(img)
            gt = fen_to_board(gt_fen)
            c, t = compare(pred, gt)
            game_correct += c
            game_total += t
        acc = game_correct / game_total * 100
        print(f"{game_name}: {game_correct}/{game_total} = {acc:.1f}%")
        all_correct += game_correct
        all_total += game_total

    for game_name in ["game5", "game2"]:
        game_dir = os.path.join(EVAL_DATA, game_name)
        csv_path = os.path.join(game_dir, f"{game_name}.csv")
        img_dir = os.path.join(game_dir, "images")
        if not os.path.exists(csv_path):
            continue
        pairs = load_csv(csv_path, img_dir)

        game_correct = 0
        game_total = 0
        for img_path, gt_fen in pairs:
            img = np.array(Image.open(img_path).convert('RGB'))
            pred = predict_board(img)
            gt = fen_to_board(gt_fen)
            c, t = compare(pred, gt)
            game_correct += c
            game_total += t
        acc = game_correct / game_total * 100
        print(f"{game_name}: {game_correct}/{game_total} = {acc:.1f}%")
        all_correct += game_correct
        all_total += game_total

    print(f"\nOverall (no OOD): {all_correct}/{all_total} = {all_correct/all_total*100:.1f}%")
