
import csv
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from predict_board import predict_board

FEN_TO_ENC = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11,
}
EMPTY = 12
OOD = 13

ENC_TO_CH = {0:'P',1:'R',2:'N',3:'B',4:'Q',5:'K',
             6:'p',7:'r',8:'n',9:'b',10:'q',11:'k',12:'.',13:'X'}


def fen_to_board_with_ood(fen):
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


def compare_boards_with_ood(pred_tensor, gt_board):
    """Compare prediction vs GT. ALL 64 squares count in accuracy.
    GT 'x' (-1) maps to OOD (13). Every square is compared directly.
    """
    correct = 0
    total = 64
    wrong = []

    for r in range(8):
        for c in range(8):
            p = pred_tensor[r][c].item()
            g = gt_board[r][c]

            # Convert GT OOD (-1) to 13 so we compare same encoding
            if g == -1:
                g = OOD

            if p == g:
                correct += 1
            else:
                gt_ch = 'x' if g == OOD else ENC_TO_CH.get(g, '?')
                pr_ch = 'X' if p == OOD else ENC_TO_CH.get(p, '?')
                wrong.append((r, c, gt_ch, pr_ch))

    return correct, total, wrong


def load_txt_dataset(game_dir):
    fen_path = os.path.join(game_dir, 'FEN.txt')
    img_dir = os.path.join(game_dir, 'images')
    if not os.path.exists(img_dir):
        img_dir = game_dir

    with open(fen_path) as f:
        fens = [line.strip() for line in f if line.strip()]

    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    pairs = []
    for i, img_name in enumerate(images):
        if i < len(fens):
            pairs.append((os.path.join(img_dir, img_name), fens[i]))
    return pairs


def load_csv_dataset(csv_path, img_dir):
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


def eval_game(pairs, game_name, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_correct = 0
    total_squares = 0
    total_images = 0
    per_class_correct = {}
    per_class_total = {}
    per_class_pred_total = {}
    per_class_pred_correct = {}

    for img_path, gt_fen in pairs:
        fname = os.path.basename(img_path)
        img = np.array(Image.open(img_path).convert('RGB'))
        pred_tensor = predict_board(img)
        gt_board = fen_to_board_with_ood(gt_fen)

        correct, squares, wrong = compare_boards_with_ood(pred_tensor, gt_board)
        total_correct += correct
        total_squares += squares
        total_images += 1

        for r in range(8):
            for c in range(8):
                g = gt_board[r][c]
                p = pred_tensor[r][c].item()
                if g == -1:
                    g = OOD
                gt_name = 'x' if g == OOD else ENC_TO_CH.get(g, '?')
                pr_name = 'x' if p == OOD else ENC_TO_CH.get(p, '?')

                per_class_total[gt_name] = per_class_total.get(gt_name, 0) + 1
                if p == g:
                    per_class_correct[gt_name] = per_class_correct.get(gt_name, 0) + 1

                per_class_pred_total[pr_name] = per_class_pred_total.get(pr_name, 0) + 1
                if p == g:
                    per_class_pred_correct[pr_name] = per_class_pred_correct.get(pr_name, 0) + 1

        acc = correct / squares * 100 if squares > 0 else 0
        if wrong:
            print(f"  {fname}: {correct}/{squares} = {acc:.1f}% (wrong: {len(wrong)})")
            for r, c, gt_ch, pr_ch in wrong[:3]:
                print(f"    ({r},{c}): GT={gt_ch} Pred={pr_ch}")
            if len(wrong) > 3:
                print(f"    ... and {len(wrong)-3} more")
        else:
            print(f"  {fname}: {correct}/{squares} = 100.0% PERFECT")

    overall_acc = total_correct / total_squares * 100 if total_squares > 0 else 0
    print(f"\n{'='*60}")
    print(f"  {game_name}: {total_correct}/{total_squares} = {overall_acc:.1f}%")
    print(f"  Images: {total_images}")
    print(f"{'='*60}")

    print(f"\n  Per-class Recall & Precision:")
    print(f"    {'Class':5s}  {'Recall':>15s}  {'Precision':>15s}")
    print(f"    {'-'*5}  {'-'*15}  {'-'*15}")
    for cls_name in ['K','Q','R','B','N','P','k','q','r','b','n','p','.','x']:
        gt_t = per_class_total.get(cls_name, 0)
        gt_c = per_class_correct.get(cls_name, 0)
        pr_t = per_class_pred_total.get(cls_name, 0)
        pr_c = per_class_pred_correct.get(cls_name, 0)
        if gt_t > 0 or pr_t > 0:
            label = "OOD" if cls_name == 'x' else cls_name
            recall = f"{gt_c}/{gt_t} = {gt_c/gt_t*100:.1f}%" if gt_t > 0 else "N/A"
            precision = f"{pr_c}/{pr_t} = {pr_c/pr_t*100:.1f}%" if pr_t > 0 else "N/A"
            print(f"    {label:5s}  {recall:>15s}  {precision:>15s}")

    return overall_acc


if __name__ == "__main__":
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    EVAL_DATA = os.path.join(ROOT, "data", "eval")

    txt_games = ["game13"]

    for game_name in txt_games:
        game_dir = os.path.join(EVAL_DATA, game_name)
        fen_path = os.path.join(game_dir, "FEN.txt")
        img_dir = os.path.join(game_dir, "images")

        if not os.path.exists(fen_path):
            print(f"Skipping {game_name}: {fen_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  {game_name.upper()} EVALUATION")
        print(f"{'='*60}")

        with open(fen_path) as f:
            fens = [line.strip() for line in f if line.strip()]
        images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        pairs = []
        for i, img_name in enumerate(images):
            if i < len(fens):
                pairs.append((os.path.join(img_dir, img_name), fens[i]))

        print(f"  Loaded {len(pairs)} image-FEN pairs")
        eval_game(pairs, game_name)

    csv_games = ["game5", "game2"]

    for game_name in csv_games:
        game_dir = os.path.join(EVAL_DATA, game_name)
        csv_path = os.path.join(game_dir, f"{game_name}.csv")
        img_dir = os.path.join(game_dir, "images")

        if not os.path.exists(csv_path):
            print(f"Skipping {game_name}: {csv_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  {game_name.upper()} EVALUATION")
        print(f"{'='*60}")
        pairs = load_csv_dataset(csv_path, img_dir)
        print(f"  Loaded {len(pairs)} image-FEN pairs")
        eval_game(pairs, game_name)
irs, game_name)
