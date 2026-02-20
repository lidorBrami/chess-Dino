"""Generate confusion matrix from eval_fen_tests results using the best model (v22)."""
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "inference")))
from predict_board import predict_board
from cfg import CLASS_TO_FEN

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

CLASS_NAMES = ['P','R','N','B','Q','K','p','r','n','b','q','k','.']
DISPLAY_NAMES = ['W.Pawn','W.Rook','W.Knight','W.Bishop','W.Queen','W.King',
                 'B.Pawn','B.Rook','B.Knight','B.Bishop','B.Queen','B.King','Empty']
NUM_CLASSES = 13

FEN_TO_ENC = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11,
}
EMPTY = 12
OOD = 13


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


def load_txt_pairs(game_dir):
    fen_path = os.path.join(game_dir, 'FEN.txt')
    img_dir = os.path.join(game_dir, 'images')
    with open(fen_path) as f:
        fens = [line.strip() for line in f if line.strip()]
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    pairs = []
    for i, img_name in enumerate(images):
        if i < len(fens):
            pairs.append((os.path.join(img_dir, img_name), fens[i]))
    return pairs


def load_csv_pairs(game_dir, game_name):
    import csv
    csv_path = os.path.join(game_dir, f'{game_name}.csv')
    img_dir = os.path.join(game_dir, 'images')
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


if __name__ == '__main__':
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    EVAL_DATA = os.path.join(ROOT, "data", "eval")

    all_pairs = []
    for game in ['game13']:
        all_pairs.extend(load_txt_pairs(os.path.join(EVAL_DATA, game)))
    for game in ['game5', 'game2']:
        all_pairs.extend(load_csv_pairs(os.path.join(EVAL_DATA, game), game))

    print(f"Total pairs: {len(all_pairs)}")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for idx, (img_path, gt_fen) in enumerate(all_pairs):
        print(f"  [{idx+1}/{len(all_pairs)}] {os.path.basename(img_path)}")
        img = np.array(Image.open(img_path).convert('RGB'))
        pred = predict_board(img)
        gt = fen_to_board(gt_fen)

        for r in range(8):
            for c in range(8):
                g = gt[r][c]
                p = pred[r][c].item()
                if g == -1:
                    continue
                if p == OOD:
                    p = EMPTY
                if 0 <= g < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                    cm[g][p] += 1

    print(f"\nConfusion matrix shape: {cm.shape}")
    print(f"Total classified: {cm.sum()}")

    fig, ax = plt.subplots(figsize=(12, 10))

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm_norm / row_sums * 100

    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_pct[i, j]
            count = cm[i, j]
            color = 'white' if val > 60 else 'black'
            if count > 0:
                ax.text(j, i, f'{val:.1f}%\n({count})', ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold' if i == j else 'normal')
            else:
                ax.text(j, i, '0', ha='center', va='center',
                        fontsize=7, color='#999')

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(DISPLAY_NAMES, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ground Truth', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix (All Evaluation Games)', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'confusion_matrix_final.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved {save_path}")
