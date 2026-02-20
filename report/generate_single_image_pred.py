"""Predict on a single image and save the FEN board visualization."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
from predict_board import predict_board
from cfg import CLASS_TO_FEN

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

PIECE_UNICODE = {
    0: '\u2659', 1: '\u2656', 2: '\u2658', 3: '\u2657', 4: '\u2655', 5: '\u2654',
    6: '\u265F', 7: '\u265C', 8: '\u265E', 9: '\u265D', 10: '\u265B', 11: '\u265A',
    12: '', 13: ''
}


def draw_board(ax, board_tensor):
    for r in range(8):
        for c in range(8):
            bg = '#F0D9B5' if (r + c) % 2 == 0 else '#B58863'
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=bg, edgecolor='#333', lw=0.5))
            val = board_tensor[r][c].item()
            if val == 13:
                pad = 0.18
                ax.plot([c + pad, c + 1 - pad], [r + pad, r + 1 - pad], color='red', lw=3, solid_capstyle='round')
                ax.plot([c + pad, c + 1 - pad], [r + 1 - pad, r + pad], color='red', lw=3, solid_capstyle='round')
            elif PIECE_UNICODE.get(val, ''):
                piece = PIECE_UNICODE[val]
                color = '#222' if 6 <= val <= 11 else '#FFF'
                stroke = '#FFF' if 6 <= val <= 11 else '#222'
                ax.text(c + 0.5, r + 0.5, piece, ha='center', va='center',
                        fontsize=28, color=color, fontweight='bold',
                        path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground=stroke)])
    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)
    ax.set_aspect('equal')
    ax.set_xticks([i + 0.5 for i in range(8)])
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], fontsize=9)
    ax.set_yticks([i + 0.5 for i in range(8)])
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'], fontsize=9)
    ax.tick_params(length=0)


if __name__ == '__main__':
    IMG_PATH = os.path.join(os.path.dirname(__file__), '..', 
        'data/dino/val/game2/game8/frame_013240_jpg.rf.43c285e640b973e228d04a8aef2fe86c.jpg')

    print("Running prediction...")
    img_arr = np.array(Image.open(IMG_PATH).convert('RGB'))
    board = predict_board(img_arr)

    print("Board:")
    for r in range(8):
        row = ''
        for c in range(8):
            v = board[r][c].item()
            if v == 13: row += 'X '
            elif v == 12: row += '. '
            else: row += CLASS_TO_FEN.get(v, '?') + ' '
        print(row)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.15, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_arr)
    ax1.set_title('Input Image (with hand occlusion)', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax_arrow = fig.add_subplot(gs[1])
    ax_arrow.axis('off')
    ax_arrow.text(0.5, 0.55, '\u2192', ha='center', va='center', fontsize=50, color='#333',
                  transform=ax_arrow.transAxes)
    ax_arrow.text(0.5, 0.4, 'DINO\n+ OOD', ha='center', va='center', fontsize=10, color='#555',
                  transform=ax_arrow.transAxes)

    ax2 = fig.add_subplot(gs[2])
    draw_board(ax2, board)
    ax2.set_title('Predicted Board State', fontsize=13, fontweight='bold')

    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, 'ood_example_hand.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")
