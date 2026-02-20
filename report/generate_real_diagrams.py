"""Generate diagrams using real chess board images:
1. Bounding boxes + labels from DINO on a real board
2. OOD crop visualization on a real board square
"""
import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "projects", "dino")))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES = {
    1: "black-bishop", 2: "black-king", 3: "black-knight", 4: "black-pawn",
    5: "black-queen", 6: "black-rook", 7: "white-bishop", 8: "white-king",
    9: "white-knight", 10: "white-pawn", 11: "white-queen", 12: "white-rook"
}

SHORT_NAMES = {
    1: "b", 2: "k", 3: "n", 4: "p", 5: "q", 6: "r",
    7: "B", 8: "K", 9: "N", 10: "P", 11: "Q", 12: "R"
}

COLORS_BLACK = '#E53935'
COLORS_WHITE = '#1E88E5'


def load_dino():
    from detectron2.config import LazyConfig, instantiate
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(root, "projects", "dino", "configs", "dino_swin_large_chess_finetune.py")
    ckpt_path = os.path.join(root, "output", "dino_chess_v22", "model_final.pth")

    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model, device


def run_dino(model, device, img_path, threshold=0.3):
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
    with torch.no_grad():
        outputs = model([{"image": tensor.to(device), "height": H, "width": W}])
    inst = outputs[0]["instances"]
    boxes = inst.pred_boxes.tensor.cpu().numpy()
    scores = inst.scores.cpu().numpy()
    classes = inst.pred_classes.cpu().numpy()
    mask = scores >= threshold
    return img, boxes[mask], scores[mask], classes[mask]


def draw_detections(img, boxes, scores, classes, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    for box, score, cls in zip(boxes, scores, classes):
        if cls not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = box
        is_black_piece = cls <= 6
        color = COLORS_BLACK if is_black_piece else COLORS_WHITE
        label = f"{SHORT_NAMES[cls]} {score:.2f}"

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot(cx, cy, 'o', color=color, markersize=5, zorder=5)
        ax.text(x1, y1 - 4, label, fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))

    ax.set_title('DINO Detections on Real Board', fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")


def draw_grid_assignment_real(img, boxes, scores, classes, save_path):
    H, W = img.shape[:2]
    sq_w, sq_h = W / 8.0, H / 8.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: detections with center dots
    ax = axes[0]
    ax.imshow(img)
    for i in range(1, 8):
        ax.axhline(y=i * sq_h, color='yellow', lw=1, alpha=0.5)
        ax.axvline(x=i * sq_w, color='yellow', lw=1, alpha=0.5)
    for box, score, cls in zip(boxes, scores, classes):
        if cls not in CLASS_NAMES:
            continue
        x1, y1, x2, y2 = box
        color = COLORS_BLACK if cls <= 6 else COLORS_WHITE
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot(cx, cy, 'o', color='yellow', markersize=8, zorder=5, markeredgecolor='black', markeredgewidth=1)
    ax.set_title('Detections + Grid + Center Points', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Right: 8x8 grid result
    ax = axes[1]
    board = [['.' for _ in range(8)] for _ in range(8)]
    board_color = [[None for _ in range(8)] for _ in range(8)]
    board_score = [[0.0 for _ in range(8)] for _ in range(8)]

    for box, score, cls in zip(boxes, scores, classes):
        if cls not in SHORT_NAMES:
            continue
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        col = int(cx / sq_w)
        row = int(cy / sq_h)
        col = max(0, min(col, 7))
        row = max(0, min(row, 7))
        if score > board_score[row][col]:
            board[row][col] = SHORT_NAMES[cls]
            board_color[row][col] = COLORS_BLACK if cls <= 6 else COLORS_WHITE
            board_score[row][col] = score

    for r in range(8):
        for c in range(8):
            bg = '#F0D9B5' if (r + c) % 2 == 0 else '#B58863'
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=bg, edgecolor='#333', lw=0.5))
            piece = board[r][c]
            if piece != '.':
                color = board_color[r][c]
                ax.add_patch(plt.Rectangle((c + 0.05, r + 0.05), 0.9, 0.9,
                             facecolor='#BBDEFB' if color == COLORS_WHITE else '#FFCDD2',
                             edgecolor='#333', lw=1.5, alpha=0.8))
                ax.text(c + 0.5, r + 0.5, piece, ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color)

    ax.set_xlim(0, 8); ax.set_ylim(8, 0)
    ax.set_aspect('equal')
    ax.set_title('Grid Assignment Result', fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")


def draw_ood_crop_real(img_path, save_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    sq_w, sq_h = W / 8.0, H / 8.0
    border = 10

    row, col = 3, 4

    x0 = int(col * sq_w)
    y0 = int(row * sq_h)
    x1 = int((col + 1) * sq_w)
    y1 = int((row + 1) * sq_h)
    standard_crop = img[y0:y1, x0:x1]

    ex0 = max(0, x0 - border)
    ey0 = max(0, y0 - border)
    ex1 = min(W, x1 + border)
    ey1 = min(H, y1 + border)
    expanded_crop = img[ey0:ey1, ex0:ex1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Full board with highlighted square
    ax = axes[0]
    ax.imshow(img)
    rect = patches.Rectangle((x0, y0), sq_w, sq_h, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    rect2 = patches.Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, linewidth=2,
                                edgecolor='#2E7D32', facecolor='none', linestyle='--')
    ax.add_patch(rect2)
    ax.set_title('Full Board (square highlighted)', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Standard crop
    ax = axes[1]
    ax.imshow(standard_crop)
    ax.set_title('Standard Crop', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Expanded crop
    ax = axes[2]
    ax.imshow(expanded_crop)
    ch, cw = expanded_crop.shape[:2]
    inner_x0 = x0 - ex0
    inner_y0 = y0 - ey0
    inner_w = x1 - x0
    inner_h = y1 - y0
    rect = patches.Rectangle((inner_x0, inner_y0), inner_w, inner_h,
                               linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    rect2 = patches.Rectangle((0, 0), cw - 1, ch - 1, linewidth=3,
                                edgecolor='#2E7D32', facecolor='none', linestyle='--')
    ax.add_patch(rect2)
    ax.set_title('Expanded Crop (+10px border)', fontsize=11, fontweight='bold')
    ax.axis('off')

    fig.suptitle('OOD Square Cropping Strategy', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")


if __name__ == '__main__':
    IMG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "game13", "images", "frame_000000.jpg")

    print("Loading DINO model...")
    model, device = load_dino()

    print("Running inference...")
    img, boxes, scores, classes = run_dino(model, device, IMG_PATH)

    print("Drawing detection diagram...")
    draw_detections(img, boxes, scores, classes, os.path.join(OUT_DIR, 'real_detections.png'))

    print("Drawing grid assignment diagram...")
    draw_grid_assignment_real(img, boxes, scores, classes, os.path.join(OUT_DIR, 'real_grid_assignment.png'))

    print("Drawing OOD crop diagram...")
    draw_ood_crop_real(IMG_PATH, os.path.join(OUT_DIR, 'real_ood_crop.png'))

    print(f"\nAll real diagrams saved to {OUT_DIR}/")
