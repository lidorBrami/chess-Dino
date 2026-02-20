#!/usr/bin/env python3
"""
Generate report figures for chess board recognition project.
- Confusion matrix: computed LIVE from DINO v18 on validation set
- Training curves: parsed from metrics.json
- Per-game accuracy: from eval results
"""
import sys
import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/lidorbr/detrex")
sys.path.insert(0, "/home/lidorbr/detrex/projects/dino")

METRICS_V18_PATH = "/home/lidorbr/detrex/output/dino_chess_v18/metrics.json"
METRICS_V19_PATH = "/home/lidorbr/detrex/output/dino_chess_v19/metrics.json"
V18_CLS_ACC_PATH = os.path.join(os.path.dirname(__file__), "v18_cls_accuracy.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "report_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_metrics(metrics_path):
    """Parse metrics.json to extract training loss, eval AP, and cls accuracy."""
    iterations = []
    loss_class = []
    total_loss = []

    eval_iters = []
    eval_ap = []
    eval_per_class = {}

    cls_acc_iters = []
    cls_acc_train = []
    cls_acc_val = []

    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line.strip())
            it = d.get("iteration", 0)

            if "loss_class" in d:
                iterations.append(it)
                loss_class.append(d["loss_class"])
                total_loss.append(d["total_loss"])

            if "bbox/AP" in d:
                eval_iters.append(it)
                eval_ap.append(d["bbox/AP"])
                for key, val in d.items():
                    if key.startswith("bbox/AP-") and key != "bbox/AP-1":
                        cls = key.replace("bbox/AP-", "")
                        if cls not in eval_per_class:
                            eval_per_class[cls] = {"iters": [], "ap": []}
                        if not np.isnan(val):
                            eval_per_class[cls]["iters"].append(it)
                            eval_per_class[cls]["ap"].append(val)

            if "cls_acc/train" in d:
                cls_acc_iters.append(it)
                cls_acc_train.append(d["cls_acc/train"])
                cls_acc_val.append(d["cls_acc/val"])

    return {
        "iterations": iterations, "loss_class": loss_class, "total_loss": total_loss,
        "eval_iters": eval_iters, "eval_ap": eval_ap, "eval_per_class": eval_per_class,
        "cls_acc_iters": cls_acc_iters, "cls_acc_train": cls_acc_train, "cls_acc_val": cls_acc_val,
    }


# ============================================================
# FIGURE 1: CONFUSION MATRIX (LIVE)
# ============================================================
def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('DINO v18 Piece Classification Confusion Matrix (%)', fontsize=14)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            pct = cm_norm[i, j]
            if val > 0:
                color = 'white' if pct > 50 else 'black'
                ax.text(j, i, f'{val}\n({pct:.0f}%)', ha='center', va='center',
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Percentage (%)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# FIGURE 2: PER-CLASS RECALL (v17 vs v18)
# ============================================================
def plot_recall_bar(cm, class_names, save_path):
    recall = []
    precision = []
    for i in range(len(class_names)):
        row_sum = cm[i].sum()
        col_sum = cm[:, i].sum()
        recall.append(cm[i, i] / row_sum * 100 if row_sum > 0 else 0)
        precision.append(cm[i, i] / col_sum * 100 if col_sum > 0 else 0)

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, recall, width, label='Recall', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, precision, width, label='Precision', color='#4ECDC4', alpha=0.8)

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('DINO v18: Per-Class Recall & Precision', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(55, 105)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# FIGURE 3: CLASSIFICATION ACCURACY CURVES (v19 + v18)
# ============================================================
def plot_cls_accuracy_curves(metrics_v19, v18_cls_data, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: learning curve (from early checkpoint)
    train_acc = [v / 100.0 for v in metrics_v19["cls_acc_train"]]
    val_acc = [v / 100.0 for v in metrics_v19["cls_acc_val"]]
    ax1.plot(metrics_v19["cls_acc_iters"], train_acc,
             'o-', color='#3498db', linewidth=2, markersize=4, label='Train')
    ax1.plot(metrics_v19["cls_acc_iters"], val_acc,
             'o-', color='#e74c3c', linewidth=2, markersize=4, label='Validation')
    ax1.fill_between(metrics_v19["cls_acc_iters"], train_acc, val_acc, alpha=0.1, color='gray')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('Learning Curve (from early checkpoint)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.grid(alpha=0.3)

    # Right: converged model
    v18_iters = [r["iteration"] for r in v18_cls_data]
    v18_train = [r["train_cls_acc"] / 100.0 for r in v18_cls_data]
    v18_val = [r["val_cls_acc"] / 100.0 for r in v18_cls_data]

    ax2.plot(v18_iters, v18_train, 'o-', color='#3498db', linewidth=2, markersize=4, label='Train')
    ax2.plot(v18_iters, v18_val, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Validation')
    ax2.fill_between(v18_iters, v18_train, v18_val, alpha=0.1, color='gray')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Converged Model (continued training)', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_loss(metrics_v19, save_path):
    """Plot training classification loss curve from v19."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(metrics_v19["iterations"], metrics_v19["loss_class"], color='#e74c3c', alpha=0.3, linewidth=0.5)
    window = 5
    if len(metrics_v19["loss_class"]) > window:
        smooth = np.convolve(metrics_v19["loss_class"], np.ones(window)/window, mode='valid')
        ax.plot(metrics_v19["iterations"][window-1:], smooth, color='#e74c3c', linewidth=2, label='Classification Loss (smoothed)')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training: Classification Loss Over Iterations', fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# FIGURE 4: VALIDATION AP PROGRESSION (overall + weak classes)
# ============================================================
def plot_ap_progression(metrics, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(metrics["eval_iters"], metrics["eval_ap"], 'o-', color='#3498db', linewidth=2, markersize=5)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Average Precision (AP)', fontsize=12)
    ax1.set_title('Validation: Overall AP Over Training', fontsize=13)
    ax1.grid(alpha=0.3)
    for i, (it, ap) in enumerate(zip(metrics["eval_iters"], metrics["eval_ap"])):
        if i == 0 or i == len(metrics["eval_iters"])-1:
            ax1.annotate(f'{ap:.1f}%', (it, ap), xytext=(5, 10), textcoords='offset points', fontsize=9, fontweight='bold')

    weak_classes = ['black-rook', 'white-bishop', 'black-bishop', 'white-queen']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    for cls, color in zip(weak_classes, colors):
        if cls in metrics["eval_per_class"]:
            data = metrics["eval_per_class"][cls]
            ax2.plot(data["iters"], data["ap"], 'o-', color=color, linewidth=1.5, markersize=4, label=cls, alpha=0.8)

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('AP', fontsize=12)
    ax2.set_title('Validation: AP for Weak Classes Over Training', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_game(save_path):
    games = ['Game 13', 'Game 5', 'Game 2']
    accuracies = [97.2, 98.6, 97.2]
    ood_rates = [85.7, 87.5, 76.0]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(games, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Per-Square Accuracy by Game (incl. OOD)', fontsize=13)
    ax1.set_ylim(94, 100)
    ax1.axhline(y=97.5, color='red', linestyle='--', alpha=0.7)
    ax1.text(4.3, 97.6, 'Mean: 97.5%', color='red', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    bars2 = ax2.bar(games, ood_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('OOD Detection Rate (%)', fontsize=12)
    ax2.set_title('OOD (Hand) Detection Accuracy by Game', fontsize=13)
    ax2.set_ylim(50, 100)
    ax2.grid(axis='y', alpha=0.3)
    for bar, rate in zip(bars2, ood_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# FIGURE 6: PRECISION vs RECALL SCATTER
# ============================================================
def plot_precision_recall(cm, class_names, save_path):
    n = len(class_names)
    recall = []
    precision = []
    for i in range(n):
        row_sum = cm[i].sum()
        col_sum = cm[:, i].sum()
        recall.append(cm[i, i] / row_sum * 100 if row_sum > 0 else 0)
        precision.append(cm[i, i] / col_sum * 100 if col_sum > 0 else 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors_scatter = ['#e74c3c' if r < 85 or p < 85 else '#2ecc71' for r, p in zip(recall, precision)]

    for i, (r, p, name) in enumerate(zip(recall, precision, class_names)):
        ax.scatter(r, p, c=colors_scatter[i], s=120, zorder=5, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (r, p), xytext=(2, 2), textcoords='offset points', fontsize=9, fontweight='bold')

    ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=85, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between([0, 85], 0, 85, alpha=0.05, color='red')
    ax.set_xlabel('Recall (%)', fontsize=12)
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Precision vs Recall per Class (from Confusion Matrix)', fontsize=14)
    ax.set_xlim(55, 105)
    ax.set_ylim(55, 105)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# FIGURE 7: WEIGHT EVOLUTION TABLE
# ============================================================
def plot_weight_table(save_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    table_data = [
        ['Class', 'Initial', 'v15-v16', 'v17', 'v18 (Final)', 'Rationale'],
        ['black-rook', '1.0', '8.0', '12.0', '18.0', 'Worst class (64%), boost hard'],
        ['white-bishop', '1.0', '12.0', '6.0', '10.0', 'Confused with queen (76%)'],
        ['black-bishop', '1.0', '8.5', '5.0', '3.0', 'Was over-predicted, moderate'],
        ['black-queen', '1.0', '1.0', '1.0', '0.5', 'Reduce: over-predicting'],
        ['white-queen', '1.0', '1.0', '1.0', '0.5', 'Reduce: over-predicting'],
        ['loss_class', '1.0', '1.0', '3.5', '5.0', 'Classification > bbox'],
        ['loss_bbox', '5.0', '5.0', '2.0', '2.0', 'Reduced priority'],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for i in range(len(table_data[0])):
        table[0, i].set_facecolor('#34495e')
        table[0, i].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[i, j].set_facecolor('#ecf0f1')
    ax.set_title('Hyperparameter Evolution Across Training Versions', fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# FIGURE 8: SYSTEM ARCHITECTURE
# ============================================================
def plot_architecture(save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    boxes = [
        (0.05, 0.5, 'Input\nImage', '#3498db'),
        (0.22, 0.7, 'DINO\n(Swin-Large)\nPiece Detection', '#2ecc71'),
        (0.22, 0.25, 'MobileNet\nOOD Detector\n(per square)', '#e74c3c'),
        (0.45, 0.7, '8x8 Grid\nMapping\n(bbox center)', '#f39c12'),
        (0.45, 0.25, '8x8 OOD\nMask', '#e74c3c'),
        (0.65, 0.5, 'Combine:\nOOD overrides\nempty only', '#9b59b6'),
        (0.85, 0.5, 'Output\n8x8 Tensor\n(0-13)', '#34495e'),
    ]
    for x, y, text, color in boxes:
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, color='white', fontweight='bold', bbox=bbox)
    arrows = [(0.12,0.55,0.14,0.7),(0.12,0.45,0.14,0.25),(0.32,0.7,0.37,0.7),(0.32,0.25,0.37,0.25),
              (0.53,0.65,0.57,0.55),(0.53,0.3,0.57,0.45),(0.73,0.5,0.77,0.5)]
    for x1,y1,x2,y2 in arrows:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.set_title('System Architecture: End-to-End Chess Board Recognition', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating report figures")
    print("=" * 60)

    # --- Parse training metrics (CPU only) ---
    print("\n[1/8] Parsing metrics.json...")
    metrics_v18 = parse_metrics(METRICS_V18_PATH)
    metrics_v19 = parse_metrics(METRICS_V19_PATH)
    print(f"  V18: {len(metrics_v18['iterations'])} steps, {len(metrics_v18['eval_iters'])} evals")
    print(f"  V19: {len(metrics_v19['iterations'])} steps, {len(metrics_v19['cls_acc_iters'])} cls_acc points")

    # Load v18 cls accuracy (computed from checkpoints)
    with open(V18_CLS_ACC_PATH) as f:
        v18_cls_data = json.load(f)

    # --- Run confusion matrix LIVE (needs GPU) ---
    print("\n[2/8] Computing confusion matrix LIVE on validation set...")
    from confusion_matrix import load_model, load_annotations, build_confusion_matrix, CLASS_NAMES

    model = load_model()
    image_annotations, image_info, cat_id_to_name, cat_id_to_idx = load_annotations()
    cm_full = build_confusion_matrix(model, image_annotations, image_info)

    # Extract the 12x12 piece-only confusion matrix (exclude MISS row/col)
    cm = cm_full[:12, :12]

    # --- Generate all figures ---
    print("\n[3/8] Plotting confusion matrix...")
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    print("[4/8] Plotting per-class recall & precision...")
    plot_recall_bar(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, "per_class_recall_precision.png"))

    print("[5/9] Plotting classification accuracy curves...")
    plot_cls_accuracy_curves(metrics_v19, v18_cls_data, os.path.join(OUTPUT_DIR, "cls_accuracy_curves.png"))

    print("[6/9] Plotting training loss curve...")
    plot_training_loss(metrics_v19, os.path.join(OUTPUT_DIR, "training_loss.png"))

    print("[6/8] Plotting validation AP progression...")
    plot_ap_progression(metrics_v18, os.path.join(OUTPUT_DIR, "validation_ap_progression.png"))

    print("[7/8] Plotting per-game accuracy...")
    plot_per_game(os.path.join(OUTPUT_DIR, "per_game_accuracy.png"))

    print("[8/8] Plotting precision vs recall scatter...")
    plot_precision_recall(cm, CLASS_NAMES, os.path.join(OUTPUT_DIR, "precision_recall_scatter.png"))

    # Bonus figures
    print("\n[+] Weight evolution table...")
    plot_weight_table(os.path.join(OUTPUT_DIR, "weight_evolution_table.png"))

    print("[+] System architecture diagram...")
    plot_architecture(os.path.join(OUTPUT_DIR, "system_architecture.png"))

    # Print summary from live confusion matrix
    print("\n" + "=" * 60)
    print("LIVE CONFUSION MATRIX SUMMARY (v18)")
    print("=" * 60)
    total = cm.sum()
    correct = np.trace(cm)
    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"\nPer-class recall:")
    for i, name in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        if row_sum > 0:
            print(f"  {name:15s}: {cm[i,i]:4d}/{row_sum:<4d} = {cm[i,i]/row_sum*100:.1f}%")

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
