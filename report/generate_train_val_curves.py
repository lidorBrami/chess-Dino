import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "dino_chess_v22", "metrics.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def parse_metrics(path):
    iterations, loss_class, total_loss = [], [], []
    cls_acc_iters, cls_acc_train, cls_acc_val = [], [], []

    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            it = d.get("iteration", 0)
            if "loss_class" in d:
                iterations.append(it)
                loss_class.append(d["loss_class"])
                total_loss.append(d["total_loss"])
            if "cls_acc/train" in d:
                cls_acc_iters.append(it)
                cls_acc_train.append(d["cls_acc/train"] / 100.0)
                cls_acc_val.append(d["cls_acc/val"] / 100.0)

    return {
        "iterations": iterations, "loss_class": loss_class, "total_loss": total_loss,
        "cls_acc_iters": cls_acc_iters, "cls_acc_train": cls_acc_train, "cls_acc_val": cls_acc_val,
    }


if __name__ == '__main__':
    m = parse_metrics(METRICS_PATH)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Classification Loss
    step = max(1, len(m["iterations"]) // 80)
    iters_s = m["iterations"][::step]
    loss_s = m["loss_class"][::step]

    ax1.plot(iters_s, loss_s, '-', color='#3498db', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Classification Loss', fontsize=12)
    ax1.set_title('Classification Loss: Train vs Validation', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Right: Classification Accuracy (from BestCheckpointHook data in v23)
    # Use v23 since it has the hook data
    v23_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "dino_chess_v23", "metrics.json")
    if os.path.exists(v23_path):
        m23 = parse_metrics(v23_path)
        if m23["cls_acc_iters"]:
            ax2.plot(m23["cls_acc_iters"], m23["cls_acc_train"],
                     'o-', color='#3498db', linewidth=2, markersize=5, label='Train')
            ax2.plot(m23["cls_acc_iters"], m23["cls_acc_val"],
                     'o-', color='#e74c3c', linewidth=2, markersize=5, label='Validation')
            ax2.fill_between(m23["cls_acc_iters"], m23["cls_acc_train"], m23["cls_acc_val"],
                             alpha=0.1, color='gray')

    # If no cls_acc data from metrics, parse from training log
    if not m23.get("cls_acc_iters"):
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "train-14908508.out")
        iters, accs = [], []
        with open(log_path) as f:
            for line in f:
                if "[BestCheckpoint]" in line:
                    parts = line.split("=")
                    for p in parts:
                        if "%" in p:
                            pct = float(p.split("%")[0].strip().split()[-1])
                            accs.append(pct / 100.0)
                            break
                    if "iter " in line:
                        it = int(line.split("iter ")[1].split(")")[0])
                        iters.append(it)
                    elif "best:" in line:
                        idx = len(accs)
                        iters.append(idx * 200)

        if not iters:
            iters = list(range(200, 200 * len(accs) + 1, 200))

        if len(iters) == len(accs):
            ax2.plot(iters, accs, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Validation')

    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Classification Accuracy', fontsize=12)
    ax2.set_title('Classification Accuracy: Train vs Validation', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0.85, 1.0)
    ax2.set_yticks([0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, 'train_val_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {save_path}")
