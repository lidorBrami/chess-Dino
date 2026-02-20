"""Generate report diagrams: system pipeline, OOD crop visualization, grid assignment."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_system_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.5)

    ax.add_patch(FancyBboxPatch((0.3, 2.2), 2.2, 1.6, **box_kw, edgecolor='#333', facecolor='#f0f0f0'))
    ax.text(1.4, 3.3, 'Input Image', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.4, 2.8, '(RGB Chessboard)', ha='center', va='center', fontsize=9, color='#555')

    ax.add_patch(FancyBboxPatch((4.0, 3.8), 3.0, 1.6, **box_kw, edgecolor='#1565C0', facecolor='#BBDEFB'))
    ax.text(5.5, 4.9, 'DINO Detector', ha='center', va='center', fontsize=11, fontweight='bold', color='#1565C0')
    ax.text(5.5, 4.4, 'Swin-Large Backbone', ha='center', va='center', fontsize=9, color='#1565C0')

    ax.add_patch(FancyBboxPatch((4.0, 0.6), 3.0, 1.6, **box_kw, edgecolor='#2E7D32', facecolor='#C8E6C9'))
    ax.text(5.5, 1.7, 'OOD Detector', ha='center', va='center', fontsize=11, fontweight='bold', color='#2E7D32')
    ax.text(5.5, 1.2, 'MobileNetV3-Small', ha='center', va='center', fontsize=9, color='#2E7D32')

    ax.add_patch(FancyBboxPatch((8.2, 1.8), 3.0, 2.4, **box_kw, edgecolor='#E65100', facecolor='#FFE0B2'))
    ax.text(9.7, 3.7, 'Postprocessing', ha='center', va='center', fontsize=11, fontweight='bold', color='#E65100')
    ax.text(9.7, 3.15, 'Confidence filter (>0.3)', ha='center', va='center', fontsize=8, color='#E65100')
    ax.text(9.7, 2.7, 'Grid assignment (8x8)', ha='center', va='center', fontsize=8, color='#E65100')
    ax.text(9.7, 2.25, 'OOD overlay', ha='center', va='center', fontsize=8, color='#E65100')

    ax.add_patch(FancyBboxPatch((12.0, 2.2), 1.7, 1.6, **box_kw, edgecolor='#333', facecolor='#f0f0f0'))
    ax.text(12.85, 3.3, '8x8 Board', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(12.85, 2.8, '(FEN)', ha='center', va='center', fontsize=9, color='#555')

    arrow_kw = dict(arrowstyle='->', color='#333', lw=2, mutation_scale=20)
    ax.annotate('', xy=(4.0, 4.6), xytext=(2.5, 3.4), arrowprops=arrow_kw)
    ax.annotate('', xy=(4.0, 1.4), xytext=(2.5, 2.6), arrowprops=arrow_kw)
    ax.annotate('', xy=(8.2, 3.6), xytext=(7.0, 4.6), arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2, mutation_scale=20))
    ax.text(7.8, 4.45, 'Boxes +\nClasses', ha='center', va='center', fontsize=7, color='#1565C0')
    ax.annotate('', xy=(8.2, 2.4), xytext=(7.0, 1.4), arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2, mutation_scale=20))
    ax.text(7.8, 1.5, 'OOD\nProbs', ha='center', va='center', fontsize=7, color='#2E7D32')
    ax.annotate('', xy=(12.0, 3.0), xytext=(11.2, 3.0), arrowprops=arrow_kw)

    fig.savefig(os.path.join(OUT_DIR, 'system_pipeline.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved system_pipeline.png")


def draw_ood_crop_visualization():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.invert_yaxis()

    ax = axes[0]
    ax.set_title('Board Grid (8x8)', fontsize=11, fontweight='bold')
    for r in range(8):
        for c in range(8):
            color = '#F0D9B5' if (r + c) % 2 == 0 else '#B58863'
            ax.add_patch(plt.Rectangle((c*12.5, r*12.5), 12.5, 12.5, facecolor=color, edgecolor='#333', lw=0.5))
    ax.add_patch(plt.Rectangle((3*12.5, 3*12.5), 12.5, 12.5, facecolor='none', edgecolor='red', lw=3))
    ax.text(3.5*12.5, 3.5*12.5, '?', ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.set_title('Standard Crop', fontsize=11, fontweight='bold')
    ax.add_patch(plt.Rectangle((15, 15), 70, 70, facecolor='#B58863', edgecolor='#333', lw=2))
    ax.text(50, 50, 'Square\nContent', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    ax.set_title('Expanded Crop (+10px)', fontsize=11, fontweight='bold')
    ax.add_patch(plt.Rectangle((5, 5), 90, 90, facecolor='#DDD', edgecolor='#2E7D32', lw=3, linestyle='--'))
    ax.add_patch(plt.Rectangle((15, 15), 70, 70, facecolor='#B58863', edgecolor='#333', lw=2))
    ax.text(50, 50, 'Square\nContent', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax.annotate('', xy=(5, 50), xytext=(15, 50), arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=1.5))
    ax.text(10, 42, '10px', ha='center', va='center', fontsize=8, color='#2E7D32', fontweight='bold')
    ax.text(50, 93, 'Extra context for OOD detection', ha='center', va='center', fontsize=8, color='#2E7D32', fontstyle='italic')
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ood_crop_visualization.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved ood_crop_visualization.png")


def draw_grid_assignment():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.set_title('DINO Detections (Bounding Boxes)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 800); ax.set_ylim(0, 800); ax.invert_yaxis(); ax.set_aspect('equal')
    for r in range(8):
        for c in range(8):
            color = '#F0D9B5' if (r+c)%2==0 else '#B58863'
            ax.add_patch(plt.Rectangle((c*100, r*100), 100, 100, facecolor=color, edgecolor='#AAA', lw=0.5))

    detections = [
        (85,15,185,95,'K','#1565C0',0.95), (310,10,390,90,'B','#1565C0',0.88),
        (605,15,695,95,'R','#1565C0',0.92), (210,110,290,190,'p','#D32F2F',0.91),
        (505,210,595,290,'N','#1565C0',0.87), (110,610,190,690,'P','#1565C0',0.93),
        (305,610,395,690,'P','#1565C0',0.96), (510,710,590,790,'k','#D32F2F',0.89),
    ]
    for x1,y1,x2,y2,label,clr,score in detections:
        ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, facecolor='none', edgecolor=clr, lw=2.5))
        cx, cy = (x1+x2)/2, (y1+y2)/2
        ax.plot(cx, cy, 'o', color=clr, markersize=6, zorder=5)
        ax.text(x1, y1-5, f'{label} ({score:.2f})', fontsize=8, color=clr, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.set_title('Grid Assignment (Center -> Cell)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 800); ax.set_ylim(850, -10); ax.set_aspect('equal')
    assignments = [(0,1,'K'),(0,3,'B'),(0,6,'R'),(1,2,'p'),(2,5,'N'),(6,1,'P'),(6,3,'P'),(7,5,'k')]
    for r in range(8):
        for c in range(8):
            color = '#F0D9B5' if (r+c)%2==0 else '#B58863'
            ax.add_patch(plt.Rectangle((c*100, r*100), 100, 100, facecolor=color, edgecolor='#AAA', lw=0.5))
    for r,c,piece in assignments:
        is_w = piece.isupper()
        bg = '#BBDEFB' if is_w else '#FFCDD2'
        ax.add_patch(plt.Rectangle((c*100+5, r*100+5), 90, 90, facecolor=bg, edgecolor='#333', lw=2, alpha=0.8))
        ax.text(c*100+50, r*100+50, piece, ha='center', va='center', fontsize=18, fontweight='bold',
                color='#1565C0' if is_w else '#D32F2F')
    for i in range(9):
        ax.axhline(y=i*100, color='#333', lw=1, alpha=0.3)
        ax.axvline(x=i*100, color='#333', lw=1, alpha=0.3)
    ax.text(400, 830, 'col = floor(cx / (W/8)),  row = floor(cy / (H/8))', ha='center', va='center',
            fontsize=10, fontstyle='italic', color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='#F9A825'))
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'grid_assignment.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved grid_assignment.png")


if __name__ == '__main__':
    draw_system_pipeline()
    draw_ood_crop_visualization()
    draw_grid_assignment()
    print(f"\nAll diagrams saved to {OUT_DIR}/")
