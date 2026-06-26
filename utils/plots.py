"""Training metric logging and plotting.

Everything written here lands in  <save_model_dir>/plots/ :
  - metrics.csv            one row per epoch (the raw history, source of truth)
  - loss_curve.png         train vs val loss
  - lr_curve.png           learning-rate schedule
  - gen_gap.png            val-minus-train loss (overfitting check)
  - ap50_curve.png         detection accuracy AP@50 (from epoch 30 on)
  - precision_recall.png   precision & recall @ IoU 0.5
  - training_overview.png  2x2 dashboard of the above

Plots are regenerated from metrics.csv every epoch, so they always show the full
history including epochs from earlier (resumed) runs.
"""
import os
import csv

import matplotlib
matplotlib.use('Agg')          # headless backend (no display needed on WSL/servers)
import matplotlib.pyplot as plt

METRIC_FIELDS = ['epoch', 'lr', 'train_loss', 'val_loss', 'AP_50', 'REC_50', 'PRE_50']


def get_plots_dir(save_model_dir):
    """Return (and create) the plots/ folder inside the run's log directory."""
    d = os.path.join(save_model_dir, 'plots')
    os.makedirs(d, exist_ok=True)
    return d


def log_metrics_csv(plots_dir, row):
    """Append one epoch's metrics to plots/metrics.csv (writes header if new)."""
    csv_path = os.path.join(plots_dir, 'metrics.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in METRIC_FIELDS})
    return csv_path


def _read_metrics(csv_path):
    cols = {k: [] for k in METRIC_FIELDS}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in METRIC_FIELDS:
                v = r.get(k, '')
                cols[k].append(float(v) if v not in ('', None) else 0.0)
    cols['epoch'] = [int(e) for e in cols['epoch']]
    return cols


def _save(fig_or_none, plots_dir, name):
    plt.savefig(os.path.join(plots_dir, name), dpi=120, bbox_inches='tight')
    plt.close()


def plot_all(plots_dir):
    """Regenerate every plot from plots/metrics.csv. Safe to call each epoch."""
    csv_path = os.path.join(plots_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        return
    m = _read_metrics(csv_path)
    epochs = m['epoch']
    if not epochs:
        return

    # Detection metrics (AP/precision/recall) are only measured from epoch 30 on.
    det = [(e, a, rc, pr) for e, a, rc, pr
           in zip(epochs, m['AP_50'], m['REC_50'], m['PRE_50'])
           if (a > 0 or rc > 0 or pr > 0)]
    de   = [d[0] for d in det]
    da   = [d[1] for d in det]
    drec = [d[2] for d in det]
    dpre = [d[3] for d in det]

    # 1) Loss: train vs val
    plt.figure()
    plt.plot(epochs, m['train_loss'], 'b-', label='train')
    plt.plot(epochs, m['val_loss'], 'r-', label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss')
    plt.legend(); plt.grid(True, alpha=.3)
    _save(None, plots_dir, 'loss_curve.png')

    # 2) Learning-rate schedule
    plt.figure()
    plt.plot(epochs, m['lr'], 'm-')
    plt.xlabel('Epoch'); plt.ylabel('Learning rate'); plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=.3)
    _save(None, plots_dir, 'lr_curve.png')

    # 3) Generalization gap (val - train): rising gap => overfitting
    plt.figure()
    gap = [v - t for v, t in zip(m['val_loss'], m['train_loss'])]
    plt.plot(epochs, gap, 'k-')
    plt.axhline(0, color='gray', lw=.8)
    plt.xlabel('Epoch'); plt.ylabel('Val - Train loss')
    plt.title('Generalization Gap (overfitting check)')
    plt.grid(True, alpha=.3)
    _save(None, plots_dir, 'gen_gap.png')

    if det:
        # 4) AP@50
        plt.figure()
        plt.plot(de, da, 'g-o', ms=3)
        plt.xlabel('Epoch'); plt.ylabel('AP@50'); plt.title('Detection Accuracy (AP@50)')
        plt.grid(True, alpha=.3)
        _save(None, plots_dir, 'ap50_curve.png')

        # 5) Precision & Recall
        plt.figure()
        plt.plot(de, dpre, 'c-o', ms=3, label='Precision@50')
        plt.plot(de, drec, 'y-o', ms=3, label='Recall@50')
        plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Precision & Recall @ IoU 0.5')
        plt.legend(); plt.grid(True, alpha=.3)
        _save(None, plots_dir, 'precision_recall.png')

    # 6) Overview dashboard (2x2)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(epochs, m['train_loss'], 'b-', label='train')
    ax[0, 0].plot(epochs, m['val_loss'], 'r-', label='val')
    ax[0, 0].set_title('Loss'); ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].legend(); ax[0, 0].grid(True, alpha=.3)

    ax[0, 1].plot(epochs, m['lr'], 'm-')
    ax[0, 1].set_title('Learning Rate'); ax[0, 1].set_xlabel('Epoch'); ax[0, 1].grid(True, alpha=.3)

    if det:
        ax[1, 0].plot(de, da, 'g-o', ms=3)
        ax[1, 0].set_title('AP@50'); ax[1, 0].set_xlabel('Epoch'); ax[1, 0].grid(True, alpha=.3)
        ax[1, 1].plot(de, dpre, 'c-o', ms=3, label='Precision')
        ax[1, 1].plot(de, drec, 'y-o', ms=3, label='Recall')
        ax[1, 1].set_title('Precision / Recall'); ax[1, 1].set_xlabel('Epoch')
        ax[1, 1].legend(); ax[1, 1].grid(True, alpha=.3)
    else:
        for a in (ax[1, 0], ax[1, 1]):
            a.text(.5, .5, 'AP / Precision / Recall\nstart at epoch 30',
                   ha='center', va='center')
            a.set_axis_off()

    fig.suptitle('Training Overview')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'training_overview.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
