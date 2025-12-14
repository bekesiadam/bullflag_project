import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import CFG
from utils import get_logger
from models import SmallCNN
from data_preprocessing import (
    load_csv,
    build_windows,
)

LOG = get_logger()


def merge_positive_windows(pred_labels, stride, window):
    """
    Egymást átfedő / közeli pozitív ablakokat egy zászlóvá von össze.
    Visszaad: list of (start_idx, end_idx, label)
    """
    segments = []
    current = None

    for i, lab in enumerate(pred_labels):
        if lab == 0:  # None
            if current is not None:
                segments.append(current)
                current = None
            continue

        if current is None:
            current = [i, i, lab]
        else:
            # ha ugyanaz a címke és közel van
            if lab == current[2] and i <= current[1] + 1:
                current[1] = i
            else:
                segments.append(current)
                current = [i, i, lab]

    if current is not None:
        segments.append(current)

    return segments


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/05-scan-timeseries.py path/to/file.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    LOG.info("=== Scan timeseries ===")
    LOG.info(f"CSV: {csv_path}")

    # betöltés
    df = load_csv(csv_path)
    if len(df) < CFG.window:
        raise ValueError("Time series shorter than window size")

    # nincs label -> üres intervallum lista
    X, _ = build_windows(df, [], CFG.window, CFG.stride)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallCNN(in_ch=X.shape[-1], n_classes=len(CFG.classes)).to(device)
    model.load_state_dict(torch.load(CFG.model_dir / "model.pt", map_location=device))
    model.eval()

    Xt = torch.tensor(X, dtype=torch.float32).transpose(1, 2).to(device)

    with torch.no_grad():
        logits = model(Xt)
        preds = logits.argmax(1).cpu().numpy()

    # statisztika
    unique, counts = np.unique(preds, return_counts=True)
    dist = {CFG.classes[int(k)]: int(v) for k, v in zip(unique, counts)}

    LOG.info(f"Scanned windows: {len(preds)}")
    LOG.info(f"Window-level distribution: {dist}")

    # összevonás
    segments = merge_positive_windows(preds, CFG.stride, CFG.window)

    LOG.info("Detected flag segments:")
    if not segments:
        LOG.info("  none")
    else:
        for s, e, lab in segments:
            label_name = CFG.classes[int(lab)]
            LOG.info(f"  {label_name}: windows {s}–{e}")

    LOG.info("=== Scan DONE ===")


if __name__ == "__main__":
    main()
