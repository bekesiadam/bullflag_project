# src/03-evaluation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import CFG
from utils import get_logger
from models import SmallCNN   # ✅ HELYES IMPORT

LOG = get_logger()

def main():
    LOG.info("=== Evaluation ===")

    data = np.load(CFG.processed_dir / "dataset.npz")
    X, y = data["X"], data["y"]

    # ugyanaz a split logika mint trainingben (signature-hoz elég)
    _, Xte, _, yte = train_test_split(
        X, y,
        test_size=0.2,
        random_state=CFG.seed,
        stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(
        in_ch=X.shape[-1],
        n_classes=len(CFG.classes)
    ).to(device)

    model.load_state_dict(
        torch.load(CFG.model_dir / "model.pt", map_location=device)
    )
    model.eval()

    Xt = torch.tensor(Xte, dtype=torch.float32).transpose(1, 2).to(device)
    with torch.no_grad():
        pred = model(Xt).argmax(1).cpu().numpy()

    LOG.info("Classification report:")
    LOG.info(
        "\n" + classification_report(
            yte,
            pred,
            target_names=CFG.classes,
            digits=4
        )
    )

    cm = confusion_matrix(yte, pred)
    LOG.info(f"Confusion matrix:\n{cm}")

    LOG.info("=== Evaluation DONE ===")

if __name__ == "__main__":
    main()
