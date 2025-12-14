# src/04-inference.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import torch

from config import CFG
from utils import get_logger
from models import SmallCNN   # ✅ HELYES IMPORT

LOG = get_logger()

def main():
    LOG.info("=== Inference ===")

    data = np.load(CFG.processed_dir / "dataset.npz")
    X, y = data["X"], data["y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(in_ch=X.shape[-1], n_classes=len(CFG.classes)).to(device)
    model.load_state_dict(
        torch.load(CFG.model_dir / "model.pt", map_location=device)
    )
    model.eval()

    idx = np.random.choice(len(y), size=min(10, len(y)), replace=False)
    Xin = torch.tensor(X[idx], dtype=torch.float32).transpose(1, 2).to(device)

    with torch.no_grad():
        logits = model(Xin)
        pred = logits.argmax(1).cpu().numpy()

    for i, j in enumerate(idx):
        LOG.info(
            f"sample={int(j)} "
            f"true={CFG.classes[int(y[j])]} "
            f"pred={CFG.classes[int(pred[i])]}"
        )

    LOG.info("=== Inference DONE ===")

if __name__ == "__main__":
    main()
