import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import CFG
from utils import get_logger, ensure_dir

LOG = get_logger()

class TSWindowDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, W, C)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].transpose(0,1)  # -> (C, W) 1D conv-hoz
        return x, self.y[i]

class SmallCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.head(z)

def count_params(m):
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    return total, trainable

def main():
    LOG.info("=== Training ===")
    LOG.info(f"epochs={CFG.epochs}, batch={CFG.batch_size}, lr={CFG.lr}, window={CFG.window}")

    data = np.load(CFG.processed_dir / "dataset.npz")
    X, y = data["X"], data["y"]
    LOG.info(f"Loaded dataset: X={X.shape}, y={y.shape}")

    # stratified 80/20
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=CFG.seed, stratify=y
    )

    tr_dl = DataLoader(TSWindowDS(Xtr, ytr), batch_size=CFG.batch_size, shuffle=True)
    va_dl = DataLoader(TSWindowDS(Xva, yva), batch_size=CFG.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info(f"Device: {device}")

    model = SmallCNN(in_ch=X.shape[-1], n_classes=len(CFG.classes)).to(device)
    total, trainable = count_params(model)
    LOG.info(f"Model: SmallCNN, total_params={total}, trainable_params={trainable}")

    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, CFG.epochs + 1):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item()) * len(yb)
            tr_ok += int((logits.argmax(1) == yb).sum().item())
            tr_n += len(yb)

        model.eval()
        va_loss, va_ok, va_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += float(loss.item()) * len(yb)
                va_ok += int((logits.argmax(1) == yb).sum().item())
                va_n += len(yb)

        LOG.info(
            f"Epoch {ep}/{CFG.epochs} | "
            f"train_loss={tr_loss/tr_n:.4f} train_acc={tr_ok/tr_n:.4f} | "
            f"val_loss={va_loss/va_n:.4f} val_acc={va_ok/va_n:.4f}"
        )

    ensure_dir(CFG.model_dir)
    torch.save(model.state_dict(), CFG.model_dir / "model.pt")
    LOG.info("Saved model to /app/models/model.pt")
    LOG.info("=== Training DONE ===")

if __name__ == "__main__":
    main()
