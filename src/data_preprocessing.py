import json
from pathlib import Path
import numpy as np
import pandas as pd

from config import CFG
from utils import get_logger, ensure_dir

LOG = get_logger()

def find_csv_path(user_dir: Path, target_name: str):
    if not target_name:
        return None

    # 1. Direct match
    p = user_dir / target_name
    if p.exists():
        return p

    # 2. UUID prefix levágása
    clean = target_name
    if "-" in target_name and len(target_name) > 36:
        clean = target_name.split("-", 1)[1]

    # 3. Rekurzív keresés
    for fp in user_dir.rglob("*.csv"):
        if fp.name == target_name:
            return fp
        if fp.name.endswith(clean) or clean.endswith(fp.name):
            return fp

    return None

def parse_timestamp_col(s: pd.Series) -> pd.Series:
    # lehet epoch ms int, vagy "YYYY-mm-dd HH:MM"
    # 1) próbáljuk numerikusan
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().mean() > 0.8:
        # valószínű epoch ms
        return pd.to_datetime(s2.astype("int64"), unit="ms", utc=True).dt.tz_convert(None)
    # 2) string datetime
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = ["timestamp","open","high","low","close"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {csv_path}")
    df = df[req].copy()
    df["timestamp"] = parse_timestamp_col(df["timestamp"])
    df = df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp")
    return df.reset_index(drop=True)

def nearest_index(ts: pd.Series, t: pd.Timestamp) -> int:
    # ts: sorted datetime series
    # find nearest by absolute difference
    i = ts.searchsorted(t)
    if i <= 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    before = ts.iloc[i-1]
    after = ts.iloc[i]
    return i-1 if abs(before - t) <= abs(after - t) else i

def parse_labelstudio_json(label_path: Path):
    try:
        txt = label_path.read_text(encoding="utf-8").strip()
        if not txt:
            LOG.warning(f"[EMPTY JSON] {label_path}")
            return []

        data = json.loads(txt)
        if isinstance(data, dict):
            data = [data]
        return data

    except Exception as e:
        LOG.warning(f"[BAD JSON] {label_path}: {e}")
        return []


def collect_tasks(data_root: Path):
    tasks = []

    stats = {
        "users": 0,
        "label_files": 0,
        "csv_found": 0,
        "csv_missing": 0,
        "intervals": 0,
    }

    for user_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        stats["users"] += 1

        # 🔑 minden JSON-t nézünk
        json_files = [
            p for p in user_dir.glob("*.json")
            if "consensus" not in p.name.lower()
            and "sample" not in p.name.lower()
        ]

        if not json_files:
            continue

        for label_json in json_files:
            stats["label_files"] += 1
            export = parse_labelstudio_json(label_json)

            for task in export:
                # CSV név kinyerése
                file_name = (
                    task.get("file_upload")
                    or Path(task.get("data", {}).get("csv", "")).name
                )

                csv_path = find_csv_path(user_dir, file_name)

                if csv_path is None:
                    stats["csv_missing"] += 1
                    LOG.warning(f"[MISS] {user_dir.name}: {file_name}")
                    continue

                stats["csv_found"] += 1

                intervals = []
                anns = task.get("annotations", [])
                if anns:
                    for r in anns[0].get("result", []):
                        if r.get("type") != "timeserieslabels":
                            continue

                        v = r.get("value", {})
                        labs = v.get("timeserieslabels", [])
                        if not labs:
                            continue

                        label_full = labs[0]
                        label_type = label_full.split()[-1]

                        if label_type not in CFG.label_map:
                            continue

                        start = pd.to_datetime(v.get("start"), errors="coerce")
                        end = pd.to_datetime(v.get("end"), errors="coerce")
                        if pd.isna(start) or pd.isna(end):
                            continue

                        intervals.append((start, end, label_type))
                        stats["intervals"] += 1

                tasks.append((csv_path, intervals))

    LOG.info("=== DATA COLLECTION STATS ===")
    for k, v in stats.items():
        LOG.info(f"{k}: {v}")

    return tasks


def build_windows(df: pd.DataFrame, intervals, window: int, stride: int):
    ts = df["timestamp"]
    X = df[["open","high","low","close"]].to_numpy(dtype=np.float32)

    # normalizálás: log-return jelleg (stabil vegyes timeframe-nél)
    # close based returns + OHLC deltas
    close = X[:, 3]
    eps = 1e-8
    ret = np.log((close[1:] + eps) / (close[:-1] + eps))
    ret = np.concatenate([[0.0], ret]).astype(np.float32)
    feats = np.column_stack([
        ret,
        (X[:,0]-X[:,3]),  # open-close
        (X[:,1]-X[:,3]),  # high-close
        (X[:,2]-X[:,3]),  # low-close
    ]).astype(np.float32)

    # intervallumok indexre
    idx_intervals = []
    for (s,e,lab) in intervals:
        si = nearest_index(ts, s)
        ei = nearest_index(ts, e)
        if ei < si:
            si, ei = ei, si
        idx_intervals.append((si, ei, lab))

    samples = []
    labels = []

    N = len(df)
    for start in range(0, N - window + 1, stride):
        end = start + window
        mid = start + window // 2

        y = 0  # None
        for (si, ei, lab) in idx_intervals:
            if si <= mid <= ei:
                y = CFG.label_map[lab]
                break

        samples.append(feats[start:end])
        labels.append(y)

    return np.stack(samples), np.array(labels, dtype=np.int64)

def main():
    LOG.info("=== Data preprocessing ===")
    LOG.info(f"DATA_ROOT={CFG.data_root}")
    LOG.info(f"window={CFG.window}, stride={CFG.stride}")
    ensure_dir(CFG.processed_dir)

    tasks = collect_tasks(CFG.data_root)
    LOG.info(f"Found {len(tasks)} csv tasks")

    X_all, y_all = [], []
    meta = []

    for csv_path, intervals in tasks:
        try:
            df = load_csv(csv_path)
            if len(df) < CFG.window:
                continue
            X, y = build_windows(df, intervals, CFG.window, CFG.stride)
            X_all.append(X)
            y_all.append(y)
            meta.append({"csv": str(csv_path), "n_windows": int(len(y)), "n_pos": int((y>0).sum())})
            LOG.info(f"OK {csv_path.name}: windows={len(y)} pos={int((y>0).sum())}")
        except Exception as e:
            LOG.info(f"SKIP {csv_path}: {e}")

    if not X_all:
        raise RuntimeError("No usable data produced.")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    np.savez_compressed(CFG.processed_dir / "dataset.npz", X=X_all, y=y_all)
    (CFG.processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # osztályeloszlás
    unique, counts = np.unique(y_all, return_counts=True)
    dist = {CFG.classes[int(k)]: int(v) for k,v in zip(unique, counts)}
    LOG.info(f"Dataset saved: X={X_all.shape}, y={y_all.shape}")
    LOG.info(f"Class distribution: {dist}")
    LOG.info("=== Data preprocessing DONE ===")

if __name__ == "__main__":
    main()
