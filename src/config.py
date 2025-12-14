from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # paths (Dockerben /app)
    data_root: Path = Path("/app/data")       
    #data_root: Path = Path(r"C:/Users/Adam/Desktop/bullflagdetector")

    processed_dir: Path = Path("/app/processed")
    model_dir: Path = Path("/app/models")

    # data
    window: int = 256
    stride: int = 16  # gyors mintavétel
    min_overlap_center: bool = True

    # labels
    classes = ["None", "Normal", "Wedge", "Pennant"]  # 4-class detector
    label_map = {"Normal": 1, "Wedge": 2, "Pennant": 3}

    # train
    seed: int = 42
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3

CFG = Config()
