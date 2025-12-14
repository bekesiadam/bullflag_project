# Deep Learning Class (VITMMA19) – Project Work

## Project Information

- **Selected Topic:** Bull-flag detector  
- **Student Name:** Ádám Békési  
- **Aiming for +1 Mark:** Yes  

---

## Solution Description

The goal of this project is to detect **bull flag and bear flag–type price patterns** in financial time series using deep learning.

I based my solution on a sliding window approach, where fixed-length windows are extracted from the OHLC (Open, High, Low, Close) price time series.
Each window is transformed into numerical features (log-returns and OHLC deltas) and classified using a very basic convolutional neural network.

The model predicts one of the following classes for each window:
- `None` – no recognizable pattern
- `Normal`
- `Wedge`
- `Pennant`

---


## Data Preparation

### Raw Data
The raw input consists of CSV files containing financial time series with the following required columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`

Label information is provided via JSON files exported from Label Studio, containing labeled time intervals.

### Preparation Process

Data preparation is implemented in `src/data_preprocessing.py`.  
The script automatically scans all user folders, matches annotation JSON files with their corresponding CSV time series, and extracts labeled time intervals.  
Each time series is converted into fixed-length sliding windows, transformed into numerical features (log-returns and OHLC deltas), and saved in a unified dataset format suitable for model training.

---

## Extra Credit Justification
Even though I used a relatively simple CNN architecture, I focused on building a robust and general data pipeline that incorporates **all valid annotations from all contributors**, not only my own.  

The model is not limited to single labeled CSV files, but can be applied to **previously unseen, unlabeled time series** through a fully automated preprocessing, inference, and scanning pipeline.

---

## Docker Instructions

This project is fully containerized using Docker and runs the complete pipeline:
data preprocessing, training, evaluation, and inference.

### Build

```bash
docker build -t dl-project .
```

### Run (with log capture)

```bash
docker run --rm \
  -v /absolute/path/to/your/local/data:/app/data \
  dl-project > log/run.log 2>&1
```

## File Structure

```text
bullflag_project/
├── src/
│   ├── data_preprocessing.py   # Data loading, JSON–CSV matching, windowing, feature extraction
│   ├── training.py             # CNN training loop and validation
│   ├── evaluation.py           # Model evaluation and metrics
│   ├── inference.py            # Window-level inference on unseen time series
│   ├── scan_timeseries.py      # Full time-series scan and flag segment detection
│   ├── models.py               # CNN model definition
│   ├── config.py               # Hyperparameters, paths, class definitions
│   └── utils.py                # Logging and helper utilities
│
├── log/
│   └── run.log                 # Example log of a full pipeline execution
│
├── processed/
│   ├── dataset.npz             # Preprocessed sliding-window dataset
│   └── meta.json               # Metadata about processed CSV files
│
├── models/
│   └── model.pt                # Trained CNN model
│
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation and instructions
```

