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

Training is performed on manually annotated time intervals.  
Inference and scanning can be applied to **previously unseen, unlabeled time series** to detect potential flag patterns.

The entire pipeline (preprocessing → training → inference → scanning) is fully automated and containerized.

---

## Extra Credit Justification

The project goes beyond the basic requirements by providing:

- A **fully automated Docker-based pipeline**
- Clear separation of preprocessing, training, inference, and scanning
- Application of the trained model to **unseen time series** using a dedicated scanning script
- Window-level predictions merged into interpretable time segments
- Robust logging and error handling for heterogeneous real-world financial data

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

Data preparation is implemented in:


### File structure

bullflag_project/
│
├── src/
│   ├── data_preprocessing.py   # Loading, matching JSON–CSV, windowing, feature extraction
│   ├── training.py             # CNN training loop and validation
│   ├── evaluation.py           # Model evaluation and metrics
│   ├── inference.py            # Window-level inference on unseen time series
│   ├── scan_timeseries.py      # Full time-series scan and flag segment detection
│   ├── models.py               # CNN (SmallCNN) model definition
│   ├── config.py               # Hyperparameters, paths, class definitions
│   ├── utils.py                # Logging and helper utilities
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
├── README.md                   # Project documentation and instructions

