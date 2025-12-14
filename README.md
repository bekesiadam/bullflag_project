# Deep Learning Class (VITMMA19) – Project Work

## Project Levels

### Basic Level (for signature)
- Containerization
- Data acquisition and analysis
- Data preparation
- Baseline (reference) model
- Model development
- Basic evaluation

### Outstanding Level (aiming for +1 mark)
- Containerization
- Data acquisition and analysis
- Data cleansing and preparation
- Defining evaluation criteria
- Baseline (reference) model
- Incremental model development
- Advanced evaluation
- ML inference on unseen time series
- Well-structured, fully automated pipeline

---

## Project Information

- **Selected Topic:** Bull-flag detector  
- **Student Name:** Békési Ádám  
- **Aiming for +1 Mark:** Yes  

---

## Solution Description

The goal of this project is to detect **bull flag and bear flag–type price patterns** in financial time series using deep learning.

The solution is based on a **sliding window approach** applied to OHLC (Open, High, Low, Close) price data.  
Each fixed-length window is transformed into numerical features (log-returns and OHLC deltas) and classified by a **convolutional neural network (CNN)**.

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

