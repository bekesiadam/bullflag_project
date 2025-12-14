FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

# default pipeline
CMD python /app/src/data_preprocessing.py && \
    python /app/src/training.py && \
    python /app/src/evaluation.py && \
    python /app/src/inference.py
