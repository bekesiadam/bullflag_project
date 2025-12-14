FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

# default pipeline
CMD python /app/src/01-data-preprocessing.py && \
    python /app/src/02-training.py && \
    python /app/src/03-evaluation.py && \
    python /app/src/04-inference.py
