FROM python:3.13-slim

# PYTHONUNBUFFERED: print() output shows up immediately in logs
# NBA_DATA_DIR:     where the combined dataset is (mounted read-only from Data on the PC)
# NBA_MODEL_DIR:    where the fitted PDF .pkl files are (mounted from models on the PC)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NBA_DATA_DIR=/data \
    NBA_MODEL_DIR=/models

# tzdata: lets the TZ variable work, so "yesterday" matches the NBA's Eastern game dates
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so code changes don't force a full reinstall
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Replace your_stats_script.py with the main script's filename.
# pipefail keeps Python's exit code (otherwise tee's would be reported).
CMD ["bash", "-c", "set -o pipefail; echo \"===== $(date '+%F %T') =====\"; python columnPicker.py 2>&1 | tee -a /logs/nbastats.log"]