import json
import os
from datetime import datetime, timedelta

import boto3

RESULTS_FOLDER = os.environ.get(
    "NBA_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)
BUCKET_NAME = os.getenv("NBA_S3_BUCKET")
S3_PREFIX = os.getenv("NBA_S3_PREFIX", "data/nbastatigami/incoming")


def saveResults(posts):
    date_str = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    data = {
        "date": date_str,
        "posts": [{"player": p.player, "tweet": p.tweet} for p in posts],
    }
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    path = os.path.join(RESULTS_FOLDER, f"statlines_{date_str}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved {path}")
    return path


def saveToCloud(path):
    key = f"{S3_PREFIX}/{os.path.basename(path)}"
    boto3.client("s3").upload_file(path, BUCKET_NAME, key)
    print(f"Uploaded {path} -> s3://{BUCKET_NAME}/{key}")