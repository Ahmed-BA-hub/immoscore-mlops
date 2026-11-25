# immoscore-mlops

ImmoScore: house price estimation (MLOps course).

## Setup
```
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install "dvc[s3]"
```

## Data
The data is versioned with DVC and stored on Supabase (S3). To get it:

1. Ask the Team Lead for the S3 keys (they are never in Git).
2. Add them on your machine only:
   ```
   dvc remote modify --local s3 access_key_id "<key>"
   dvc remote modify --local s3 secret_access_key "<secret>"
   ```
3. Download the data:
   ```
   dvc pull
   ```

## How to run
```
python src/data_process.py
python src/split.py
python src/train.py
python src/eval.py
```
All settings are in `params.yaml`.

## Data versions (DVC)
- V1: `corr_threshold` 0.5 (11 features)
- V2: `corr_threshold` 0.6 (6 features)
