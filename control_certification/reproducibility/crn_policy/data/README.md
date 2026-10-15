# Recorded experiment data

The raw data is distributed as a [release asset](https://github.com/yeoneung/crn_policy/releases/download/experiments-v1/crn_policy_experiment_data.zip), approximately 291 MB compressed.

```sh
python download_data.py
python reproduce.py verify --data data/crn_policy_experiment_data.zip
```

The downloader verifies the archive SHA-256 and byte count. The runner also verifies all 984 members before extraction. Exact hashes and sizes are in `manifest.json`.

The asset contains synthetic calibration observations, ordered return banks, policy checkpoints, experiment protocols, original source snapshots, and all retained positive, negative, censored and fallback outcomes for the included studies. The small `recorded_summaries/` directory contains aggregate views of those records.

Keep this ZIP outside Git; `.gitignore` excludes it and partial downloads.
