"""Recompute the original main statistics and support/calibration figure.

Uses only imported data and frozen source manifests; no original folder needed.
The original summarizer verifies all required frozen source hashes.
"""
import json
import summarize_phase15 as summary
import make_phase15_artifacts as figures

summary.OUT.mkdir(parents=True,exist_ok=True)
for name in ['support','stability','nonnormal']:
    result=getattr(summary,name)()
    print(name,json.dumps(result['primary'],indent=2))
figures.FIG.mkdir(parents=True,exist_ok=True)
figures.TABLE.mkdir(parents=True,exist_ok=True)
figures.support()
print('Rebuilt main support/calibration figure and Table 1 from frozen records.')
