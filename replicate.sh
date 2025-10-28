#!/usr/bin/env bash
set -euo pipefail

# 1) Generate analysis figures
jupyter nbconvert --to notebook --execute src/Analysis/Analysis.ipynb

echo "Done. Figures are under src/Analysis."

# 2) Ensure the pretrained weights are downloaded and present under 'src/TestEvaluation/chkpt/'
# then Evaluate using pretrained weights
jupyter nbconvert --to notebook --execute src/TestEvaluation/Main_Evaluation.ipynb

echo "Done. Results are under src/TestEvaluation/results ."
