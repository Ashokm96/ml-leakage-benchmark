# Rubric - Evaluate Model-Generated ML Code

1. Leakage (40%): No test info in train. Scaler/encoder fit only on train.
2. Validation (30%): Correct split for data type. Time-series = TimeSeriesSplit, no shuffle.
3. Reproducibility (20%): random_state set, seeded, locked requirements, same result on rerun.
4. Performance (10%): Accuracy not worse than baseline, no NaNs, no crash.

**Pass = All 4 categories pass.**
