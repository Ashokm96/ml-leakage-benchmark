# ML Leakage & Validation Benchmark

Benchmark for evaluating model-generated ML code for data leakage, incorrect validation, and unstable training.

## Tasks

| Task | Failure Tested | Wrong Pattern | Correct Pattern |
| :--- | :--- | :--- | :--- |
| Task 1: Leakage | Data Leakage | Scaler fit on full data, target encoding before split | Pipeline, fit-only-on-train |
| Task 2: Validation | Incorrect Validation | Random split on time-series | TimeSeriesSplit, no shuffle |
| Task 3: Unstable Training | Unstable Training | Unscaled features + high LR, no seed | StandardScaler in Pipeline, lower LR, seeded |

## Run

```bash
pip install -r requirements.txt
pytest -v
```

Starter code intentionally contains the failure pattern. The solution code demonstrates the corrected pattern, and the graders verify the expected implementation and behavior.

## Evaluation Rubric

1. **Leakage:** No test information is used during training; preprocessing is fit only on training data.
2. **Validation:** The split strategy matches the data type, including `TimeSeriesSplit` for time-series data.
3. **Reproducibility:** Randomness is seeded and repeated runs produce the same result.
4. **Performance:** Accuracy or score is not worse than the baseline, with no NaNs or crashes.

## Results

Run `pytest -v` to reproduce the grader results locally. The suite contains three checks for each of the three tasks: solution structure/behavior and confirmation that each starter contains its intended failure pattern.

No network calls or external data are required; all datasets are synthetic.
