# Confidence calibration

Both edge sources are overconfident out of the box: the VLM asserts an occlusion at 95%
when it is right about 72% of the time, and the UOAIS geometry score is worse. The math
stack marginalises over occlusion graphs, so it needs probabilities that mean what they
say. This directory holds the calibrators and the frozen parameters.

**These parameters are trained once, on a held-out slice of the synthetic training split,
and then used unchanged for synthetic evaluation, real-world evaluation and deployment.
Nothing is refit per dataset.** That is the point: if the calibration had to be retuned on
each domain it would not be a calibration, it would be a second fit to the test set.

## The two calibrators

**Global Platt** — one temperature and one bias for every edge:

```
p_cal = sigmoid(a · logit(p_raw) + c)
```

**Adaptive Platt** — the same, but the temperature and bias move with scene clutter.
`N_cand` is the number of candidate objects in the scene; a claim about one of three
objects and a claim about one of twelve do not deserve the same trust:

```
phi     = (log1p(N_cand) - mu_phi) / sigma_phi
a_scene = exp(alpha0 + alphaN · phi)
p_cal   = sigmoid(a_scene · logit(p_raw) + c0 + cN · phi)
```

Adaptive is what the pipeline uses. Global is kept as the ablation it is measured against.

## Frozen parameters

`adaptive_platt_model.json` — VLM edges:

| alpha0 | alphaN | c0 | cN | mu_phi | sigma_phi | lambda |
|---:|---:|---:|---:|---:|---:|---:|
| -0.30788 | 0.05107 | -0.61925 | 0.04537 | 1.57988 | 0.44408 | 1.0 |

`adaptive_platt_3d_model.json` — UOAIS 3D edges:

| alpha0 | alphaN | c0 | cN | mu_phi | sigma_phi | lambda |
|---:|---:|---:|---:|---:|---:|---:|
| -2.01835 | -0.41131 | -0.79327 | -0.57137 | 2.71262 | 0.48134 | 0.001 |

## What they were fit on

Scene-disjoint splits by image id, chosen to match row count, label rate, probability band
and `N_cand` band. No sample was dropped for its label or outcome, and model validation
scores were not used to pick the split.

| source | train edges / scenes | valid edges / scenes | train ECE | after |
|---|---|---|---:|---:|
| VLM | 726 / 283 | 181 / 70 | 0.160 | **0.029** |
| UOAIS 3D | 2042 / 364 | 445 / 91 | 0.163 | **0.022** |

## Fusion

Calibrated VLM and 3D probabilities are combined in logit space, with the prior removed
once so it is not double-counted:

```
logit p_fuse = w_vlm · logit p_vlm + w_3d · logit p_3d + w_prior · logit P(y)
```

The shipped setting is **w_vlm = 0.5, w_3d = 0.5, w_prior = -1**, `P(y) = 0.270914`, the
`w0.5_0.5_m1` in the CSV filenames. An edge that only one source sees keeps that source's
calibrated probability untouched — there is nothing to combine and no prior to remove.

Equal weights are deliberate. The two sources are close in quality after calibration and
disagree on different scenes; tuning the split on the test set would be fitting the answer.

## Files

| | |
|---|---|
| `train_adaptive_platt.py`, `train_global_platt.py` | fit, and write the `*_model.json` |
| `infer_adaptive_platt.py`, `infer_global_platt.py` | apply a fitted model to a CSV |
| `*_model.json` | the frozen parameters used everywhere |
| `*_report.json` | fit diagnostics: NLL, Brier, ECE before and after |
| `calibration_split_report*.json` | how the splits were drawn |
| `train_*.csv`, `valid_*.csv` | the exact rows each model was fit on |

## Refitting

Only if you change the upstream scorer. The pipeline does not need it.

```bash
cd calibration
python train_adaptive_platt.py --train-csv train_adaptive.csv --valid-csv valid_adaptive.csv \
    --model-out adaptive_platt_model.json --report-out adaptive_platt_report.json
```

Applying a fitted model to new scores:

```bash
python infer_adaptive_platt.py --model adaptive_platt_model.json \
    --input-csv scores.csv --output-csv scores_calibrated.csv
```
