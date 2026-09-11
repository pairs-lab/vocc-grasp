# Full results — UnoBench `test_GT_small_1800`

Every system evaluated with the same script, the same split, and (for the fused rows)
the same frozen calibration. Reproduce any row with:

```bash
cd math && python report_unobench.py --csv <the csv named in the section> --tau-edge 0.09
```


- generated: 2026-08-19 23:03:45
- GT: `UnoBench/subset_difficulty/test_GT_small_1800.json`, grouped by the original `difficulty` field (600/600/600)
- Scored from each log's stored predictions; **no model was re-run**.
- `evaluate_nlp.py` labels the 600 records whose GT paths are all length 1 as
  `No-Occ`; renamed to `Easy` here so every block uses the same three groups.
  Same 600 cases, same numbers - name only.
- Calibration tables use `recompute_calibration_metrics.metrics`: 10 equal-width
  bins on [0,1]. Edge label = the pair is a GT occlusion edge of that sample
  (`1->2` means object 1 occludes object 2). Object label = the object is a
  **source** of the sample's GT occlusion DAG, i.e. nothing occludes it - which
  is what the object scores predict (`prob_free` for the VLMs, the free-set
  marginal `q` for the math stack).
- Calibration is reported as a single **overall** row per table (all 1800 cases
  pooled); the per-difficulty ECE/NLL/Brier split has been removed.
- Per-row scores live in `scores/<system>_edges.csv` and `<system>_objects.csv`.
- Exception: the math stack's SR/OR numbers are its final post-math result, but its
  **edge** calibration is measured on the pre-math scores (adaptive VLM temperature
  fused with the 3D prior, no math layer) from
  `logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv`.

## UOAIS 3D (filtered edges)

- source : logs/uoais_pipeline_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/uoais_pipeline_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 6838
  Valid hits (object_id>0): 6838
  Miss / empty (object_id=0): 0
  Hit ratio: 100.00%
========================================

=== Easy ===
SR: P=0.9217, R=0.9217, F1=0.9217 (600 samples)
MP_NED: 0.0373

=== Medium ===
SR: P=0.6639, R=0.6492, F1=0.6480 (600 samples)
Occlusion reasoning: P=0.6469, R=0.6908, F1=0.6524
MP_NED: 0.2217

=== Hard ===
SR: P=0.5553, R=0.5204, F1=0.5238 (600 samples)
Occlusion reasoning: P=0.6350, R=0.4834, F1=0.5102
MP_NED: 0.4467

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.523
```

**Edge level** — `scores/uoais_pipeline_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1850 | 0.7227 | 0.3660 | 0.9301 | 0.3427 |

**Object level** — `scores/uoais_pipeline_1800_objects.csv`

_No per-object confidence in this log._

## UOAIS 3D (unfiltered edges, legacy)

- source : logs/uoais_1800_legacy/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/uoais_1800_legacy/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 10169
  Valid hits (object_id>0): 10169
  Miss / empty (object_id=0): 0
  Hit ratio: 100.00%
========================================

=== Easy ===
SR: P=0.8583, R=0.8583, F1=0.8583 (600 samples)
MP_NED: 0.0723

=== Medium ===
SR: P=0.6131, R=0.6275, F1=0.6107 (600 samples)
Occlusion reasoning: P=0.5755, R=0.6900, F1=0.6001
MP_NED: 0.2735

=== Hard ===
SR: P=0.4821, R=0.4668, F1=0.4617 (600 samples)
Occlusion reasoning: P=0.5712, R=0.4881, F1=0.4806
MP_NED: 0.4856

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.483
```

**Edge level** — `scores/uoais_1800_legacy_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 2436 | 0.5595 | 0.2540 | 0.8110 | 0.2893 |

**Object level** — `scores/uoais_1800_legacy_objects.csv`

_No per-object confidence in this log._

## UnoGrasp

- source : logs/unograsp_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/unograsp_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5377
  Valid hits (object_id>0): 5294
  Miss / empty (object_id=0): 83
  Hit ratio: 98.46%
========================================

=== Easy ===
SR: P=0.9150, R=0.9150, F1=0.9150 (600 samples)
MP_NED: 0.0699

=== Medium ===
SR: P=0.7333, R=0.6828, F1=0.6989 (600 samples)
Occlusion reasoning: P=0.6078, R=0.6000, F1=0.5974
MP_NED: 0.2519

=== Hard ===
SR: P=0.5825, R=0.4997, F1=0.5238 (600 samples)
Occlusion reasoning: P=0.5224, R=0.2631, F1=0.3276
MP_NED: 0.5707

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.534
```

UnoGrasp only stores rendered `<think>/<answer>` text, with no per-edge or per-object confidence, so there is nothing to calibrate.

## Gemini (no 3D ref)

- source : logs/gemini_noref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/gemini_noref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5947
  Valid hits (object_id>0): 5945
  Miss / empty (object_id=0): 2
  Hit ratio: 99.97%
========================================

=== Easy ===
SR: P=0.9231, R=0.9267, F1=0.9242 (600 samples)
MP_NED: 0.0694

=== Medium ===
SR: P=0.7026, R=0.7044, F1=0.6913 (600 samples)
Occlusion reasoning: P=0.6020, R=0.6450, F1=0.6102
MP_NED: 0.2714

=== Hard ===
SR: P=0.4701, R=0.4665, F1=0.4477 (600 samples)
Occlusion reasoning: P=0.5781, R=0.3366, F1=0.4015
MP_NED: 0.5472

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.516
```

**Edge level** — `scores/gemini_noref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1456 | 0.7129 | 0.2019 | 0.7691 | 0.2412 |

**Object level** — `scores/gemini_noref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 6501 | 0.2737 | 0.5857 | 5.9126 | 0.5887 |

## Gemini + 3D ref

- source : logs/gemini_uoais_ref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/gemini_uoais_ref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5811
  Valid hits (object_id>0): 5809
  Miss / empty (object_id=0): 2
  Hit ratio: 99.97%
========================================

=== Easy ===
SR: P=0.9433, R=0.9483, F1=0.9446 (600 samples)
MP_NED: 0.0650

=== Medium ===
SR: P=0.6525, R=0.6278, F1=0.6302 (600 samples)
Occlusion reasoning: P=0.5492, R=0.5633, F1=0.5472
MP_NED: 0.2953

=== Hard ===
SR: P=0.5199, R=0.4686, F1=0.4755 (600 samples)
Occlusion reasoning: P=0.5611, R=0.3421, F1=0.3939
MP_NED: 0.5505

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.513
```

**Edge level** — `scores/gemini_uoais_ref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1400 | 0.7286 | 0.1462 | 0.6777 | 0.2150 |

**Object level** — `scores/gemini_uoais_ref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 7541 | 0.2419 | 0.4569 | 2.8521 | 0.4374 |

## GPT-4o (no 3D ref)

- source : logs/gpt4o_noref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/gpt4o_noref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 4777
  Valid hits (object_id>0): 4775
  Miss / empty (object_id=0): 2
  Hit ratio: 99.96%
========================================

=== Easy ===
SR: P=0.6819, R=0.6850, F1=0.6826 (600 samples)
MP_NED: 0.1896

=== Medium ===
SR: P=0.5969, R=0.5661, F1=0.5732 (600 samples)
Occlusion reasoning: P=0.5169, R=0.5058, F1=0.5066
MP_NED: 0.3300

=== Hard ===
SR: P=0.3119, R=0.2485, F1=0.2658 (600 samples)
Occlusion reasoning: P=0.5050, R=0.1795, F1=0.2580
MP_NED: 0.6606

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.380
```

**Edge level** — `scores/gpt4o_noref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1186 | 0.5320 | 0.3625 | 1.1015 | 0.3774 |

**Object level** — `scores/gpt4o_noref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 6297 | 0.2576 | 0.6090 | 7.9075 | 0.6100 |

## GPT-4o + 3D ref

- source : logs/gpt4o_uoaisref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/gpt4o_uoaisref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 4679
  Valid hits (object_id>0): 4677
  Miss / empty (object_id=0): 2
  Hit ratio: 99.96%
========================================

=== Easy ===
SR: P=0.8239, R=0.8333, F1=0.8263 (600 samples)
MP_NED: 0.1326

=== Medium ===
SR: P=0.6143, R=0.5778, F1=0.5864 (600 samples)
Occlusion reasoning: P=0.5117, R=0.4942, F1=0.4989
MP_NED: 0.3356

=== Hard ===
SR: P=0.2807, R=0.2304, F1=0.2407 (600 samples)
Occlusion reasoning: P=0.5200, R=0.1843, F1=0.2646
MP_NED: 0.6672

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.413
```

**Edge level** — `scores/gpt4o_uoaisref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1105 | 0.5991 | 0.2571 | 0.8763 | 0.3057 |

**Object level** — `scores/gpt4o_uoaisref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 9824 | 0.2070 | 0.6505 | 8.8984 | 0.6506 |

## InternVL3.5-14B-4bit (no 3D ref)

- source : logs/internvl3_5_14b_4bit_noref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/internvl3_5_14b_4bit_noref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 7189
  Valid hits (object_id>0): 7176
  Miss / empty (object_id=0): 13
  Hit ratio: 99.82%
========================================

=== Easy ===
SR: P=0.3088, R=0.7533, F1=0.4017 (600 samples)
MP_NED: 0.1475

=== Medium ===
SR: P=0.4412, R=0.6892, F1=0.4790 (600 samples)
Occlusion reasoning: P=0.2562, R=0.2442, F1=0.2473
MP_NED: 0.4621

=== Hard ===
SR: P=0.2776, R=0.3899, F1=0.2680 (600 samples)
Occlusion reasoning: P=0.3457, R=0.1176, F1=0.1708
MP_NED: 0.7073

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.287
```

**Edge level** — `scores/internvl3_5_14b_4bit_noref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 913 | 0.4053 | 0.4554 | 1.2418 | 0.4456 |

**Object level** — `scores/internvl3_5_14b_4bit_noref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 7733 | 0.2136 | 0.5977 | 3.3530 | 0.5510 |

## InternVL3.5-14B-4bit + 3D ref

- source : logs/internvl3_5_14b_4bit_uoaisref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/internvl3_5_14b_4bit_uoaisref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5925
  Valid hits (object_id>0): 5916
  Miss / empty (object_id=0): 9
  Hit ratio: 99.85%
========================================

=== Easy ===
SR: P=0.3024, R=0.6100, F1=0.3704 (600 samples)
MP_NED: 0.1783

=== Medium ===
SR: P=0.6163, R=0.6356, F1=0.6045 (600 samples)
Occlusion reasoning: P=0.4700, R=0.4550, F1=0.4594
MP_NED: 0.3507

=== Hard ===
SR: P=0.3180, R=0.2935, F1=0.2856 (600 samples)
Occlusion reasoning: P=0.4772, R=0.1779, F1=0.2505
MP_NED: 0.6708

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.315
```

**Edge level** — `scores/internvl3_5_14b_4bit_uoaisref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1574 | 0.4079 | 0.3135 | 0.9310 | 0.3461 |

**Object level** — `scores/internvl3_5_14b_4bit_uoaisref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 10656 | 0.1994 | 0.5666 | 2.5109 | 0.5217 |

## Qwen3.5-9B-4bit (no 3D ref)

- source : logs/qwen3_5_9b_4bit_noref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/qwen3_5_9b_4bit_noref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5017
  Valid hits (object_id>0): 5015
  Miss / empty (object_id=0): 2
  Hit ratio: 99.96%
========================================

=== Easy ===
SR: P=0.7664, R=0.7717, F1=0.7681 (600 samples)
MP_NED: 0.1518

=== Medium ===
SR: P=0.5414, R=0.5172, F1=0.5201 (600 samples)
Occlusion reasoning: P=0.4675, R=0.4567, F1=0.4569
MP_NED: 0.3485

=== Hard ===
SR: P=0.2887, R=0.2585, F1=0.2588 (600 samples)
Occlusion reasoning: P=0.5253, R=0.1926, F1=0.2739
MP_NED: 0.6516

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.387
```

**Edge level** — `scores/qwen3_5_9b_4bit_noref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1216 | 0.5247 | 0.3643 | 1.1608 | 0.3804 |

**Object level** — `scores/qwen3_5_9b_4bit_noref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 2561 | 0.3354 | 0.4446 | 5.9601 | 0.4465 |

## Qwen3.5-9B-4bit + 3D ref

- source : logs/qwen3_5_9b_4bit_uoaisref_test_1800/predictions.jsonl (1800 predictions)
- command: python evaluate_nlp.py --pred_path logs/qwen3_5_9b_4bit_uoaisref_test_1800/predictions.jsonl --gt_path UnoBench/subset_difficulty/test_GT_small_1800.json --npz_root UnoBench/annotations --dataset_type synthetic --difficulty_field difficulty

```text
========== Evaluation Summary ==========
Total parsed coordinates : 5208
  Valid hits (object_id>0): 5206
  Miss / empty (object_id=0): 2
  Hit ratio: 99.96%
========================================

=== Easy ===
SR: P=0.6792, R=0.6800, F1=0.6794 (600 samples)
MP_NED: 0.1946

=== Medium ===
SR: P=0.6442, R=0.6122, F1=0.6192 (600 samples)
Occlusion reasoning: P=0.5700, R=0.5608, F1=0.5608
MP_NED: 0.3187

=== Hard ===
SR: P=0.3167, R=0.2746, F1=0.2812 (600 samples)
Occlusion reasoning: P=0.5783, R=0.2172, F1=0.3062
MP_NED: 0.6307

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.395
```

**Edge level** — `scores/qwen3_5_9b_4bit_uoaisref_test_1800_edges.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 1384 | 0.5405 | 0.2480 | 0.9415 | 0.3252 |

**Object level** — `scores/qwen3_5_9b_4bit_uoaisref_test_1800_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 5186 | 0.2661 | 0.5782 | 7.4645 | 0.5759 |

## D3G

- source : d3g_unobench/output/pred_1800/results.jsonl (1800 predictions)
- command: cd d3g_unobench && python report_nlp_style.py --skip-selfcheck

```text
=== Easy ===
SR: P=0.9800, R=0.9800, F1=0.9800 (600 samples)
MP_NED: 0.0107

=== Medium ===
SR: P=0.5494, R=0.5461, F1=0.5417 (600 samples)
Occlusion reasoning: P=0.5241, R=0.5642, F1=0.5314
MP_NED: 0.2714

=== Hard ===
SR: P=0.3861, R=0.3800, F1=0.3748 (600 samples)
Occlusion reasoning: P=0.4270, R=0.3272, F1=0.3410
MP_NED: 0.5376

Balanced SR-F1 (Group-weighted) = 0.632
```

D3G stores `candidate_edges` with only `from`/`to` - it publishes no per-edge or per-object confidence, so there is nothing to calibrate.

## VLM + math stack, t=0 (no edge filtering)

- source : logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv
- command: cd math_run && python report_unobench.py --csv ../logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv --tau-edge 0.09

```text
========== Evaluation Summary ==========
Scenes loaded from CSV      : 1553  (targets guessed 0)
GT records with no candidate: 247
Coverage mode               : full  (absent => 'no occlusion', pred = ({X}, [[X]]))
tau_set=0.50  tau_act=0.50
========================================

=== Easy ===
SR: P=0.9383, R=0.9383, F1=0.9383 (600 samples)
MP_NED: 0.0335

=== Medium ===
SR: P=0.6861, R=0.6775, F1=0.6724 (600 samples)
Occlusion reasoning: P=0.6725, R=0.7450, F1=0.6856
MP_NED: 0.2134

=== Hard ===
SR: P=0.5861, R=0.5525, F1=0.5559 (600 samples)
Occlusion reasoning: P=0.6578, R=0.5398, F1=0.5491
MP_NED: 0.4273

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.722
```

**Edge level** — `scores/math_stack_edges_calib_fused.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 6467 | 0.2709 | 0.1393 | 0.5184 | 0.1680 |

**Object level** — `scores/math_stack_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 7530 | 0.2141 | 0.0856 | 0.9493 | 0.1040 |

## Qwen3.5-9B-4bit + 3D fused + math stack

- source : logs/fused_qwen_uoaisref/fused_adaptive_w0.5_0.5_m1.csv
- command: cd math_run && python report_unobench.py --csv ../logs/fused_qwen_uoaisref/fused_adaptive_w0.5_0.5_m1.csv --tau-edge 0.09

```text
========== Evaluation Summary ==========
Scenes loaded from CSV      : 1560  (targets guessed 0)
GT records with no candidate: 240
Coverage mode               : full  (absent => 'no occlusion', pred = ({X}, [[X]]))
tau_set=0.50  tau_act=0.50
========================================

=== Easy ===
SR: P=0.7500, R=0.7500, F1=0.7500 (600 samples)
MP_NED: 0.1435

=== Medium ===
SR: P=0.6739, R=0.6744, F1=0.6642 (600 samples)
Occlusion reasoning: P=0.6651, R=0.7458, F1=0.6819
MP_NED: 0.2198

=== Hard ===
SR: P=0.5756, R=0.5383, F1=0.5423 (600 samples)
Occlusion reasoning: P=0.6686, R=0.5267, F1=0.5440
MP_NED: 0.4382

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.652
```

**Edge level** — `scores/math_stack_qwen_edges_calib_fused.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 6754 | 0.2567 | 0.1344 | 0.5628 | 0.1881 |

**Object level** — `scores/math_stack_qwen_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 7580 | 0.2148 | 0.0974 | 0.8837 | 0.1258 |

## InternVL3.5-14B-4bit + 3D fused + math stack

- source : logs/fused_internvl_uoaisref/fused_adaptive_w0.5_0.5_m1.csv
- command: cd math_run && python report_unobench.py --csv ../logs/fused_internvl_uoaisref/fused_adaptive_w0.5_0.5_m1.csv --tau-edge 0.09

```text
========== Evaluation Summary ==========
Scenes loaded from CSV      : 1572  (targets guessed 0)
GT records with no candidate: 228
Coverage mode               : full  (absent => 'no occlusion', pred = ({X}, [[X]]))
tau_set=0.50  tau_act=0.50
========================================

=== Easy ===
SR: P=0.7750, R=0.7750, F1=0.7750 (600 samples)
MP_NED: 0.1360

=== Medium ===
SR: P=0.6417, R=0.6403, F1=0.6324 (600 samples)
Occlusion reasoning: P=0.6226, R=0.6950, F1=0.6373
MP_NED: 0.2363

=== Hard ===
SR: P=0.5350, R=0.4962, F1=0.5021 (600 samples)
Occlusion reasoning: P=0.6199, R=0.4833, F1=0.5019
MP_NED: 0.4562

=== Overall (Group-weighted) ===
Balanced SR-F1 (Group-weighted) = 0.636
```

**Edge level** — `scores/math_stack_internvl_edges_calib_fused.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 6854 | 0.2477 | 0.1426 | 0.5838 | 0.1977 |

**Object level** — `scores/math_stack_internvl_objects.csv`

| group | n | pos_rate | ECE | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| overall | 7647 | 0.2149 | 0.1047 | 0.9877 | 0.1335 |

## Summary

| system | Easy SR-F1 | Medium SR-F1 | Medium OR-F1 | Hard SR-F1 | Hard OR-F1 | Balanced SR-F1 | edge ECE | object ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UOAIS 3D (filtered edges) | 0.9217 | 0.6480 | 0.6524 | 0.5238 | 0.5102 | 0.5230 | 0.3660 | - |
| UOAIS 3D (unfiltered edges, legacy) | 0.8583 | 0.6107 | 0.6001 | 0.4617 | 0.4806 | 0.4830 | 0.2540 | - |
| UnoGrasp | 0.9150 | 0.6989 | 0.5974 | 0.5238 | 0.3276 | 0.5340 | - | - |
| Gemini (no 3D ref) | 0.9242 | 0.6913 | 0.6102 | 0.4477 | 0.4015 | 0.5160 | 0.2019 | 0.5857 |
| Gemini + 3D ref | 0.9446 | 0.6302 | 0.5472 | 0.4755 | 0.3939 | 0.5130 | 0.1462 | 0.4569 |
| GPT-4o (no 3D ref) | 0.6826 | 0.5732 | 0.5066 | 0.2658 | 0.2580 | 0.3800 | 0.3625 | 0.6090 |
| GPT-4o + 3D ref | 0.8263 | 0.5864 | 0.4989 | 0.2407 | 0.2646 | 0.4130 | 0.2571 | 0.6505 |
| InternVL3.5-14B-4bit (no 3D ref) | 0.4017 | 0.4790 | 0.2473 | 0.2680 | 0.1708 | 0.2870 | 0.4554 | 0.5977 |
| InternVL3.5-14B-4bit + 3D ref | 0.3704 | 0.6045 | 0.4594 | 0.2856 | 0.2505 | 0.3150 | 0.3135 | 0.5666 |
| Qwen3.5-9B-4bit (no 3D ref) | 0.7681 | 0.5201 | 0.4569 | 0.2588 | 0.2739 | 0.3870 | 0.3643 | 0.4446 |
| Qwen3.5-9B-4bit + 3D ref | 0.6794 | 0.6192 | 0.5608 | 0.2812 | 0.3062 | 0.3950 | 0.2480 | 0.5782 |
| D3G | 0.9800 | 0.5417 | 0.5314 | 0.3748 | 0.3410 | 0.6320 | - | - |
| VLM + math stack, t=0 (no edge filtering) | 0.9383 | 0.6724 | 0.6856 | 0.5559 | 0.5491 | 0.7220 | 0.1393 | 0.0856 |
| Qwen3.5-9B-4bit + 3D fused + math stack | 0.7500 | 0.6642 | 0.6819 | 0.5423 | 0.5440 | 0.6520 | 0.1344 | 0.0974 |
| InternVL3.5-14B-4bit + 3D fused + math stack | 0.7750 | 0.6324 | 0.6373 | 0.5021 | 0.5019 | 0.6360 | 0.1426 | 0.1047 |

OR-F1 is blank for Easy: those 600 cases have no GT occlusion edge.
