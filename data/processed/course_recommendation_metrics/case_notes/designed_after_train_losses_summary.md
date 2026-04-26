# Designed After Train Loses to Baseline After Train

This note summarizes the cases where `designed_after_train` loses to `baseline_after_train` on `hit@10` in the latest run.

## Summary
- Total cases where designed loses baseline on `hit@10`: 6
- Total cases where designed loses baseline on at least one hit metric: 27

## Cases where designed loses on `hit@10`

| pair_id | query_text | truth_courses | baseline hit@10 | designed hit@10 |
|---|---|---|---:|---:|
| 142 | Data Analysis for Facebook Ad Campaigns | TKKT1133; TKkD1127 | 1 | 0 |
| 155 | E-learning product knowledge, English proficiency (IELTS 5.5 or equivalent) | NNTM1176; NNKC 1131 | 1 | 0 |
| 435 | Material Design, AI | CNTT1140 | 1 | 0 |
| 468 | Access, Data analysis, Reports, Interpreting | TKKT1133; TKKT1134; TKKT1124 | 1 | 0 |
| 492 | Microsoft Teams, Basic English reading comprehension, General office software | NNKC 1131; CNTT1192; NNKC 1132; NNTM1181; NNTM1180 | 1 | 0 |
| 753 | Design Tokens, Heuristic Evaluation, A/B Testing, AI Copilot Studio | TIHT1104; CNTT1178 | 1 | 0 |

## What goes wrong in these cases
- The designed pipeline often shifts toward semantically broad but less relevant courses.
- In several cases, the baseline keeps a more literal or skill-aligned ranking and reaches the ground-truth course by top-10.
- The pattern is especially visible on mixed queries that combine office tools, English skills, or product/design terms with specialized software topics.

## Representative cases
- [Pair 142](pair_142_baseline_beats_designed.md)
- [Pair 155](pair_155_baseline_beats_designed.md)
- [Pair 435](pair_435_baseline_beats_designed.md)
- [Pair 468](pair_468_baseline_beats_designed.md)
- [Pair 492](pair_492_baseline_beats_designed.md)
- [Pair 753](pair_753_baseline_beats_designed.md)
