# Pair 87 - Baseline after train beats designed after train

## Summary
- `pair_id`: 87
- `query_text`: FMEA, project-management-tools-JIRA
- `truth_courses`: CNTT1159; TIKT1113

## Metric comparison
- Baseline after train: hit@1=0, hit@3=0, hit@5=0, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Project Management
2. Project Execution Management
3. Chuyên đề Tin học quản lý đầu tư
4. Microsoft Project
5. Project Quality Management
6. Quantitative risk management 1
7. Program and Project Management
8. Information Technology Project Management
9. Management Control
10. Investment Project Design and Management

### Designed after train
1. Project Quality Management
2. Quality Management
3. Project Execution Management
4. Project Management
5. Product Management

## Why this case matters
- Baseline catches the project-management signal strongly enough to recover one of the ground-truth courses by top-10.
- Designed method truncates earlier and returns a shorter, less useful list here.
- This suggests the designed pipeline may be over-constrained on some mixed skill queries.
