# Pair 226 - Baseline after train beats designed after train

## Summary
- `pair_id`: 226
- `query_text`: database management
- `truth_courses`: TIKT1124; TIKT1130; CNTT1152

## Metric comparison
- Baseline after train: hit@1=1, hit@3=1, hit@5=1, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Database Application
2. Database
3. Database
4. Database Management Systems
5. Unstructured database
6. Land Data Base
7. Accounting Informatics
8. Accounting Information Systems 1
9. Accounting Information Systems 1
10. Software and Database Security

### Designed after train
1. Principles of Logistics Management
2. Logistics and Supply Chain Management
3. Commercial Enterprise Management
4. Public Accounting
5. Knowledge Discovery in Database
6. Logistics Planning and Control Systems
7. Revenue Management in Hospitality Business
8. E-Logistics
9. Public Accounting 1
10. Accounting Information Systems 1

## Why this case matters
- Baseline preserves the exact database intent and ranks database courses at the top.
- Designed method drifts into logistics/accounting courses, which are semantically farther from the query.
- This is a strong failure case for the designed method after training.
