# Pair 492 - Baseline after train beats designed after train

## Summary
- `pair_id`: 492
- `query_text`: Microsoft Teams, Basic English reading comprehension, General office software
- `truth_courses`: NNKC 1131; CNTT1192; NNKC 1132; NNTM1181; NNTM1180

## Metric comparison
- Baseline after train: hit@1=0, hit@3=1, hit@5=1, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Basic Informatics
2. Introduction to Information Technology
3. English 1
4. Microsoft Project
5. Administrative Management
6. English 2
7. English 3
8. Integrated skills - English for economics and business 1
9. Second Foreign Language (French 1)
10. English – Reading & Writing Skills 3

### Designed after train
1. Data Analytics for Accounting
2. Accounting Information Systems 1
3. Accounting Information Systems 2
4. Basic Informatics
5. Marketing Management
6. Kiểm soát nội bộ
7. Information Resources Management
8. Social Networks
9. Web Programming
10. Data Driven Marketing

## Why this case matters
- Baseline captures the office-software and English intent better.
- Designed over-weights unrelated analytics/accounting/marketing courses.