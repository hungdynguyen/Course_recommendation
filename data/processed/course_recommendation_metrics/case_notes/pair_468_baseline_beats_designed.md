# Pair 468 - Baseline after train beats designed after train

## Summary
- `pair_id`: 468
- `query_text`: Access, Data analysis, Reports, Interpreting
- `truth_courses`: TKKT1133; TKKT1134; TKKT1124

## Metric comparison
- Baseline after train: hit@1=0, hit@3=0, hit@5=0, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Data Analytics for Accounting
2. Data Analytics for Accounting
3. English for Economic Statistics
4. Business Intelligence
5. Financial Analysis
6. Data Analysis 1
7. Decision Support Systems
8. Data Analysis Programming
9. Statistics in Enterprises
10. Data Analysis

### Designed after train
1. Selling Skills
2. Database
3. Database Application
4. Commercial Enterprise Management
5. Business Management 1 (E)
6. Principles of Logistics Management
7. Accounting Informatics
8. Data Analytics for Accounting
9. Quản trị bán hàng

## Why this case matters
- Baseline is much closer to the analytics/reporting signal in the query.
- Designed shifts into sales/logistics/accounting and loses the ground-truth coverage.