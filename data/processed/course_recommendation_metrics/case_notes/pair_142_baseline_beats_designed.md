# Pair 142 - Baseline after train beats designed after train

## Summary
- `pair_id`: 142
- `query_text`: Data Analysis for Facebook Ad Campaigns
- `truth_courses`: TKKT1133; TKkD1127

## Metric comparison
- Baseline after train: hit@1=0, hit@3=0, hit@5=0, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Data Analytics for Accounting
2. Data Analytics for Accounting
3. Data Driven Marketing
4. Digital Marketing
5. Social Networks
6. Data Analysis Programming
7. Advertising Management
8. Insurance Business Analysis
9. Advertising agency management
10. Data Analysis 1

### Designed after train
1. Digital Marketing
2. Social Networks
3. Advertising Management
4. Marketing Management
5. E-commerce operation management
6. Data Driven Marketing
7. Advertising agency management
8. Data Analytics for Accounting
9. Models of Financial Corporate
10. English - Writing Skills 4

## Why this case matters
- Baseline keeps the recommendation list closer to the analytics/advertising intent.
- Designed method shifts too much toward generic marketing/e-commerce content.
- This is a useful counterexample showing that designed_after_train is not uniformly better.
