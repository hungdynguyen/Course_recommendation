# Pair 75 - Designed after train beats baseline after train

## Summary
- `pair_id`: 75
- `query_text`: Financial models, Media coordination, Profit margin management
- `truth_courses`: TOTC1106; TOTC1108; TOTC1109

## Metric comparison
- Baseline after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0
- Designed after train: hit@1=0, hit@3=1, hit@5=1, hit@10=1

## Top-10 recommendations
### Baseline after train
1. Advertising media planning
2. Advertising agency management
3. Revenue Management in Hospitality Business
4. Advertising Management
5. Modern Journalism
6. Quản trị Tài chính trong du lịch và khách sạn
7. Financial Economics
8. Monetary and Financial Theories 1
9. Corporate Finance
10. Corporate Finance 1

### Designed after train
1. Financial Economics
2. Corporate Finance
3. Fundamentals of Mathematical Finance
4. Models of Financial Corporate
5. The models for analyzing and evaluating the financial assets 1
6. The models for analyzing and evaluating the financial assets 2
7. Business Planning
8. Chuyên đề Tin học quản lý đầu tư
9. Probability Theory

## Why this case matters
- Designed method moved from marketing/media-oriented courses to finance-oriented courses.
- It recovered all 3 ground-truth courses inside top-10, while baseline missed all of them.
- This is a clean example where semantic filtering and ranking improved alignment with the target gap.
