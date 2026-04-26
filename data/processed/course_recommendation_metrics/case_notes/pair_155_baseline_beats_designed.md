# Pair 155 - Baseline after train beats designed after train

## Summary
- `pair_id`: 155
- `query_text`: E-learning product knowledge, English proficiency (IELTS 5.5 or equivalent)
- `truth_courses`: NNTM1176; NNKC 1131

## Metric comparison
- Baseline after train: hit@1=0, hit@3=1, hit@5=1, hit@10=1
- Designed after train: hit@1=0, hit@3=0, hit@5=0, hit@10=0

## Top-10 recommendations
### Baseline after train
1. Applied English Grammar
2. English – Listening and Speaking skills 1
3. English – Listening and Speaking skills 2
4. English – Reading & Writing Skills 3
5. English – Listening and Speaking skills 3
6. English for IT
7. English for International Economics and Business
8. Second Foreign Language (French 2)
9. Interpretation 1
10. Vietnamese Language 3 (Vietnamese Language in Economics and Business 3)

### Designed after train
1. Second Foreign Language (French 2)
2. Fundamentals of E-commerce
3. English – Reading & Writing Skills 3
4. English for International Economics and Business
5. Drafting Legal Documents
6. English for Public Relations
7. Private International Law
8. E-commerce operation management
9. Selling Skills
10. Electronic Commerce Systems

## Why this case matters
- Baseline stays close to the English/office-skills intent and recovers a ground-truth course inside top-10.
- Designed drifts toward unrelated business/legal/e-commerce content.