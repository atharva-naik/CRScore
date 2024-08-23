# CodeReviewEval
Comprehensive semi-automatic evaluation of code reviews

# Compute Metrics:
All the metrics, conciseness (P), comprehensiveness (R) and relevance (F) are all computed using the script below:
```python -m src.metrics.claim_based.relevance_score```
To run all the reference free metrics used for comparison with our metric:
```python -m scripts.run_ref_based_eval```