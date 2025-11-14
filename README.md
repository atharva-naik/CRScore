# CodeReviewEval
Comprehensive semi-automatic evaluation of code reviews

## Fine-tuned Magicoder Model for Claim Generation
For downloading the model, please use the HuggingFace link here: https://huggingface.co/teledia-cmu/Magicoder-6.7B-code-change-summ-impl

## Download dataset for Comment Generation:
Please use the following [Zenodo](https://zenodo.org/records/6900648) link to access the CodeReviewer message/code comment generation dataset. You will need to download and extract the ```Comment_Generation.zip``` folder and place it inside the ```data/``` folder

## Compute Metrics:
All the metrics, conciseness (P), comprehensiveness (R) and relevance (F) are all computed using the script below:
```python -m src.metrics.claim_based.relevance_score```
To run all the reference free metrics used for comparison with our metric:
```python -m scripts.run_ref_based_eval```

```
@article{naik2024crscore,
  title={CRScore: Grounding Automated Evaluation of Code Review Comments in Code Claims and Smells},
  author={Naik, Atharva and Alenius, Marcus and Fried, Daniel and Rose, Carolyn},
  journal={arXiv preprint arXiv:2409.19801},
  year={2024}
}
```
