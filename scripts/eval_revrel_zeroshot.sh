python -m src.models.code_review_rel --eval --output_dir ./ckpts/crr_rcr_ccr_zero_shot \
--test_filename ./data/Comment_Generation/msg-test.jsonl \
--dev_filename ./data/Comment_Generation/msg-valid.jsonl \
--code_model_type codereviewer \
--code_model_path microsoft/codereviewer \
--review_model_type codereviewer \
--review_model_path microsoft/codereviewer