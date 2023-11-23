TEMP=0.005
python -m src.models.code_review_rel --eval --checkpoint_path "ckpts/crr_rcr_ccr_unnorm/best_model.pth" \
--test_filename ./data/Comment_Generation/msg-test.jsonl \
--dev_filename ./data/Comment_Generation/msg-valid.jsonl \
--code_model_type codereviewer \
--code_model_path microsoft/codereviewer \
--review_model_type codereviewer \
--review_model_path microsoft/codereviewer \
--temperature $TEMP \
--use_unnormalized_loss