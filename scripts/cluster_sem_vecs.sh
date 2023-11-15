python -m src.indexing.cluster_code_changes_semvec --checkpoint_path "./ckpts/crr_rcr_ccr_0.05/best_model.pth" \
--code_model_type codereviewer \
--code_model_path microsoft/codereviewer \
--review_model_type codereviewer \
--review_model_path microsoft/codereviewer \
--device "cuda:0" 
python -m src.indexing.check_cluster_alignment