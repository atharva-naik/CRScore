python -m src.models.lstm_review_gen --predict_mode \
--test_filename "./data/Comment_Generation/msg-test.jsonl" \
--output_dir "ckpts/lstm_reviewer" \
--num_layers 6 \
--checkpoint_path "ckpts/lstm_reviewer/best_model.pt"