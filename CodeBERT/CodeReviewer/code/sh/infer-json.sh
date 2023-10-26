# batch size 6 for 16 GB GPU

mnt_dir="/home/arnaik/CodeReviewEval"

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23332 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

# change break_cnt to truncate the number of examples (useful at debug time maybe)
#   --break_cnt -1 \  will keep the whole dataset 
python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_infer_msg.py  \
  --model_name_or_path microsoft/codereviewer \
  --load_model_path ../../../../ckpts/gen_study_rel/checkpoints-1800-5.62 \
  --output_dir ../../../../ckpts/gen_study_rel/checkpoints-1800-5.62 \
  --eval_file ../../../../data/Comment_Generation/msg-test.jsonl \
  --out_file test_out.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 32 \
  --beam_size 10 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input \
