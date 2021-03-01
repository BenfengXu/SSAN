set -eux

pretrained_model=./pretrained_lm/roberta_base/
data_dir=./data/DocRED/

lr=5e-5
epoch=40
batch_size=4

CUDA_VISIBLE_DEVICES=0 python ./run_docred.py \
  --model_type roberta \
  --entity_structure biaffine \
  --model_name_or_path ${pretrained_model} \
  --do_train \
  --do_eval \
  --data_dir ${data_dir} \
  --max_seq_length 512 \
  --max_ent_cnt 42 \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate ${lr} \
  --num_train_epochs ${epoch} \
  --warmup_ratio 0.1 \
  --output_dir checkpoints \
  --seed 42 \
  --logging_steps 10
