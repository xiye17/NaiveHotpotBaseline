export SQUAD_DIR=outputs

exp_prefix="exps/hpqa_${exp_id}/"

mkdir ${exp_prefix}

CUDA_VISIBLE_DEVICES=0 \
python -u run_sq_qa.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --do_train \
  --do_eval \
  --disable_tqdm \
  --train_file $SQUAD_DIR/train_hp_squad.json \
  --predict_file $SQUAD_DIR/dev_hp_squad.json \
  --learning_rate 3e-5 \
  --weight_decay 0.1 \
  --evaluate_during_training \
  --num_train_epochs 2.0 \
  --overwrite_output_dir \
  --max_seq_length 512 \
  --doc_stride 128 \
  --logging_steps 3000 \
  --save_steps 3000 \
  --warmup_steps 1500 \
  --data_dir "${exp_prefix}" \
  --output_dir "${exp_prefix}output" \
  --logging_dir "${exp_prefix}tblogging" \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 1 2>&1 | tee "${exp_prefix}log.txt"
