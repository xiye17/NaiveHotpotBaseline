#/bin/sh!
exp_id=$1

exp_prefix="exps/mini_${exp_id}/"

mkdir ${exp_prefix}

CUDA_VISIBLE_DEVICES=1,2,3 \
python -u run_ranker.py \
    --do_mini \
    --do_train \
    --do_eval \
    --disable_tqdm \
    --model_name_or_path roberta-base \
    --output_dir "${exp_prefix}output" \
    --logging_dir "${exp_prefix}tblogging" \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --label_names "sp_labels" "ct_labels" "doc_masks" \
    --evaluation_strategy "steps" \
    --eval_steps 32 \
    --save_steps 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 2>&1 | tee "${exp_prefix}log.txt"