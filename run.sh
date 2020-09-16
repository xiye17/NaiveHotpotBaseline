#/bin/sh!
exp_id=$1

exp_prefix="exps/${exp_id}/"

mkdir ${exp_prefix}

CUDA_VISIBLE_DEVICES=0 \
python run_ranker.py \
    --do_train \
    --do_eval \
    --model_name_or_path roberta-base \
    --output_dir "${exp_prefix}output" \
    --logging_dir "${exp_prefix}tblogging" \
    --overwrite_output_dir \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --weight_decay 0.1 \
    --evaluate_during_training \
    --eval_steps 50000 \
    --save_steps 50000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 2>&1 | tee "${exp_prefix}log.txt"