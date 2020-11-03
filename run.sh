#/bin/sh!
exp_id=$1

exp_prefix="exps/${exp_id}/"

mkdir ${exp_prefix}
cp run.sh "${exp_prefix}/run.sh"

CUDA_VISIBLE_DEVICES=1,2,3 \
python -u run_ranker.py \
    --do_train \
    --do_eval \
    --disable_tqdm \
    --model_name_or_path roberta-base \
    --output_dir "${exp_prefix}output" \
    --logging_dir "${exp_prefix}tblogging" \
    --overwrite_output_dir \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --evaluation_strategy "steps" \
    --label_names "sp_labels" "ct_labels" "doc_masks" \
    --weight_decay 0.1 \
    --logging_steps 500 \
    --gradient_accumulation_steps 4 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --warmup_steps 1000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 2>&1 | tee "${exp_prefix}log.txt"
