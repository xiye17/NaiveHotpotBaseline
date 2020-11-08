#/bin/sh!
export SQUAD_DIR=outputs
ACTION=${1:-none}

if [ "$ACTION" = "train" ]; then
    exp_id=$2

    exp_prefix="exps/hpqa_${exp_id}/"

    mkdir ${exp_prefix}
    cp train_qa.sh "${exp_prefix}/train_qa.sh"

    CUDA_VISIBLE_DEVICES=1,2,3 \
    python -u run_qa.py \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --do_train \
        --do_eval \
        --disable_tqdm \
        --train_file $SQUAD_DIR/train_hpqa.json \
        --predict_file $SQUAD_DIR/dev_hpqa.json \
        --learning_rate 3e-5 \
        --weight_decay 0.1 \
        --evaluate_during_training \
        --num_train_epochs 4 \
        --overwrite_output_dir \
        --max_seq_length 512 \
        --logging_steps 500 \
        --eval_steps 2000 \
        --save_steps 2000 \
        --warmup_steps 1000 \
        --output_dir "${exp_prefix}output" \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 8 2>&1 | tee "${exp_prefix}log.txt"
elif [ "$ACTION" = "eval" ]; then
    CUDA_VISIBLE_DEVICES=1,2,3 \
    python -u run_qa.py \
        --model_type roberta \
        --model_name_or_path checkpoints/hpqa_roberta-base \
        --do_eval \
        --predict_file $SQUAD_DIR/gtsp-dev_hpqa.json \
        --overwrite_output_dir \
        --max_seq_length 512 \
        --output_dir "predictions/qa" \
        --per_gpu_eval_batch_size 50
else
    echo "train or eval"
fi

## mini version
# CUDA_VISIBLE_DEVICES=0,1 \
# python -u run_qa.py \s
#     --model_type roberta \
#     --model_name_or_path roberta-base \
#     --do_train \
#     --do_eval \
#     --disable_tqdm \
#     --train_file $SQUAD_DIR/mini_train_hpqa.json \
#     --predict_file $SQUAD_DIR/mini_dev_hpqa.json \
#     --learning_rate 5e-5 \
#     --weight_decay 0.1 \
#     --evaluate_during_training \
#     --num_train_epochs 50.0 \
#     --overwrite_output_dir \
#     --max_seq_length 512 \
#     --logging_steps 2 \
#     --eval_steps 4 \
#     --save_steps 1000 \
#     --output_dir "${exp_prefix}output" \
#     --per_gpu_train_batch_size 8 \
#     --per_gpu_eval_batch_size 8 2>&1 | tee "${exp_prefix}log.txt"
