export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH

accelerate launch --config_file /home/ruomeng/.cache/huggingface/accelerate/default_config.yaml \
    /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces_promaxmax" \
    --split='24' \
    --model_name='Llama-3.2-1B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --neighbor="1" \
    --mode="all" \
    --wandb

accelerate launch --config_file /home/ruomeng/.cache/huggingface/accelerate/default_config.yaml \
    /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces_promaxmax" \
    --split='24' \
    --model_name='Llama-3.2-1B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --neighbor="0" \
    --mode="all" \
    --wandb
