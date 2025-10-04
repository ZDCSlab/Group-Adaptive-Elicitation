export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH

accelerate launch --main_process_port 29600 --config_file /home/ruomeng/.cache/huggingface/accelerate/default_config2.yaml \
    /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces_golden" \
    --split='22-24' \
    --model_name='Llama-3.2-1B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --neighbor="0" \
    --wandb 