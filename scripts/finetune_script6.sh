export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH
python /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces" \
    --split='22-24' \
    --model_name='Llama-3.1-8B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --device="cuda:6" \
    --neighbor="1" \
    --wandb 

python /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces" \
    --split='22-24' \
    --model_name='Llama-3.1-8B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --device="cuda:6" \
    --neighbor="0" \
    --wandb 