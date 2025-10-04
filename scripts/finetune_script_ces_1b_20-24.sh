export PYTHONPATH=/home/ruomeng/gae/src:$PYTHONPATH

python /home/ruomeng/gae/scripts/run_finetune.py \
    --root_data_dir="/home/ruomeng/gae/dataset"\
    --dataset="ces" \
    --split='20-24' \
    --model_name='Llama-3.2-1B'\
    --save_dir="/home/ruomeng/gae/logs" \
    --device="cuda:7" \
    --mode="neighbor_hist" \
    --wandb 

# accelerate launch /home/ruomeng/gae/scripts/run_finetune.py \
#   --root_data_dir="/home/ruomeng/gae/dataset"\
#     --dataset="ces" \
#     --split='20' \
#     --model_name='Llama-3.1-8B'\
#     --save_dir="/home/ruomeng/gae/logs" 
#     # --device="cuda:1" \