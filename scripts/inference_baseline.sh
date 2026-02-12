
export PYTHONPATH=/home/ruomeng/gae_graph/src:$PYTHONPATH

# checkpoint=/home/ruomeng/gae_graph/logs/ces/24/neighbor_0/Llama-3.2-1B/20251120_163157
# checkpoint=/home/ruomeng/gae_graph/logs/ces/neighbor_0/Llama-3.1-8B/20260104_044722
# checkpoint=/home/ruomeng/gae_graph/logs/opinionQA/neighbor_0/Llama-3.2-1B/20251128_163151
# checkpoint=/home/ruomeng/gae_graph/logs/opinionQA/neighbor_0/Llama-3.1-8B/20260104_113916

########################################################
cuda=0,1,2,3
T=4
q_num=5
dataset=ces
llm_batch_size=512
runs=/home/ruomeng/gae_graph/scripts/runs/${dataset}.csv
gnn_config_path=/home/ruomeng/gae_graph/src/gnn/config_${dataset}.yaml
gnn_batch_size=8192
infer_data=/home/ruomeng/gae_graph/dataset/ces/raw/24/question_West_test.csv
impute_thres=0.0
model_name=Llama-3.2-1B



## Meta Model + Greedy query selection + Random node selection
log_path_hybrid=/home/ruomeng/gae_graph/results/${dataset}/${model_name}-meta-randomQ
checkpoint=/home/ruomeng/gae_graph/logs/ces/24/neighbor_0/Llama-3.2-1B/20251120_163157
# checkpoint=/home/ruomeng/gae_graph/logs/ces/neighbor_0/Llama-3.1-8B/20260104_044722
for node_selection_prec in 0.1 0.3 0.5; do
  for node_selection in random; do
    for query_selection in random; do
        CUDA_VISIBLE_DEVICES=$cuda python /home/ruomeng/gae_graph/scripts/run_inference_baselines.py \
          --llm_batch_size $llm_batch_size  --T $T\
          --dataset $dataset --infer_data $infer_data --runs $runs\
          --llm_checkpoint $checkpoint --log_path $log_path_hybrid \
          --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
          --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --impute_thres $impute_thres
    done
  done
done


## Meta Model + Greedy query selection + Random node selection
log_path_hybrid=/home/ruomeng/gae_graph/results/${dataset}/${model_name}-meta-greedyQ
checkpoint=/home/ruomeng/gae_graph/logs/ces/24/neighbor_0/Llama-3.2-1B/20251120_163157
# checkpoint=/home/ruomeng/gae_graph/logs/ces/neighbor_0/Llama-3.1-8B/20260104_044722
for node_selection_prec in 0.1 0.3 0.5; do
  for node_selection in random; do
    for query_selection in info_gain; do
        CUDA_VISIBLE_DEVICES=$cuda python /home/ruomeng/gae_graph/scripts/run_inference_baselines.py \
          --llm_batch_size $llm_batch_size  --T $T\
          --dataset $dataset --infer_data $infer_data --runs $runs\
          --llm_checkpoint $checkpoint --log_path $log_path_hybrid \
          --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
          --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --impute_thres $impute_thres
    done
  done
done


## Meta Model + Greedy query selection + Random node selection
log_path_hybrid=/home/ruomeng/gae_graph/results/${dataset}/${model_name}-meta-greedyQ-imputation
checkpoint=/home/ruomeng/gae_graph/logs/ces/24/neighbor_0/Llama-3.2-1B/20251120_163157
# checkpoint=/home/ruomeng/gae_graph/logs/ces/neighbor_0/Llama-3.1-8B/20260104_044722
for node_selection_prec in 0.1 0.3 0.5; do
  for node_selection in random; do
    for query_selection in info_gain; do
        CUDA_VISIBLE_DEVICES=$cuda python /home/ruomeng/gae_graph/scripts/run_inference_baselines.py \
          --llm_batch_size $llm_batch_size  --T $T\
          --dataset $dataset --infer_data $infer_data --runs $runs\
          --llm_checkpoint $checkpoint --log_path $log_path_hybrid \
          --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
          --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --impute_thres $impute_thres
    done
  done
done