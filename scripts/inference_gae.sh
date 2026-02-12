
export PYTHONPATH=/home/ruomeng/gae_graph/src:$PYTHONPATH


cuda=0,1,2,3
T=4
q_num=5
dataset=ces
model_name=Llama-3.1-8B
llm_batch_size=128
runs=/home/ruomeng/gae_graph/scripts/runs/${dataset}.csv
infer_data=/home/ruomeng/gae_graph/dataset/ces/raw/24/question_Midwest_test.csv
checkpoint=/home/ruomeng/gae_graph/logs/ces/neighbor_0/Llama-3.1-8B/20260104_044722
log_path_hybrid=/home/ruomeng/gae_graph/results/${dataset}_sub/${model_name}-finetuned-gae
gnn_config_path=/home/ruomeng/gae_graph/src/gnn/config_${dataset}_sub.yaml
gnn_batch_size=8192



for node_selection_prec in 0.1 0.3 0.5; do
  for node_selection in cluster; do
    for query_selection in info_gain; do
      for impute_thres in 0.0; do
        CUDA_VISIBLE_DEVICES=$cuda python /home/ruomeng/gae_graph/scripts/run_inference_gae.py \
          --llm_batch_size $llm_batch_size  --T $T\
          --dataset $dataset --infer_data $infer_data --runs $runs\
          --llm_checkpoint $checkpoint --log_path $log_path_hybrid \
          --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
          --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --impute_thres $impute_thres --imputation
      done
    done
  done
done

for node_selection_prec in 0.0; do
  for node_selection in random; do
    for query_selection in info_gain; do
      for impute_thres in 0.0; do
        CUDA_VISIBLE_DEVICES=$cuda python /home/ruomeng/gae_graph/scripts/run_inference_gae.py \
          --llm_batch_size $llm_batch_size  --T $T\
          --dataset $dataset --infer_data $infer_data --runs $runs\
          --llm_checkpoint $checkpoint --log_path $log_path_hybrid \
          --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
          --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --impute_thres $impute_thres --imputation
      done
    done
  done
done


