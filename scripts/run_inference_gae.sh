export PYTHONPATH=src:$PYTHONPATH


########################################################
cuda=4,5,6,7
T=4 # number of rounds of query selection
dataset=ces # dataset name
model_name=Llama-3.2-1B # model name

runs_id=0 # runs id (we sampled 10 different runs for each setting to get the average results)
node_selection_prec=0.5 # fraction of nodes to select for node selection
########################################################
region=West # region 
checkpoint=checkpoints/meta_train/${dataset}/${model_name}/20251120_163157 # path to the checkpoint
runs=scripts/runs/${dataset}.csv # path to the runs file
########################################################
llm_batch_size=128 # batch size for LLM inference
gnn_config_path=scripts/args_gnn/config_${dataset}.yaml
gnn_batch_size=2048 # batch size for GNN inference
infer_data=dataset/${dataset}/data/question_${region}_test.csv # path to the test dat
########################################################
# Multi-step planning parameters
mcts_n_iter=1 # number of iterations of the MCTS
mcts_top_k=2 # number of top nodes to consider for the MCTS
########################################################

## Group Adaptive Elicitation Framework
## Meta Model + Greedy query selection + Group relational node selection + Imputation
query_selection=info_gain # query selection strategy
node_selection=relational # node selection strategy
log_path=results/${dataset}/gae_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_gae.py --cuda $cuda \
  --llm_batch_size $llm_batch_size  --T $T\
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id\
  --llm_checkpoint $checkpoint --log_path $log_path \
  --gnn_config_path $gnn_config_path --gnn_batch_size $gnn_batch_size \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec \
  --mcts_n_iter $mcts_n_iter --mcts_top_k $mcts_top_k \
  --imputation 
