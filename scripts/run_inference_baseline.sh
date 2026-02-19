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
checkpoint=checkpoints/meta_train/${dataset}/${model_name}/your_checkpoint_path # path to the checkpoint
runs=scripts/runs/${dataset}.csv # path to the runs file
########################################################
llm_batch_size=128 # batch size for LLM inference
infer_data=dataset/${dataset}/data/question_${region}_test.csv # path to the test dat
########################################################


## Meta Model + Random query selection + Random node selection
query_selection=random # query selection strategy
node_selection=random # node selection strategy
log_path=results/${dataset}/meta_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_baseline.py --cuda $cuda \
  --llm_batch_size $llm_batch_size  --T $T\
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id\
  --llm_checkpoint $checkpoint --log_path $log_path \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec


## Meta Model + Greedy query selection + Random node selection
query_selection=info_gain
node_selection=random
log_path=results/${dataset}/meta_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_baseline.py --cuda $cuda \
  --llm_batch_size $llm_batch_size  --T $T\
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id\
  --llm_checkpoint $checkpoint --log_path $log_path \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec


## Meta Model + Greedy query selection + Random node selection + Imputation
query_selection=info_gain
node_selection=random
log_path=results/${dataset}/meta_${model_name}-Q-${query_selection}-N-${node_selection}
CUDA_VISIBLE_DEVICES=$cuda python scripts/run_inference_baseline.py --cuda $cuda \
  --llm_batch_size $llm_batch_size  --T $T\
  --dataset $dataset --infer_data $infer_data --runs $runs --runs_id $runs_id\
  --llm_checkpoint $checkpoint --log_path $log_path \
  --query_selection $query_selection --node_selection $node_selection --node_selection_prec $node_selection_prec --imputation