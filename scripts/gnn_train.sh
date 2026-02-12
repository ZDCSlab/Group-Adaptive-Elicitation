export PYTHONPATH=/home/ruomeng/gae_graph/src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python /home/ruomeng/gae_graph/src/gnn/train.py --config /home/ruomeng/gae_graph/src/gnn/config_ces.yaml

# CUDA_VISIBLE_DEVICES=1 python /home/ruomeng/gae_graph/src/gnn/train.py --config /home/ruomeng/gae_graph/src/gnn/config_opinionQA.yaml
