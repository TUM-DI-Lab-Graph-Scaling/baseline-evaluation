# Baseline: Single Machine Multi-GPU Minibatch Node Classification

* Deep Graph Library (DGL) paper link: https://arxiv.org/abs/1909.01315
* GraphSAGE paper link: https://arxiv.org/pdf/1706.02216.pdf
* Graph Attention Networks (GAT) paper link: https://arxiv.org/pdf/1710.10903.pdf

## How to run

Run with following (with preconfigured arguments):

```shell
bash run_baseline_dgl.sh
```

Or the following with customizable parameters:
```shell
python train.py --dataset ogbn-products --model GraphSAGE --gpus 4
```
```shell
python train.py --dataset ogbn-products --model GAT --gpus 4
```
```shell
python train.py --dataset ogbn-papers100M --model GraphSAGE --gpus 4
```
```shell
python train.py --dataset ogbn-papers100M --model GAT --gpus 4
```

