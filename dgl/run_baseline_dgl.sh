# Define CUDA installation path
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export CUDADIR=/usr/local/cuda-11.3
export CUDA_HOME=/usr/local/cuda-11.3

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=3,4,5,6

# Launch trainings script
python train.py --dataset ogbn-products --model GraphSAGE --gpus 4