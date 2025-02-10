#!/bin/bash

# Environment setup for H100
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# H100-specific optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NET_GDR_LEVEL=3
export NCCL_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=^lo,docker0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Optional: for maximum performance
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=4 \
        train.py \
        --cuda \
        --data ../data/estonian/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 16 \
        --d_head 64 \
        --d_inner 4096 \
        --dropout 0.2 \
        --dropatt 0.2 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 8000 \
        --max_step 2000000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 256 \
        --batch_chunk 4 \
        --multi_gpu \
        --clip 0.25 \
        --use_tf32 \
        --use_cudnn_benchmark \
        --matmul_precision highest \
        --log-interval 50 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/estonian/ \
        --dataset wt103 \
        --tgt_len 128 \
        --mem_len 1600 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argument 1'
fi 