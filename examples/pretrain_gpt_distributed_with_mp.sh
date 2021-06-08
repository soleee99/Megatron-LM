#! /bin/bash

# Runs the "345M" parameter model

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_DIR=/cmsdata/hdd2/the-pile/data
CURR_DIR=$(pwd)
echo $CURR_DIR
DATA_PATH="1 ${DATA_DIR}/00_text_document 1 ${DATA_DIR}/01_text_document"
CHECKPOINT_PATH=/cmsdata/hdd2/the-pile/model/testing2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --eval-path /cmsdata/hdd1/the-pile/val_text_document \
       --test-path /cmsdata/hdd1/the-pile/test_text_document \
       --vocab-file $CURR_DIR/downloaded/vocab.json \
       --merge-file $CURR_DIR/downloaded/merges.txt \
       --data-impl mmap \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 5 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --no-split
