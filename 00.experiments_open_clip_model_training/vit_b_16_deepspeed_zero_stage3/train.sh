deepspeed \
    --include localhost:0 \
    --master_addr 127.0.1.1 \
    --master_port=10001 \
    ../../tools/train_deepspeed.py \
    --work-dir ./