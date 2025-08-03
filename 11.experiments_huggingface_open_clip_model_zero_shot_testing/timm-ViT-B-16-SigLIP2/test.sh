CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr 127.0.1.0 \
    --master_port 10000 \
    ../../tools/test_huggingface_open_clip_model_zero_shot_classify.py \
    --work-dir ./