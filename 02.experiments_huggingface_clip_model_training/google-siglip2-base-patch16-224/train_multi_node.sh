# sudo apt install net-tools
# ifconfig
# eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
#         inet 172.17.0.3  netmask 255.255.0.0  broadcast 172.17.255.255
#         ether 02:42:ac:11:00:03  txqueuelen 0  (Ethernet)
#         RX packets 752870  bytes 122681763 (122.6 MB)
#         RX errors 0  dropped 0  overruns 0  frame 0
#         TX packets 890936  bytes 137830625 (137.8 MB)
#         TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

# NCCL_SOCKET_IFNAME 值是 eth0
# master_addr 值是 inet 172.17.0.3

NCCL_SOCKET_IFNAME=eth0 CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr 172.17.0.5 \
    --master_port 10000 \
    ../../tools/train_huggingface_clip_model.py \
    --work-dir ./