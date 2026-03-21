import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import re
import time

import torch
import deepspeed
from torch.utils.data import DataLoader

from tools.scripts import train_clip_model_deepspeed
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         build_optimizer, Scheduler)


def build_deepspeed_config(config):
    """Build DeepSpeed config dict from training config."""
    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size // config.gpus_num,
        "gradient_accumulation_steps": config.accumulation_steps,
        # never print by deepspeed
        "steps_per_print": 2**31,
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": config.deepspeed_zero_stage,
        },
    }

    # Gradient clipping
    if hasattr(config, 'clip_max_norm') and config.clip_max_norm > 0:
        ds_config["gradient_clipping"] = config.clip_max_norm
    else:
        ds_config["gradient_clipping"] = 0.0

    # Mixed precision
    if config.use_amp:
        if config.amp_type == torch.float16:
            ds_config["fp16"] = {
                "enabled": True,
                # 0代表启用dynamic loss scaling
                "loss_scale": 0,
                # dynamic loss scaling的初始值为65536,2的16次方
                "initial_scale_power": 16,
                # 连续1000个step没有出现overflow的情况下loss scale会翻倍(尝试更激进的缩放以获得更好的梯度精度),1000是DeepSpeed默认值
                "loss_scale_window": 1000,
                # 连续发生2次overflow之后才真正将loss scale减半
                "hysteresis": 2,
                # loss scale下限为1
                "min_loss_scale": 1,
            }
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "float16",
            }
        elif config.amp_type == torch.bfloat16:
            ds_config["bf16"] = {
                "enabled": True,
            }
            ds_config["torch_autocast"] = {
                "enabled": True,
                "dtype": "bfloat16",
            }
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}

    # ZeRO-Offload
    if hasattr(config, 'deepspeed_offload') and config.deepspeed_offload:
        if config.deepspeed_zero_stage >= 2:
            # 将优化器状态(如Adam的一阶矩m和二阶矩v)卸载到CPU内存,仅在ZeRO Stage≥2时有意义
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }
        if config.deepspeed_zero_stage == 3:
            # 将模型参数本身卸载到CPU内存,仅在ZeRO Stage=3时可用
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

    # ZeRO Stage 3 specific
    if config.deepspeed_zero_stage == 3:
        # ZeRO-3下每个rank只持有1/N参数分片,直接state_dict()只能拿到本rank的模型分片参数。开启此选项后,调用model_engine.save_checkpoint()时DeepSpeed 会自动执行all-gather将完整的16-bit权重收集到一起保存
        ds_config["zero_optimization"][
            "stage3_gather_16bit_weights_on_model_save"] = True

    return ds_config


def get_model_state_dict(model_engine, config):
    """
    Get full model state dict for saving.
    For ZeRO-3, all ranks must call this function (GatheredParameters is
    collective), but only rank 0 returns a non-None dict.
    """
    if config.use_compile:
        module = model_engine.module._orig_mod
    else:
        module = model_engine.module

    if config.deepspeed_zero_stage == 3:
        state_dict = {}
        for name, param in module.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if config.local_rank == 0:
                    state_dict[name] = param.data.cpu().clone()
        for name, buf in module.named_buffers():
            if config.local_rank == 0:
                state_dict[name] = buf.cpu().clone()
        return state_dict if config.local_rank == 0 else None
    else:
        return module.state_dict()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Clip Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')
    args, _ = parser.parse_known_args()

    return args


def main():
    assert torch.cuda.is_available(), 'need gpu to train network!'
    torch.cuda.empty_cache()

    args = parse_args()
    sys.path.append(args.work_dir)
    from train_config import config
    log_dir = os.path.join(args.work_dir, 'log')
    checkpoint_dir = os.path.join(args.work_dir, 'checkpoints')
    config.checkpoint_dir = checkpoint_dir
    config.gpus_type = torch.cuda.get_device_name()
    config.gpus_num = torch.cuda.device_count()

    if config.deepspeed_zero_stage == 3:
        resume_model = os.path.join(
            checkpoint_dir, 'zero_pp_rank_0_mp_rank_00_model_states.pt')
    else:
        resume_model = os.path.join(checkpoint_dir,
                                    'mp_rank_00_model_states.pt')

    set_seed(config.seed)

    local_rank = int(os.environ['LOCAL_RANK'])
    config.local_rank = local_rank
    # start init process
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend='nccl')

    # 获取total_rank
    total_rank = torch.distributed.get_rank()
    config.total_rank = total_rank

    config.group = torch.distributed.new_group(list(range(config.gpus_num)))

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    torch.distributed.barrier(device_ids=[local_rank])

    logger = get_logger('train', log_dir)

    batch_size, num_workers = config.batch_size, config.num_workers
    assert config.batch_size % config.gpus_num == 0, 'config.batch_size is not divisible by config.gpus_num!'
    assert config.num_workers % config.gpus_num == 0, 'config.num_workers is not divisible by config.gpus_num!'
    batch_size = int(config.batch_size // config.gpus_num)
    num_workers = int(config.num_workers // config.gpus_num)

    init_fn = functools.partial(worker_seed_init_fn,
                                num_workers=num_workers,
                                local_rank=local_rank,
                                seed=config.seed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        config.train_dataset, shuffle=True)
    train_loader = DataLoader(config.train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=num_workers,
                              collate_fn=config.train_collater,
                              sampler=train_sampler,
                              worker_init_fn=init_fn)

    for key, value in config.__dict__.items():
        if not key.startswith('__'):
            if key not in ['model']:
                log_info = f'{key}: {value}'
                logger.info(log_info) if local_rank == 0 else None

    model = config.model.cuda()
    train_criterion = config.train_criterion.cuda()

    # parameters needs to be updated by the optimizer
    # buffers doesn't needs to be updated by the optimizer
    log_info = f'--------------------parameters--------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, param in model.named_parameters():
        log_info = f'name: {name}, grad: {param.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    log_info = f'--------------------buffers--------------------'
    logger.info(log_info) if local_rank == 0 else None
    for name, buffer in model.named_buffers():
        log_info = f'name: {name}, grad: {buffer.requires_grad}'
        logger.info(log_info) if local_rank == 0 else None

    optimizer, model_layer_weight_decay_list = build_optimizer(config, model)

    log_info = f'-------------layers weight decay---------------'
    logger.info(log_info) if local_rank == 0 else None
    for per_layer_list in model_layer_weight_decay_list:
        layer_name_list, layer_lr, layer_weight_decay = per_layer_list[
            'name'], per_layer_list['lr'], per_layer_list['weight_decay']

        lr_scale = 'not setting!'
        if 'lr_scale' in per_layer_list.keys():
            lr_scale = per_layer_list['lr_scale']

        for name in layer_name_list:
            log_info = f'name: {name}, lr: {layer_lr}, weight_decay: {layer_weight_decay}, lr_scale: {lr_scale}'
            logger.info(log_info) if local_rank == 0 else None

    # Build scheduler with original optimizer before DeepSpeed wrapping,
    # consistent with the original code where scheduler is built before DDP.
    # DeepSpeed's optimizer wrapper internally delegates to this same optimizer
    # object, so LR changes made by the scheduler are reflected in training.
    scheduler = Scheduler(config, optimizer)

    # Check torch compile support
    config.compile_support = False
    log_info = f'using torch version:{torch.__version__}'
    logger.info(log_info) if local_rank == 0 else None
    if re.match(r'2.\d*.\d*', torch.__version__):
        config.compile_support = True
        log_info = f'this torch version support torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    elif re.match(r'1.\d*.\d*', torch.__version__):
        log_info = f'this torch version unsupport torch.compile function.'
        logger.info(log_info) if local_rank == 0 else None
    else:
        log_info = f'unsupport torch version:{torch.__version__}'
        logger.info(log_info) if local_rank == 0 else None
        return

    config.use_compile = (config.compile_support and config.use_compile)

    if config.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if config.use_compile:
        # _orig_mod
        model = torch.compile(model, **config.compile_params)

    # Build DeepSpeed config and initialize engine
    ds_config = build_deepspeed_config(config)
    log_info = f'DeepSpeed config: {ds_config}'
    logger.info(log_info) if local_rank == 0 else None

    # Pass optimizer to DeepSpeed but do NOT reassign the optimizer variable.
    # The scheduler holds a reference to the original optimizer object, and
    # DeepSpeed's wrapper internally delegates to it, so LR adjustments by
    # the scheduler are correctly picked up during model_engine.step().
    model_engine, _, _, _ = deepspeed.initialize(model=model,
                                                 optimizer=optimizer,
                                                 config=ds_config)

    # deepspeed==0.18.8
    if config.accumulation_steps > 1:
        model_engine._scale_wrt_gas = False

    start_epoch, train_time = 1, 0
    best_loss, train_loss = 1e9, 0
    # Resume from DeepSpeed checkpoint (tag="" saves directly in checkpoint_dir)
    if os.path.exists(resume_model):
        _, client_state = model_engine.load_checkpoint(checkpoint_dir, tag="")
        if client_state is not None:
            saved_epoch = client_state['epoch']
            start_epoch += saved_epoch
            used_time = client_state['time']
            train_time += used_time

            best_loss = client_state['best_loss']
            train_loss = client_state['train_loss']
            scheduler.load_state_dict(client_state['scheduler_state_dict'])

            log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch:0>3d}, used_time: {used_time:.3f} hours, best_loss: {best_loss:.4f}, lr: {scheduler.current_lr:.6f}'
            logger.info(log_info) if local_rank == 0 else None

    for epoch in range(start_epoch, config.epochs + 1):
        per_epoch_start_time = time.time()

        log_info = f'epoch {epoch:0>3d} lr: {scheduler.current_lr:.6f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_sampler.set_epoch(epoch)
        train_loss = train_clip_model_deepspeed(train_loader, model_engine,
                                                train_criterion, optimizer,
                                                scheduler, epoch, logger,
                                                config)
        log_info = f'train: epoch {epoch:0>3d}, train_loss: {train_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

        torch.cuda.empty_cache()

        train_time += (time.time() - per_epoch_start_time) / 3600

        # train_loss is consistent across all ranks (all_reduced in
        # train_clip_model), so is_best is identical on every rank.
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss

        if epoch % config.save_interval == 0:
            if config.deepspeed_zero_stage == 3:
                # ZeRO-3: all ranks must participate in GatheredParameters
                save_model = get_model_state_dict(model_engine, config)
                if local_rank == 0 and save_model is not None:
                    torch.save(
                        save_model,
                        os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))
            else:
                # ZeRO-0/1/2: only rank 0 needs to call state_dict
                if local_rank == 0:
                    save_model = get_model_state_dict(model_engine, config)
                    torch.save(
                        save_model,
                        os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))

        if is_best:
            if config.deepspeed_zero_stage == 3:
                save_best_model = get_model_state_dict(model_engine, config)
                if local_rank == 0 and save_best_model is not None:
                    torch.save(save_best_model,
                               os.path.join(checkpoint_dir, 'best.pth'))
            else:
                if local_rank == 0:
                    save_best_model = get_model_state_dict(
                        model_engine, config)
                    torch.save(save_best_model,
                               os.path.join(checkpoint_dir, 'best.pth'))

        # Save DeepSpeed checkpoint for resume (all ranks participate)
        client_state = {
            'epoch': epoch,
            'time': train_time,
            'best_loss': best_loss,
            'train_loss': train_loss,
            'lr': scheduler.current_lr,
            'scheduler_state_dict': scheduler.state_dict(),
        }
        model_engine.save_checkpoint(checkpoint_dir,
                                     tag="",
                                     client_state=client_state,
                                     save_latest=False)

        log_info = f'until epoch: {epoch:0>3d}, best_loss: {best_loss:.4f}'
        logger.info(log_info) if local_rank == 0 else None

    if local_rank == 0:
        if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
            os.rename(
                os.path.join(checkpoint_dir, 'best.pth'),
                os.path.join(checkpoint_dir,
                             f'{config.network}-loss{best_loss:.3f}.pth'))

    log_info = f'train done. model: {config.network}, train time: {train_time:.3f} hours, best_loss: {best_loss:.4f}'
    logger.info(log_info) if local_rank == 0 else None

    torch.distributed.destroy_process_group()

    return


if __name__ == '__main__':
    main()
