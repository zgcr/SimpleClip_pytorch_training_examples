import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.amp.autocast_mode import autocast

from SimpleClip.common import AverageMeter, AccMeter


def all_reduce_operation_in_group_for_variables(variables, operator, group):
    for i in range(len(variables)):
        if not torch.is_tensor(variables[i]):
            variables[i] = torch.tensor(variables[i]).cuda()
        torch.distributed.all_reduce(variables[i], op=operator, group=group)
        variables[i] = variables[i].item()

    return variables


def train_clip_model(train_loader, model, criterion, optimizer, scheduler,
                     epoch, logger, config):
    '''
    train clip model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    if config.accumulation_steps > 1:
        accumulation_images = []
        accumulation_texts = []
        accumulation_features = {
            'image_features': [],
            'text_features': [],
        }

    for data_idx, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        # captions必须是一个list, list中每个元素都是字符串
        captions = data['caption']
        tokens = config.tokenizer(captions)
        tokens = tokens.long().cuda()

        if config.accumulation_steps == 1:
            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(images, tokens)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)
            else:
                if config.use_siglip_loss:
                    outputs = model(images, tokens)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    logit_bias = outputs['logit_bias']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, logit_bias, config)
                else:
                    outputs = model(images, tokens)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, config)

            loss = 0.
            for key, value in loss_value.items():
                loss += value

            if config.use_amp:
                config.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(images, tokens)
                        else:
                            outputs = model(images, tokens)
                else:
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                    else:
                        outputs = model(images, tokens)

                image_features = outputs['image_features']
                text_features = outputs['text_features']

                accumulation_features['image_features'].append(image_features)
                accumulation_features['text_features'].append(text_features)

                accumulation_images.append(images)
                accumulation_texts.append(tokens)

            # 只有data_idx循环到被config.accumulation_steps整除时才不会continue,继续往下执行
            if (data_idx + 1) % config.accumulation_steps > 0:
                # continue后,跳到下一个data_idx重新开始取数据
                iter_index += 1
                continue

            assert len(accumulation_images) == len(
                accumulation_texts) == config.accumulation_steps

            assert len(accumulation_features['image_features']) == len(
                accumulation_features['text_features']
            ) == config.accumulation_steps

            optimizer.zero_grad()

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            for accumulation_idx in range(config.accumulation_steps):
                images = accumulation_images[accumulation_idx]
                tokens = accumulation_texts[accumulation_idx]
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(images, tokens)
                            logit_scale = outputs['logit_scale']
                            logit_bias = outputs['logit_bias']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   logit_bias, config)
                        else:
                            outputs = model(images, tokens)
                            logit_scale = outputs['logit_scale']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   config)
                else:
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(images, tokens)
                        logit_scale = outputs['logit_scale']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)

                # 各loss数量级除以accumulation_steps
                for key, value in loss_value.items():
                    loss_value[key] = value / config.accumulation_steps

                # 总loss是各loss的累加,因此其数量级也已经除以accumulation_steps
                loss = 0.
                for key, value in loss_value.items():
                    loss += value

                if config.use_amp:
                    config.scaler.scale(loss).backward()
                else:
                    loss.backward()

        if config.use_amp:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):
                config.scaler.unscale_(optimizer)

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)

            config.scaler.step(optimizer)
            config.scaler.update()
            optimizer.zero_grad()
        else:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # reset
        if config.accumulation_steps > 1:
            accumulation_images = []
            accumulation_texts = []
            accumulation_features = {
                'image_features': [],
                'text_features': [],
            }

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            if config.use_compile:
                model._orig_mod.module.logit_scale.clamp_(0, math.log(100))
            else:
                model.module.logit_scale.clamp_(0, math.log(100))

        for key, value in loss_value.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_value[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)
        losses.update(loss, images.size(0))

        scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if total_accumulation_iters % config.step_save_interval == 0:
                if local_rank == 0 and total_rank == 0:
                    if config.use_compile:
                        save_model = model._orig_mod.module.state_dict()
                    else:
                        save_model = model.module.state_dict()

                    torch.save(
                        save_model,
                        os.path.join(config.checkpoint_dir,
                                     f'step_{total_accumulation_iters}.pth'))

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_huggingface_open_clip_model(train_loader, model, criterion,
                                      optimizer, scheduler, epoch, logger,
                                      config):
    '''
    train clip model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    if config.accumulation_steps > 1:
        accumulation_images = []
        accumulation_texts = []
        accumulation_features = {
            'image_features': [],
            'text_features': [],
        }

    for data_idx, data in enumerate(train_loader):
        images = data['image']
        images = images.cuda()

        # captions必须是一个list, list中每个元素都是字符串
        captions = data['caption']
        tokens = config.tokenizer(captions)
        tokens = tokens.long().cuda()

        if config.accumulation_steps == 1:
            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(images, tokens)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)
            else:
                if config.use_siglip_loss:
                    outputs = model(images, tokens)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    logit_bias = outputs['logit_bias']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, logit_bias, config)
                else:
                    outputs = model(images, tokens)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, config)

            loss = 0.
            for key, value in loss_value.items():
                loss += value

            if config.use_amp:
                config.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(images, tokens)
                        else:
                            outputs = model(images, tokens)
                else:
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                    else:
                        outputs = model(images, tokens)

                image_features = outputs['image_features']
                text_features = outputs['text_features']

                accumulation_features['image_features'].append(image_features)
                accumulation_features['text_features'].append(text_features)

                accumulation_images.append(images)
                accumulation_texts.append(tokens)

            # 只有data_idx循环到被config.accumulation_steps整除时才不会continue,继续往下执行
            if (data_idx + 1) % config.accumulation_steps > 0:
                # continue后,跳到下一个data_idx重新开始取数据
                iter_index += 1
                continue

            assert len(accumulation_images) == len(
                accumulation_texts) == config.accumulation_steps

            assert len(accumulation_features['image_features']) == len(
                accumulation_features['text_features']
            ) == config.accumulation_steps

            optimizer.zero_grad()

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            for accumulation_idx in range(config.accumulation_steps):
                images = accumulation_images[accumulation_idx]
                tokens = accumulation_texts[accumulation_idx]
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(images, tokens)
                            logit_scale = outputs['logit_scale']
                            logit_bias = outputs['logit_bias']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   logit_bias, config)
                        else:
                            outputs = model(images, tokens)
                            logit_scale = outputs['logit_scale']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   config)
                else:
                    if config.use_siglip_loss:
                        outputs = model(images, tokens)
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(images, tokens)
                        logit_scale = outputs['logit_scale']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)

                # 各loss数量级除以accumulation_steps
                for key, value in loss_value.items():
                    loss_value[key] = value / config.accumulation_steps

                # 总loss是各loss的累加,因此其数量级也已经除以accumulation_steps
                loss = 0.
                for key, value in loss_value.items():
                    loss += value

                if config.use_amp:
                    config.scaler.scale(loss).backward()
                else:
                    loss.backward()

        if config.use_amp:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):
                config.scaler.unscale_(optimizer)

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)

            config.scaler.step(optimizer)
            config.scaler.update()
            optimizer.zero_grad()
        else:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # reset
        if config.accumulation_steps > 1:
            accumulation_images = []
            accumulation_texts = []
            accumulation_features = {
                'image_features': [],
                'text_features': [],
            }

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.module.model.logit_scale.clamp_(0, math.log(100))

        for key, value in loss_value.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_value[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)
        losses.update(loss, images.size(0))

        scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if total_accumulation_iters % config.step_save_interval == 0:
                if local_rank == 0 and total_rank == 0:
                    if config.use_compile:
                        save_model = model._orig_mod.module.state_dict()
                    else:
                        save_model = model.module.state_dict()

                    torch.save(
                        save_model,
                        os.path.join(config.checkpoint_dir,
                                     f'step_{total_accumulation_iters}.pth'))

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def train_huggingface_clip_model(train_loader, model, criterion, optimizer,
                                 scheduler, epoch, logger, config):
    '''
    train clip model for one epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    local_rank = config.local_rank
    if hasattr(config, 'total_rank'):
        total_rank = config.total_rank
    else:
        total_rank = 0

    log_info = f'use_amp: {config.use_amp}, amp_type: {config.amp_type}!'
    logger.info(log_info) if local_rank == 0 and total_rank == 0 else None

    iters = len(train_loader.dataset) // config.batch_size
    iter_index = 1
    assert config.accumulation_steps >= 1, 'illegal accumulation_steps!'

    if config.accumulation_steps > 1:
        accumulation_inputs = []
        accumulation_features = {
            'image_features': [],
            'text_features': [],
        }

    for data_idx, data in enumerate(train_loader):
        inputs = data['input']
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if config.accumulation_steps == 1:
            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    if config.use_siglip_loss:
                        outputs = model(inputs)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(inputs)
                        image_features = outputs['image_features']
                        text_features = outputs['text_features']
                        logit_scale = outputs['logit_scale']
                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)
            else:
                if config.use_siglip_loss:
                    outputs = model(inputs)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    logit_bias = outputs['logit_bias']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, logit_bias, config)
                else:
                    outputs = model(inputs)
                    image_features = outputs['image_features']
                    text_features = outputs['text_features']
                    logit_scale = outputs['logit_scale']
                    loss_value = criterion(image_features, text_features,
                                           logit_scale, config)

            loss = 0.
            for key, value in loss_value.items():
                loss += value

            if config.use_amp:
                config.scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(inputs)
                        else:
                            outputs = model(inputs)
                else:
                    if config.use_siglip_loss:
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs)

                image_features = outputs['image_features']
                text_features = outputs['text_features']

                accumulation_features['image_features'].append(image_features)
                accumulation_features['text_features'].append(text_features)

                accumulation_inputs.append(inputs)

            # 只有data_idx循环到被config.accumulation_steps整除时才不会continue,继续往下执行
            if (data_idx + 1) % config.accumulation_steps > 0:
                # continue后,跳到下一个data_idx重新开始取数据
                iter_index += 1
                continue

            assert len(accumulation_inputs) == config.accumulation_steps

            assert len(accumulation_features['image_features']) == len(
                accumulation_features['text_features']
            ) == config.accumulation_steps

            optimizer.zero_grad()

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            for accumulation_idx in range(config.accumulation_steps):
                inputs = accumulation_inputs[accumulation_idx]
                if config.use_amp:
                    with autocast(device_type="cuda", dtype=config.amp_type):
                        if config.use_siglip_loss:
                            outputs = model(inputs)
                            logit_scale = outputs['logit_scale']
                            logit_bias = outputs['logit_bias']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   logit_bias, config)
                        else:
                            outputs = model(inputs)
                            logit_scale = outputs['logit_scale']

                            accumulation_image_features = accumulation_features[
                                'image_features']
                            accumulation_text_features = accumulation_features[
                                'text_features']

                            image_features = torch.cat(
                                accumulation_image_features[0:accumulation_idx]
                                + [outputs['image_features']] +
                                accumulation_image_features[accumulation_idx +
                                                            1:])
                            text_features = torch.cat(
                                accumulation_text_features[0:accumulation_idx]
                                + [outputs['text_features']] +
                                accumulation_text_features[accumulation_idx +
                                                           1:])

                            loss_value = criterion(image_features,
                                                   text_features, logit_scale,
                                                   config)
                else:
                    if config.use_siglip_loss:
                        outputs = model(inputs)
                        logit_scale = outputs['logit_scale']
                        logit_bias = outputs['logit_bias']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, logit_bias, config)
                    else:
                        outputs = model(inputs)
                        logit_scale = outputs['logit_scale']

                        accumulation_image_features = accumulation_features[
                            'image_features']
                        accumulation_text_features = accumulation_features[
                            'text_features']

                        image_features = torch.cat(
                            accumulation_image_features[0:accumulation_idx] +
                            [outputs['image_features']] +
                            accumulation_image_features[accumulation_idx + 1:])
                        text_features = torch.cat(
                            accumulation_text_features[0:accumulation_idx] +
                            [outputs['text_features']] +
                            accumulation_text_features[accumulation_idx + 1:])

                        loss_value = criterion(image_features, text_features,
                                               logit_scale, config)

                # 各loss数量级除以accumulation_steps
                for key, value in loss_value.items():
                    loss_value[key] = value / config.accumulation_steps

                # 总loss是各loss的累加,因此其数量级也已经除以accumulation_steps
                loss = 0.
                for key, value in loss_value.items():
                    loss += value

                if config.use_amp:
                    config.scaler.scale(loss).backward()
                else:
                    loss.backward()

        if config.use_amp:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):
                config.scaler.unscale_(optimizer)

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)

            config.scaler.step(optimizer)
            config.scaler.update()
            optimizer.zero_grad()
        else:
            if (hasattr(config, 'clip_grad_value')
                    and config.clip_grad_value > 0) or (hasattr(
                        config, 'clip_max_norm') and config.clip_max_norm > 0):

                if hasattr(config, 'clip_grad_value'):
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    config.clip_grad_value)

                if hasattr(config, 'clip_max_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config.clip_max_norm)
            optimizer.step()
            optimizer.zero_grad()

        # reset
        if config.accumulation_steps > 1:
            accumulation_inputs = []
            accumulation_features = {
                'image_features': [],
                'text_features': [],
            }

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            model.module.model.logit_scale.clamp_(0, math.log(100))

        for key, value in loss_value.items():
            [value] = all_reduce_operation_in_group_for_variables(
                variables=[value],
                operator=torch.distributed.ReduceOp.SUM,
                group=config.group)
            loss_value[key] = value / float(config.gpus_num)

        [loss] = all_reduce_operation_in_group_for_variables(
            variables=[loss],
            operator=torch.distributed.ReduceOp.SUM,
            group=config.group)
        loss = loss / float(config.gpus_num)
        losses.update(loss, 1)

        scheduler.step(optimizer, iter_index / iters + (epoch - 1))

        accumulation_iter_index, accumulation_iters = int(
            iter_index // config.accumulation_steps), int(
                iters // config.accumulation_steps)
        if iter_index % int(
                config.print_interval * config.accumulation_steps) == 0:
            log_info = f'train: epoch {epoch:0>4d}, iter [{accumulation_iter_index:0>5d}, {accumulation_iters:0>5d}], lr: {scheduler.current_lr:.6f}, loss: {loss*config.accumulation_steps:.4f}, '
            for key, value in loss_value.items():
                log_info += f'{key}: {value*config.accumulation_steps:.4f}, '
            logger.info(
                log_info) if local_rank == 0 and total_rank == 0 else None

        total_accumulation_iters = accumulation_iters * (
            epoch - 1) + accumulation_iter_index
        if hasattr(config,
                   'use_step_save_interval') and config.use_step_save_interval:
            if total_accumulation_iters % config.step_save_interval == 0:
                if local_rank == 0 and total_rank == 0:
                    if config.use_compile:
                        save_model = model._orig_mod.module.state_dict()
                    else:
                        save_model = model.module.state_dict()

                    torch.save(
                        save_model,
                        os.path.join(config.checkpoint_dir,
                                     f'step_{total_accumulation_iters}.pth'))

        iter_index += 1

    avg_loss = losses.avg
    avg_loss = avg_loss * config.accumulation_steps

    return avg_loss


def test_huggingface_open_clip_model(test_loader, model, config):
    accs = AccMeter()

    # switch to evaluate mode
    model.eval()

    templates = config.val_dataset.templates
    class_name_list = config.val_dataset.sub_class_imagenet_classname_list
    template_num = len(templates)
    class_name_num = len(class_name_list)

    all_classes_templates_texts = [
        template(class_name) for class_name in class_name_list
        for template in templates
    ]
    all_tokens = config.tokenizer(all_classes_templates_texts)
    all_tokens = all_tokens.long().cuda()

    text_batch_size = 10
    text_features = []
    total_samples = all_tokens.shape[0]
    batch_nums = (total_samples + text_batch_size - 1) // text_batch_size

    with torch.no_grad():
        for idx in tqdm(range(batch_nums)):
            start_idx = idx * text_batch_size
            end_idx = min((idx + 1) * text_batch_size, total_samples)

            batch_tokens = all_tokens[start_idx:end_idx]

            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    batch_features = model.module.encode_text(batch_tokens)
            else:
                batch_features = model.module.encode_text(batch_tokens)

            text_features.append(batch_features)

    text_features = torch.cat(text_features, dim=0)

    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.reshape(class_name_num, template_num,
                                          -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            images, labels = data['image'], data['label']
            if model_on_cuda:
                images = images.cuda()

            labels = torch.tensor(labels).long().cuda()

            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    if config.is_siglip_model:
                        image_features = model.module.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)

                        logit_scale = model.module.model.logit_scale.exp()
                        logit_bias = model.module.model.logit_bias

                        preds = torch.sigmoid(
                            image_features @ text_features.T * logit_scale +
                            logit_bias)
                    else:
                        image_features = model.module.encode_image(images)
                        image_features = F.normalize(image_features, dim=-1)

                        logit_scale = model.module.model.logit_scale.exp()

                        preds = (logit_scale *
                                 image_features @ text_features.T).softmax(
                                     dim=-1)
            else:
                if config.is_siglip_model:
                    image_features = model.module.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)

                    logit_scale = model.module.model.logit_scale.exp()
                    logit_bias = model.module.model.logit_bias

                    preds = torch.sigmoid(image_features @ text_features.T *
                                          logit_scale + logit_bias)
                else:
                    image_features = model.module.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)

                    logit_scale = model.module.model.logit_scale.exp()

                    preds = (logit_scale *
                             image_features @ text_features.T).softmax(dim=-1)

            _, topk_indexes = torch.topk(preds,
                                         k=5,
                                         dim=1,
                                         largest=True,
                                         sorted=True)

            correct_mask = topk_indexes.eq(
                labels.unsqueeze(-1).expand_as(topk_indexes)).float()
            correct_mask = correct_mask.cpu().numpy()

            acc1_correct_num, acc5_correct_num, sample_num = correct_mask[:, :1].sum(
            ), correct_mask[:, :5].sum(), images.size(0)
            acc1_correct_num, acc5_correct_num, sample_num = float(
                acc1_correct_num), float(acc5_correct_num), float(sample_num)

            # please keep same variable on different gpus has same data type for all reduce operation
            [acc1_correct_num, acc5_correct_num,
             sample_num] = all_reduce_operation_in_group_for_variables(
                 variables=[acc1_correct_num, acc5_correct_num, sample_num],
                 operator=torch.distributed.ReduceOp.SUM,
                 group=config.group)

            accs.update(acc1_correct_num, acc5_correct_num, sample_num)

    # top1(%)、top5(%)
    accs.compute()
    acc1 = accs.acc1 * 100
    acc5 = accs.acc5 * 100

    return acc1, acc5


def test_huggingface_clip_model(test_loader, model, config):
    accs = AccMeter()

    # switch to evaluate mode
    model.eval()

    templates = config.val_dataset.templates
    class_name_list = config.val_dataset.sub_class_imagenet_classname_list
    template_num = len(templates)
    class_name_num = len(class_name_list)

    all_classes_templates_texts = [
        template(class_name) for class_name in class_name_list
        for template in templates
    ]

    text_inputs = config.val_processor(
        text=all_classes_templates_texts,
        return_tensors='pt',
        padding="max_length",
        max_length=config.val_collater.max_length)
    input_ids = text_inputs['input_ids']
    attention_mask = text_inputs.get('attention_mask', None)

    input_ids = input_ids.cuda()
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    text_batch_size = 10
    text_features = []
    total_samples = input_ids.shape[0]
    batch_nums = (total_samples + text_batch_size - 1) // text_batch_size

    with torch.no_grad():
        for idx in tqdm(range(batch_nums)):
            start_idx = idx * text_batch_size
            end_idx = min((idx + 1) * text_batch_size, total_samples)

            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[
                start_idx:end_idx] if attention_mask is not None else None

            batch_text_inputs = {
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask
            }

            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    batch_features = model.module.model.get_text_features(
                        **batch_text_inputs)
            else:
                batch_features = model.module.model.get_text_features(
                    **batch_text_inputs)

            text_features.append(batch_features)

    text_features = torch.cat(text_features, dim=0)

    text_features = F.normalize(text_features, dim=-1)
    text_features = text_features.reshape(class_name_num, template_num,
                                          -1).mean(dim=1)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    with torch.no_grad():
        model_on_cuda = next(model.parameters()).is_cuda
        for _, data in tqdm(enumerate(test_loader)):
            inputs, labels = data['input'], data['label']
            if model_on_cuda:
                labels = torch.tensor(labels).long().cuda()

            pixel_values = inputs['pixel_values'].cuda()

            if config.use_amp:
                with autocast(device_type="cuda", dtype=config.amp_type):
                    image_features = model.module.model.get_image_features(
                        pixel_values=pixel_values)
            else:
                image_features = model.module.model.get_image_features(
                    pixel_values=pixel_values)

            image_features = F.normalize(image_features, dim=-1)

            logit_scale = model.module.get_logit_scale()
            logit_bias = model.module.get_logit_bias()

            logits_per_image = logit_scale * image_features @ text_features.T
            if logit_bias is not None:
                logits_per_image += logit_bias

            if config.is_siglip_model:
                preds = torch.sigmoid(logits_per_image)
            else:
                preds = F.softmax(logits_per_image, dim=-1)

            _, topk_indexes = torch.topk(preds,
                                         k=5,
                                         dim=1,
                                         largest=True,
                                         sorted=True)

            correct_mask = topk_indexes.eq(
                labels.unsqueeze(-1).expand_as(topk_indexes)).float()
            correct_mask = correct_mask.cpu().numpy()

            acc1_correct_num, acc5_correct_num, sample_num = correct_mask[:, :1].sum(
            ), correct_mask[:, :5].sum(), len(labels)
            acc1_correct_num, acc5_correct_num, sample_num = float(
                acc1_correct_num), float(acc5_correct_num), float(sample_num)

            # please keep same variable on different gpus has same data type for all reduce operation
            [acc1_correct_num, acc5_correct_num,
             sample_num] = all_reduce_operation_in_group_for_variables(
                 variables=[acc1_correct_num, acc5_correct_num, sample_num],
                 operator=torch.distributed.ReduceOp.SUM,
                 group=config.group)

            accs.update(acc1_correct_num, acc5_correct_num, sample_num)

    # top1(%)、top5(%)
    accs.compute()
    acc1 = accs.acc1 * 100
    acc5 = accs.acc5 * 100

    return acc1, acc5
