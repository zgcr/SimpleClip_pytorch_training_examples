import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. compute_local_loss=True + gather_features_with_grad=True 所有特征有梯度
# 2. compute_local_loss=False + gather_features_with_grad=False 仅当前GPU特征有梯度
# 3. compute_local_loss=True + gather_features_with_grad=False 不允许这么用,text_features无梯度!
class ClipLoss(nn.Module):
    '''
    Learning Transferable Visual Models From Natural Language Supervision: https://arxiv.org/abs/2103.00020
    '''

    def __init__(self,
                 cache_labels=True,
                 compute_local_loss=False,
                 gather_features_with_grad=False):
        super(ClipLoss, self).__init__()
        self.cache_labels = cache_labels
        self.compute_local_loss = compute_local_loss
        self.gather_features_with_grad = gather_features_with_grad

        # cache state
        self.labels = {}
        self.pre_logit_nums = 0

    def gather_features(self, image_features, text_features, config):
        # We gather tensors from all gpus
        if self.gather_features_with_grad:
            # 保留特征梯度的特征聚合
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            # 无梯度的特征聚合（默认）
            gathered_image_features = [
                torch.zeros_like(image_features)
                for _ in range(config.gpus_num)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(config.gpus_num)
            ]
            # 通过 dist.all_gather收集其他GPU的特征(无梯度)
            torch.distributed.all_gather(gathered_image_features,
                                         image_features)
            torch.distributed.all_gather(gathered_text_features, text_features)

            if not self.compute_local_loss:
                # 全局损失模式下恢复当前GPU的梯度
                gathered_image_features[
                    config.total_rank] = image_features  # 回补梯度
                gathered_text_features[
                    config.total_rank] = text_features  # 回补梯度
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

        return all_image_features, all_text_features

    def get_logits(self, image_features, text_features, logit_scale, config):
        if config.gpus_num > 1:
            # 多GPU
            all_image_features, all_text_features = self.gather_features(
                image_features, text_features, config)

            if self.compute_local_loss:
                # 局部相似度计算: 仅计算当前GPU特征与全局特征的相似度
                # 局部到全局相似度矩阵 shape: [B, world_size*B]
                # 计算开销中, 显存占用中, 所有GPU的特征都有完整梯度, 适用于大规模分布式训练
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                # 全局相似度计算: 计算所有特征间的相似度
                # 全局相似度矩阵 shape: [world_size*B, world_size*B]
                # 计算开销高, 显存占用高, 仅当前GPU的特征有梯度, 适用于小batch_size
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # 单GPU
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def get_ground_truth(self, image_features, logit_nums, config):
        device = image_features.device

        # 缓存机制避免重复创建标签
        if self.pre_logit_nums != logit_nums or device not in self.labels:
            labels = torch.arange(logit_nums, device=device, dtype=torch.long)
            # 多GPU局部损失模式下的标签偏移
            if config.gpus_num > 1 and self.compute_local_loss:
                # 标签偏移
                labels = labels + logit_nums * config.total_rank
            if self.cache_labels:
                self.labels[device] = labels
                self.pre_logit_nums = logit_nums
        else:
            labels = self.labels[device]

        return labels

    def forward(self, image_features, text_features, logit_scale, config):
        # 1. 计算相似度矩阵
        # 可学习的温度系数logit_scale
        image_logits, text_logits = self.get_logits(image_features,
                                                    text_features, logit_scale,
                                                    config)

        # 2. 生成真实标签
        logit_nums = image_logits.shape[0]
        labels = self.get_ground_truth(image_features, logit_nums, config)

        # 3. 计算对称交叉熵损失
        contrastive_loss = (F.cross_entropy(image_logits, labels) +
                            F.cross_entropy(text_logits, labels)) / 2

        loss_dict = {
            'contrastive_loss': contrastive_loss,
        }

        return loss_dict


def neighbour_exchange(from_rank, to_rank, tensor, group):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(torch.distributed.isend,
                                      tensor,
                                      to_rank,
                                      group=group)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                      tensor_recv,
                                      from_rank,
                                      group=group)
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])

    for req in reqs:
        req.wait()

    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left,
                             tensor_to_right, group):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(torch.distributed.isend,
                                           tensor_to_left,
                                           left_rank,
                                           group=group)
    send_op_right = torch.distributed.P2POp(torch.distributed.isend,
                                            tensor_to_right,
                                            right_rank,
                                            group=group)
    recv_op_left = torch.distributed.P2POp(torch.distributed.irecv,
                                           tensor_from_left,
                                           left_rank,
                                           group=group)
    recv_op_right = torch.distributed.P2POp(torch.distributed.irecv,
                                            tensor_from_right,
                                            right_rank,
                                            group=group)
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left])

    for req in reqs:
        req.wait()

    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):

    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(
            ctx.to_rank, ctx.from_rank, ctx.group, grad_output), )


class NeighbourExchangeBidir(torch.autograd.Function):

    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left,
                tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank,
                                        right_rank,
                                        tensor_to_left,
                                        tensor_to_right,
                                        group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


def neighbour_exchange_bidir_with_grad(left_rank,
                                       right_rank,
                                       tensor_to_left,
                                       tensor_to_right,
                                       group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group,
                                        tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    ''' 
    Sigmoid Loss for Language Image Pre-Training (SigLIP): https://arxiv.org/abs/2303.15343
    '''

    def __init__(self, dist_type='bidir'):
        super(SigLipLoss, self).__init__()
        assert dist_type in ['bidir', 'shift', 'reduce', 'gather']

        self.dist_type = dist_type

    def get_ground_truth(self, device, dtype, logit_nums, negative_only=False):
        labels = -torch.ones(
            (logit_nums, logit_nums), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(logit_nums, device=device,
                                   dtype=dtype) + labels

        return labels

    def get_logits(self,
                   image_features,
                   text_features,
                   logit_scale,
                   logit_bias=None):
        logits = logit_scale * image_features @ text_features.T

        if logit_bias is not None:
            logits += logit_bias

        return logits

    def compute_sigmoid_loss(self,
                             image_features,
                             text_features,
                             logit_scale,
                             logit_bias=None,
                             negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale,
                                 logit_bias)
        labels = self.get_ground_truth(image_features.device,
                                       image_features.dtype,
                                       image_features.shape[0],
                                       negative_only=negative_only)
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]

        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias,
                config):
        loss = self.compute_sigmoid_loss(image_features,
                                         text_features,
                                         logit_scale,
                                         logit_bias,
                                         negative_only=False)

        if config.gpus_num > 1:
            if self.dist_type == 'bidir':
                right_rank = (config.total_rank + 1) % config.gpus_num
                left_rank = (config.total_rank - 1 +
                             config.gpus_num) % config.gpus_num
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(config.gpus_num - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank, right_rank, text_features_to_left,
                        text_features_to_right)
                    for f in text_features_recv:
                        loss += self.compute_sigmoid_loss(image_features,
                                                          f,
                                                          logit_scale,
                                                          logit_bias,
                                                          negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    loss += self.compute_sigmoid_loss(image_features,
                                                      text_features_recv,
                                                      logit_scale,
                                                      logit_bias,
                                                      negative_only=True)

            elif self.dist_type == 'shift':
                right_rank = (config.total_rank + 1) % config.gpus_num
                left_rank = (config.total_rank - 1 +
                             config.gpus_num) % config.gpus_num
                text_features_to_right = text_features
                for i in range(config.gpus_num - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    loss += self.compute_sigmoid_loss(image_features,
                                                      text_features_from_left,
                                                      logit_scale,
                                                      logit_bias,
                                                      negative_only=True)
                    text_features_to_right = text_features_from_left

            elif self.dist_type == 'reduce':
                for i in range(config.gpus_num):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (config.total_rank == i),
                        torch.distributed.ReduceOp.SUM)
                    loss += float(
                        i != config.total_rank) * self.compute_sigmoid_loss(
                            image_features,
                            text_from_other,
                            logit_scale,
                            logit_bias,
                            negative_only=True)

            elif self.dist_type == 'gather':
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(config.gpus_num):
                    loss += float(
                        i != config.total_rank) * self.compute_sigmoid_loss(
                            image_features,
                            all_text[i],
                            logit_scale,
                            logit_bias,
                            negative_only=True)

        loss_dict = {
            'contrastive_loss': loss,
        }

        return loss_dict


class DistillClipLoss(nn.Module):

    def __init__(self,
                 cache_labels=True,
                 compute_local_loss=True,
                 gather_features_with_grad=True):
        super(DistillClipLoss, self).__init__()
        self.cache_labels = cache_labels
        self.compute_local_loss = compute_local_loss
        self.gather_features_with_grad = gather_features_with_grad

        # cache state
        self.labels = {}
        self.pre_logit_nums = 0

    def gather_features(self, image_features, text_features, config):
        # We gather tensors from all gpus
        if self.gather_features_with_grad:
            # 保留特征梯度的特征聚合
            all_image_features = torch.cat(
                torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            # 无梯度的特征聚合（默认）
            gathered_image_features = [
                torch.zeros_like(image_features)
                for _ in range(config.gpus_num)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(config.gpus_num)
            ]
            # 通过 dist.all_gather收集其他GPU的特征(无梯度)
            torch.distributed.all_gather(gathered_image_features,
                                         image_features)
            torch.distributed.all_gather(gathered_text_features, text_features)

            if not self.compute_local_loss:
                # 全局损失模式下恢复当前GPU的梯度
                gathered_image_features[
                    config.total_rank] = image_features  # 回补梯度
                gathered_text_features[
                    config.total_rank] = text_features  # 回补梯度
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

        return all_image_features, all_text_features

    def get_logits(self, image_features, text_features, logit_scale, config):
        if config.gpus_num > 1:
            # 多GPU
            all_image_features, all_text_features = self.gather_features(
                image_features, text_features, config)

            if self.compute_local_loss:
                # 局部相似度计算: 仅计算当前GPU特征与全局特征的相似度
                # 局部到全局相似度矩阵 shape: [B, world_size*B]
                # 计算开销中, 显存占用中, 所有GPU的特征都有完整梯度, 适用于大规模分布式训练
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                # 全局相似度计算: 计算所有特征间的相似度
                # 全局相似度矩阵 shape: [world_size*B, world_size*B]
                # 计算开销高, 显存占用高, 仅当前GPU的特征有梯度, 适用于小batch_size
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # 单GPU
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def get_ground_truth(self, image_features, logit_nums, config):
        device = image_features.device

        # 缓存机制避免重复创建标签
        if self.pre_logit_nums != logit_nums or device not in self.labels:
            labels = torch.arange(logit_nums, device=device, dtype=torch.long)
            # 多GPU局部损失模式下的标签偏移
            if config.gpus_num > 1 and self.compute_local_loss:
                # 标签偏移
                labels = labels + logit_nums * config.total_rank
            if self.cache_labels:
                self.labels[device] = labels
                self.pre_logit_nums = logit_nums
        else:
            labels = self.labels[device]

        return labels

    def compute_distill_loss(self, teacher_logits, student_logits):
        distill_loss = -(teacher_logits.softmax(
            dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

        return distill_loss

    def forward(self, stu_image_features, stu_text_features, stu_logit_scale,
                tea_image_features, tea_text_features, tea_logit_scale,
                config):
        # 1. 计算学生模型相似度矩阵
        # 可学习的温度系数logit_scale
        stu_image_logits, stu_text_logits = self.get_logits(
            stu_image_features, stu_text_features, stu_logit_scale, config)

        # 2. 计算教师模型相似度矩阵
        # 可学习的温度系数logit_scale
        tea_image_logits, tea_text_logits = self.get_logits(
            tea_image_features, tea_text_features, tea_logit_scale, config)

        # 3. 生成真实标签
        logit_nums = stu_image_logits.shape[0]
        labels = self.get_ground_truth(stu_image_features, logit_nums)

        # 4. 计算对称交叉熵损失
        contrastive_loss = (F.cross_entropy(stu_image_logits, labels) +
                            F.cross_entropy(stu_text_logits, labels)) / 2

        # 5. 计算蒸馏损失
        distill_loss = (
            self.compute_distill_loss(tea_image_logits, stu_image_logits) +
            self.compute_distill_loss(tea_text_logits, stu_text_logits)) / 2

        loss_dict = {
            'contrastive_loss': contrastive_loss,
            'distill_loss': distill_loss
        }

        return loss_dict
