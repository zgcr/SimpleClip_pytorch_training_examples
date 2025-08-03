import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import image_caption_pair_dataset_path

from SimpleClip.huggingface_clip_models.huggingface_clip_model import HuggingFaceClipModel
from SimpleClip.losses import ClipLoss
from SimpleClip.datasets.cc3m_huggingface_clip_model_dataset import CC3MDataset
from SimpleClip.common import HuggingFaceClipModelImageCaptionPairCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'openai/clip-vit-base-patch16'

    model = HuggingFaceClipModel(
        hf_model_name=network,
        pretrained=True,
        cache_dir='/root/autodl-tmp/huggingface_clip_pretrained_model',
        local_files_only=False,
        use_gradient_checkpoint=True)

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model.model)

    train_processor = model.processor

    use_siglip_loss = False
    train_criterion = ClipLoss(cache_labels=True,
                               compute_local_loss=True,
                               gather_features_with_grad=True)

    train_dataset = CC3MDataset(root_dir=image_caption_pair_dataset_path,
                                dataset_name=[
                                    'cc3m_final',
                                ],
                                set_name='val')
    train_collater = HuggingFaceClipModelImageCaptionPairCollater(
        processor=train_processor, max_length=77)

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 4
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 1e-5,
            'global_weight_decay': False,
            # if global_weight_decay = False
            # all bias, bn and other 1d params weight set to 0 weight decay
            'weight_decay': 0.2,
            'no_weight_decay_layer_name_list': [],
            'beta1': 0.9,
            'beta2': 0.98,
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 1,
            'min_lr': 1e-6,
        },
    )

    epochs = 100
    print_interval = 100
    save_interval = 10

    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16

    sync_bn = False
    use_amp = True
    use_compile = False
    compile_params = {
        # 'default': optimizes for large models, low compile-time and no extra memory usage.
        # 'reduce-overhead': optimizes to reduce the framework overhead and uses some extra memory, helps speed up small models, model update may not correct.
        # 'max-autotune': optimizes to produce the fastest model, but takes a very long time to compile and may failed.
        'mode': 'default',
    }
