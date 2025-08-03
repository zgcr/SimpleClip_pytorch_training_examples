import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import image_caption_pair_dataset_path

from SimpleClip.models import clip_model
from SimpleClip.models.tokenizer import SimpleTokenizer
from SimpleClip.losses import ClipLoss
from SimpleClip.datasets.cc3m_dataset import CC3MDataset
from SimpleClip.common import Opencv2PIL, TorchRandomResizedCrop, TorchMeanStdNormalize, ImageCaptionPairCollater, load_state_dict

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class config:
    network = 'ViT_B_16_clip'

    model = clip_model.__dict__[network](**{
        'use_gradient_checkpoint': True,
    })

    print(f'context_length:{model.context_length}')

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    tokenizer = SimpleTokenizer(context_length=model.context_length)

    use_siglip_loss = False
    train_criterion = ClipLoss(cache_labels=True,
                               compute_local_loss=True,
                               gather_features_with_grad=True)

    train_dataset = CC3MDataset(
        root_dir=image_caption_pair_dataset_path,
        dataset_name=[
            'cc3m_final',
        ],
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224,
                                   scale=(0.9, 1.0),
                                   interpolation=InterpolationMode.BICUBIC),
            TorchMeanStdNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711]),
        ]))
    train_collater = ImageCaptionPairCollater()

    seed = 0
    # batch_size is total size
    batch_size = 32
    # num_workers is total workers
    num_workers = 4
    accumulation_steps = 1

    optimizer = (
        'AdamW',
        {
            'lr': 5e-4,
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
            'warm_up_epochs': 30,
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
