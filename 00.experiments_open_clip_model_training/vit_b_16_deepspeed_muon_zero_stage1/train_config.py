import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import realsyn_dataset_path

from SimpleClip.models import clip
from SimpleClip.models.tokenizer import Qwen35Tokenizer
from SimpleClip.losses import ClipLoss
from SimpleClip.datasets.realsyn_dataset import RealSynDataset
from SimpleClip.common import Opencv2PIL, TorchRandomResizedCrop, TorchMeanStdNormalize, ImageCaptionPairCollater, load_state_dict

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class config:
    network = 'vit_base_patch16_clip'

    model = clip.__dict__[network](**{
        'use_gradient_checkpoint': True,
    })

    print(f'context_length:{model.context_length}')

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model)

    tokenizer = Qwen35Tokenizer(context_length=model.context_length)

    use_siglip_loss = False
    train_criterion = ClipLoss(cache_labels=True,
                               compute_local_loss=True,
                               gather_features_with_grad=True)

    train_dataset = RealSynDataset(
        root_dir=realsyn_dataset_path,
        dataset_name=[
            'RealSyn15M_1_5',
            'RealSyn15M_6_10',
            'RealSyn15M_11_15',
            'RealSyn15M_16_21',
        ],
        caption_type='web_caption',
        random_select_caption=True,
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
    batch_size = 512
    # num_workers is total workers
    num_workers = 8
    accumulation_steps = 1

    optimizer = (
        'Muon',
        {
            'lr': 5e-4,
            'weight_decay': 0.2,
            'exclude_muon_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0,
            'min_lr': 5e-6,
        },
    )

    epochs = 100
    print_interval = 100
    save_interval = 10

    use_step_save_interval = False
    step_save_interval = 10000

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

    # ZeRO stage: 1, 2, 3
    deepspeed_zero_stage = 1
    # ZeRO-Offload: offload optimizer states (and params for stage 3) to CPU
    deepspeed_offload = False

    use_ema_model = True
    ema_model_decay = 0.9999
    ema_model_tau = 2000
