import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import realsyn_dataset_path

from SimpleClip.huggingface_clip_models.huggingface_open_clip_model import HuggingFaceOpenClipModel
from SimpleClip.losses import SigLipLoss
from SimpleClip.datasets.realsyn_huggingface_open_clip_model_dataset import RealSynDataset
from SimpleClip.common import HuggingFaceOpenClipModelImageCaptionPairCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'timm/ViT-B-16-SigLIP2'

    model = HuggingFaceOpenClipModel(
        hf_model_name=network,
        cache_dir='/root/autodl-tmp/huggingface/hub',
        use_gradient_checkpoint=True)

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model.model)

    train_processor = model.train_preprocess

    tokenizer = model.tokenizer

    use_siglip_loss = True
    train_criterion = SigLipLoss(dist_type='bidir')

    train_dataset = RealSynDataset(root_dir=realsyn_dataset_path,
                                   dataset_name=[
                                       'RealSyn15M_1_5',
                                       'RealSyn15M_6_10',
                                       'RealSyn15M_11_15',
                                       'RealSyn15M_16_21',
                                   ],
                                   caption_type='web_caption',
                                   random_select_caption=True,
                                   image_processor=train_processor)
    train_collater = HuggingFaceOpenClipModelImageCaptionPairCollater()

    seed = 0
    # batch_size is total size
    batch_size = 1024
    # num_workers is total workers
    num_workers = 8
    accumulation_steps = 1

    optimizer = (
        'MuonAdamW',
        {
            'lr': 1e-4,
            'weight_decay': 0.2,
            'global_weight_decay': False,
            'exclude_muon_layer_name_list': [],
        },
    )

    scheduler = (
        'CosineLR',
        {
            'warm_up_epochs': 0.1,
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

    use_ema_model = True
    ema_model_decay = 0.9999
    ema_model_tau = 2000

    clip_max_norm = 1.0
