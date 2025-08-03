import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from SimpleClip.huggingface_clip_models.huggingface_clip_model import HuggingFaceClipModel
from SimpleClip.datasets.ilsvrc2012_clip_model_dataset import ILSVRC2012Dataset
from SimpleClip.datasets.ilsvrc2012_class_info import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES
from SimpleClip.common import HuggingFaceClipModelZeroShortCollater, load_state_dict

import torch
import torchvision.transforms as transforms


class config:
    network = 'openai/clip-vit-base-patch16'

    model = HuggingFaceClipModel(
        hf_model_name=network,
        pretrained=True,
        cache_dir='/root/autodl-tmp/huggingface_clip_pretrained_model',
        local_files_only=False,
        use_gradient_checkpoint=False)

    is_siglip_model = False

    # load pretrained model or not
    trained_model_path = ''
    load_state_dict(trained_model_path, model.model)

    val_processor = model.processor

    val_dataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        class_idx_to_imagenet_classname_dict=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        set_name='val')
    val_collater = HuggingFaceClipModelZeroShortCollater(
        processor=val_processor, max_length=77)

    seed = 0
    # batch_size is total size
    batch_size = 256
    # num_workers is total workers
    num_workers = 4

    # torch.float16 or torch.bfloat16
    amp_type = torch.bfloat16

    use_amp = False
