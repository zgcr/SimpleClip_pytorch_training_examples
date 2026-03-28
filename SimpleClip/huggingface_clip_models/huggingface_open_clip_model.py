'''
clip model under OpenCLIP library tag
https://huggingface.co/models?library=open_clip
clip model example:
laion/CLIP-ViT-B-32-laion2B-s34B-b79K:用laion2B数据集训练,s34B是训练步数为34B,b79K是批次大小为79K
clip dataset:
laion2B/laion5B/DataComp-1.4B/CommonPool-12.8B
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import create_model_and_transforms, get_tokenizer

# no weight decay layer:'positional_embedding', 'class_embedding', 'logit_scale', 'logit_bias'

# hf国内代理
# https://hf-mirror.com/
# export HF_ENDPOINT=https://hf-mirror.com

# support all open_clip models:
# https://github.com/mlfoundations/open_clip
# https://huggingface.co/models?library=open_clip
#############################################################################################
# laion/CLIP-ViT-B-16-laion2B-s34B-b88K: ImageNet-1k zero-shot top1 is 70.2%
# laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K: ImageNet-1k zero-shot top1 is 73.5%
#############################################################################################
# laion/CLIP-ViT-B-32-laion2B-s34B-b79K: ImageNet-1k zero-shot top1 is 66.6%
# laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K: ImageNet-1k zero-shot top1 is 72.7%
# laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K: ImageNet-1k zero-shot top1 is 72.7%
#############################################################################################
# laion/CLIP-ViT-L-14-laion2B-s32B-b82K: ImageNet-1k zero-shot top1 is 75.3%
# laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K: ImageNet-1k zero-shot top1 is 79.2%
#############################################################################################
# laion/CLIP-ViT-H-14-laion2B-s32B-b79K: ImageNet-1k zero-shot top1 is 78.0%
#############################################################################################
# laion/CLIP-ViT-g-14-laion2B-s12B-b42K: ImageNet-1k zero-shot top1 is 76.6%
# laion/CLIP-ViT-g-14-laion2B-s34B-b88K: ImageNet-1k zero-shot top1 is 78.4%
#############################################################################################
# laion/CLIP-ViT-bigG-14-laion2B-39B-b160k: ImageNet-1k zero-shot top1 is 80.1%
#############################################################################################
# apple/DFN2B-CLIP-ViT-B-16: ImageNet-1k zero-shot top1 is 76.2%
# apple/DFN2B-CLIP-ViT-L-14: ImageNet-1k zero-shot top1 is 81.4%
# apple/DFN2B-CLIP-ViT-L-14-39B: ImageNet-1k zero-shot top1 is 82.1%
# apple/DFN5B-CLIP-ViT-H-14: ImageNet-1k zero-shot top1 is 83.4%
# apple/DFN5B-CLIP-ViT-H-14-378: ImageNet-1k zero-shot top1 is 84.2%
#############################################################################################
# timm/vit_base_patch16_clip_224.openai
# timm/vit_base_patch32_clip_224.openai
# timm/vit_large_patch14_clip_224.openai
# timm/vit_large_patch14_clip_336.openai
#############################################################################################
# timm/ViT-B-16-SigLIP
# timm/ViT-B-16-SigLIP-256
# timm/ViT-B-16-SigLIP-i18n-256
# timm/ViT-B-16-SigLIP-384
# timm/ViT-B-16-SigLIP-512
#############################################################################################
# timm/ViT-L-16-SigLIP-256
# timm/ViT-L-16-SigLIP-384
#############################################################################################
# timm/ViT-B-16-SigLIP2
# timm/ViT-B-16-SigLIP2-256
# timm/ViT-B-16-SigLIP2-384
# timm/ViT-B-16-SigLIP2-512
#############################################################################################
# timm/ViT-L-16-SigLIP2-256
# timm/ViT-L-16-SigLIP2-384
# timm/ViT-L-16-SigLIP2-512
#############################################################################################
# timm/ViT-gopt-16-SigLIP2-256
# timm/ViT-gopt-16-SigLIP2-384
#############################################################################################


class HuggingFaceOpenClipModel(nn.Module):

    def __init__(self,
                 hf_model_name,
                 cache_dir=None,
                 new_image_size=None,
                 use_gradient_checkpoint=False):
        '''
        hf_model_name: Hugging Face model ID: 如 laion/CLIP-ViT-B-16-laion2B-s34B-b88K
        pretrained: 是否加载模型预训练权重
        cache_dir: 模型预训练下载到本地的路径(可选), 为None则下载到Hugging Face默认下载目录
        local_model_path: 是否仅使用本地路径模型预训练权重加载(可选), 为None则下载到Hugging Face默认下载目录
        '''
        super(HuggingFaceOpenClipModel, self).__init__()
        support_model_name_list = [
            'laion/clip',
            'apple/dfn2b',
            'apple/dfn5b',
            'timm/vit',
        ]
        assert any(per_model_name in hf_model_name.lower() for per_model_name
                   in support_model_name_list), 'Unsupported clip model!'

        hf_model_name = f'hf-hub:{hf_model_name}'

        self.model, self.train_preprocess, self.val_preprocess = create_model_and_transforms(
            model_name=hf_model_name,
            cache_dir=cache_dir,
            force_image_size=new_image_size)

        self.tokenizer = get_tokenizer(
            hf_model_name, context_length=self.model.context_length)

        if use_gradient_checkpoint:
            if hasattr(self.model, 'set_grad_checkpointing'):
                self.model.set_grad_checkpointing(enable=True)
                print("Using standard gradient checkpointing")
            else:
                if hasattr(self.model.visual, 'set_grad_checkpointing'):
                    self.model.visual.set_grad_checkpointing(enable=True)
                    print("Using gradient checkpointing for visual encoder")

                if hasattr(self.model.text, 'set_grad_checkpointing'):
                    self.model.text.set_grad_checkpointing(enable=True)
                    print("Using gradient checkpointing for text encoder")

    def encode_image(self, images):
        image_features = self.model.encode_image(images)

        return image_features

    def encode_text(self, texts):
        text_features = self.model.encode_text(texts)

        return text_features

    def forward(self, images, tokens):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        outputs = {
            'image_features': image_features,
            'text_features': text_features,
        }

        outputs['logit_scale'] = self.model.logit_scale.exp()

        if hasattr(self.model, 'logit_bias'):
            outputs['logit_bias'] = self.model.logit_bias

        return outputs


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from PIL import Image

    ###############################################################################
    # laion/CLIP-ViT-B-16-laion2B-s34B-b88K
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(
        hf_model_name='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()

        probs = (logit_scale *
                 image_features @ text_features.T).softmax(dim=-1)

    print(f'probs: {probs}')

    ###############################################################################
    # timm/vit_base_patch16_clip_224.openai
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(
        hf_model_name='timm/vit_base_patch16_clip_224.openai',
        use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()

        probs = (logit_scale *
                 image_features @ text_features.T).softmax(dim=-1)

    print(f'probs: {probs}')

    ###############################################################################
    # apple/DFN2B-CLIP-ViT-B-16
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(hf_model_name='apple/DFN2B-CLIP-ViT-B-16',
                                   use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()

        probs = (logit_scale *
                 image_features @ text_features.T).softmax(dim=-1)

    print(f'probs: {probs}')

    ###############################################################################
    # timm/ViT-B-16-SigLIP
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(hf_model_name='timm/ViT-B-16-SigLIP',
                                   use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()
        logit_bias = net.model.logit_bias

        probs = torch.sigmoid(image_features @ text_features.T * logit_scale +
                              logit_bias)

    print(f'probs: {probs}')

    ###############################################################################
    # timm/ViT-B-16-SigLIP2
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(hf_model_name='timm/ViT-B-16-SigLIP2',
                                   use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()
        logit_bias = net.model.logit_bias

        probs = torch.sigmoid(image_features @ text_features.T * logit_scale +
                              logit_bias)

    print(f'probs: {probs}')

    ###############################################################################
    # local_files_only = True
    # laion/CLIP-ViT-B-16-laion2B-s34B-b88K
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceOpenClipModel(
        hf_model_name='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        cache_dir='/root/autodl-tmp/cache/hub',
        use_gradient_checkpoint=False)
    net = net.cuda()
    net.eval()
    processor = net.val_preprocess
    tokenizer = net.tokenizer

    image = processor(
        Image.open('./beignets-task-guide.png')).unsqueeze(0).cuda()
    text = tokenizer(['a dog', 'a cat', 'a donut', 'a beignet']).cuda()

    with torch.no_grad():
        image_features = net.encode_image(image)
        text_features = net.encode_text(text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = net.model.logit_scale.exp()

        probs = (logit_scale *
                 image_features @ text_features.T).softmax(dim=-1)

    print(f'probs: {probs}')
