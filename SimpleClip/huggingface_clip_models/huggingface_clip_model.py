import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor, AutoConfig

# no weight decay layer:'positional_embedding', 'class_embedding', 'logit_scale', 'logit_bias'

# hf国内代理
# https://hf-mirror.com/
# export HF_ENDPOINT=https://hf-mirror.com

# Supported Clip Models:
# not support open_clip models
#############################################################################################
# openai/clip-vit-base-patch16
# openai/clip-vit-base-patch32
# openai/clip-vit-large-patch14
# openai/clip-vit-large-patch14-336
#############################################################################################
# google/siglip-base-patch16-224
# google/siglip-base-patch16-256
# google/siglip-base-patch16-384
# google/siglip-base-patch16-512
#############################################################################################
# google/siglip-large-patch16-256
# google/siglip-large-patch16-384
#############################################################################################
# google/siglip2-base-patch16-224
# google/siglip2-base-patch16-256
# google/siglip2-base-patch16-384
# google/siglip2-base-patch16-512
#############################################################################################
# google/siglip2-base-patch32-256
#############################################################################################
# google/siglip2-large-patch16-256
# google/siglip2-large-patch16-384
# google/siglip2-large-patch16-512
#############################################################################################
# google/siglip2-giant-opt-patch16-256
# google/siglip2-giant-opt-patch16-384
#############################################################################################


class HuggingFaceClipModel(nn.Module):

    def __init__(self,
                 hf_model_name,
                 pretrained=True,
                 cache_dir=None,
                 local_files_only=False,
                 use_gradient_checkpoint=False):
        '''
        hf_model_name: Hugging Face model ID: 如 laion/CLIP-ViT-B-16-laion2B-s34B-b88K
        pretrained: 是否加载模型预训练权重
        cache_dir: 模型预训练权重下载到本地的路径(可选), 为None则预训练权重下载到Hugging Face默认下载目录
        local_model_path: 是否仅使用本地路径预训练权重加载(可选), 为None则预训练权重下载到Hugging Face默认下载目录
        use_gradient_checkpoint: 模型训练时是否开启gradient checkpoint
        '''
        super(HuggingFaceClipModel, self).__init__()
        support_model_name_list = [
            'openai/clip',
            'google/siglip',
            'google/siglip2',
        ]
        assert any(per_model_name in hf_model_name.lower() for per_model_name
                   in support_model_name_list), 'Unsupported clip model!'

        self.config = AutoConfig.from_pretrained(
            hf_model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only)

        self.processor = AutoProcessor.from_pretrained(
            hf_model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=False)

        if pretrained:
            self.model = AutoModel.from_pretrained(
                hf_model_name,
                config=self.config,
                cache_dir=cache_dir,
                local_files_only=local_files_only)
        else:
            self.model = AutoModel.from_config(self.config)

        if use_gradient_checkpoint:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print(f'using gradient checkpoint!')
            else:
                if hasattr(self.model.visual, 'gradient_checkpointing_enable'):
                    self.model.visual.gradient_checkpointing_enable()
                    print("Using gradient checkpoint for visual encoder")

                if hasattr(self.model.text, 'gradient_checkpointing_enable'):
                    self.model.text.gradient_checkpointing_enable()
                    print("Using gradient checkpointi for text encoder")

    def get_logit_scale(self):
        if hasattr(self.model, 'logit_scale'):
            return self.model.logit_scale

        # 某些模型可能将 logit_scale 放在 text_model 或 vision_model 中
        for module in [self.model.text_model, self.model.vision_model]:
            if hasattr(module, 'logit_scale'):
                return module.logit_scale

        return None

    def get_logit_bias(self):
        if hasattr(self.model, 'logit_bias'):
            return self.model.logit_bias

        # 某些模型可能将 logit_bias 放在 text_model 或 vision_model 中
        for module in [self.model.text_model, self.model.vision_model]:
            if hasattr(module, 'logit_bias'):
                return module.logit_bias

        return None

    def forward(self, inputs):
        outputs = self.model(**inputs)
        image_features = outputs['image_embeds']
        text_features = outputs['text_embeds']

        outputs = {
            'image_features': image_features,
            'text_features': text_features,
        }

        logit_scale = self.get_logit_scale()
        logit_bias = self.get_logit_bias()

        outputs['logit_scale'] = logit_scale.exp()

        if logit_bias is not None:
            outputs['logit_bias'] = logit_bias

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
    # openai/clip-vit-base-patch16
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceClipModel(hf_model_name='openai/clip-vit-base-patch16',
                               pretrained=True,
                               local_files_only=False,
                               use_gradient_checkpoint=False)
    model = net.model
    processor = net.processor
    model = model.cuda()
    model.eval()

    image = Image.open('./000000039769.jpg')
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(images=image,
                       text=text,
                       return_tensors='pt',
                       padding="max_length",
                       max_length=77)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f'process inputs keys: {inputs.keys()}')
    for key, value in inputs.items():
        print(key, inputs[key].shape)

    # official inference
    with torch.no_grad():
        outputs = model(**inputs)
        print(f'outputs keys: {outputs.keys()}')
        probs = outputs.logits_per_image.softmax(dim=1)

    print(f'Text Probs with offical inference: {probs}')

    # self forward inference
    with torch.no_grad():
        outputs = net(inputs)
        print(f'outputs keys: {outputs.keys()}')
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        logit_scale = outputs['logit_scale']

        probs = (logit_scale *
                 image_features @ text_features.t()).softmax(dim=-1)

    print(f'Text Probs with self forward inference: {probs}')

    ###############################################################################
    # google/siglip-base-patch16-224
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceClipModel(hf_model_name='google/siglip-base-patch16-224',
                               pretrained=True,
                               local_files_only=False,
                               use_gradient_checkpoint=False)
    model = net.model
    processor = net.processor
    model = model.cuda()
    model.eval()

    image = Image.open('./000000039769.jpg')
    text = ["a photo of 2 cats", "a photo of 2 dogs"]
    inputs = processor(images=image,
                       text=text,
                       return_tensors='pt',
                       padding="max_length",
                       max_length=64)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f'process inputs keys: {inputs.keys()}')
    for key, value in inputs.items():
        print(key, inputs[key].shape)

    # official inference
    with torch.no_grad():
        outputs = model(**inputs)
        print(f'outputs keys: {outputs.keys()}')
        probs = torch.sigmoid(outputs.logits_per_image)

    print(f'Text Probs with offical inference: {probs}')

    # forward inference
    with torch.no_grad():
        outputs = net(inputs)
        print(f'outputs keys: {outputs.keys()}')
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        logit_scale = outputs['logit_scale']
        logit_bias = outputs['logit_bias']

        probs = torch.sigmoid(image_features @ text_features.T * logit_scale +
                              logit_bias)

    print(f'Text Probs with self forward inference: {probs}')

    ###############################################################################
    # google/siglip2-base-patch16-224
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceClipModel(hf_model_name='google/siglip2-base-patch16-224',
                               pretrained=True,
                               local_files_only=False,
                               use_gradient_checkpoint=False)
    model = net.model
    processor = net.processor
    model = model.cuda()
    model.eval()

    image = Image.open('./000000039769.jpg')
    text = ["a photo of 2 cats", "a photo of a plane"]
    inputs = processor(images=image,
                       text=text,
                       return_tensors='pt',
                       padding="max_length",
                       max_length=64)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f'process inputs keys: {inputs.keys()}')
    for key, value in inputs.items():
        print(key, inputs[key].shape)

    # official inference
    with torch.no_grad():
        outputs = model(**inputs)
        print(f'outputs keys: {outputs.keys()}')
        probs = torch.sigmoid(outputs.logits_per_image)

    print(f'Text Probs with offical inference: {probs}')

    # forward inference
    with torch.no_grad():
        outputs = net(inputs)
        print(f'outputs keys: {outputs.keys()}')
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        logit_scale = outputs['logit_scale']
        logit_bias = outputs['logit_bias']

        probs = torch.sigmoid(image_features @ text_features.T * logit_scale +
                              logit_bias)

    print(f'Text Probs with self forward inference: {probs}')

    ###############################################################################
    # local_files_only = True
    # openai/clip-vit-base-patch16
    # cache_dir: HF_HOME dir +/hub
    net = HuggingFaceClipModel(hf_model_name='openai/clip-vit-base-patch16',
                               pretrained=True,
                               cache_dir='/root/autodl-tmp/cache/hub',
                               local_files_only=True,
                               use_gradient_checkpoint=False)
    model = net.model
    processor = net.processor
    model = model.cuda()
    model.eval()

    image = Image.open('./000000039769.jpg')
    text = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(images=image,
                       text=text,
                       return_tensors='pt',
                       padding="max_length",
                       max_length=77)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f'process inputs keys: {inputs.keys()}')
    for key, value in inputs.items():
        print(key, inputs[key].shape)

    # official inference
    with torch.no_grad():
        outputs = model(**inputs)
        print(f'outputs keys: {outputs.keys()}')
        probs = outputs.logits_per_image.softmax(dim=1)

    print(f'Text Probs with offical inference: {probs}')

    # self forward inference
    with torch.no_grad():
        outputs = net(inputs)
        print(f'outputs keys: {outputs.keys()}')
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        logit_scale = outputs['logit_scale']

        probs = (logit_scale *
                 image_features @ text_features.t()).softmax(dim=-1)

    print(f'Text Probs with self forward inference: {probs}')
