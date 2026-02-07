import math
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode


class Opencv2PIL:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = Image.fromarray(np.uint8(image))

        sample['image'], sample['caption'] = image, caption

        return sample


class PIL2Opencv:

    def __init__(self):
        pass

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = np.asarray(image).astype(np.float32)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchRandomResizedCrop:

    def __init__(self,
                 resize=224,
                 scale=(0.9, 1.0),
                 interpolation=InterpolationMode.BICUBIC):
        self.RandomResizedCrop = transforms.RandomResizedCrop(
            int(resize), scale=scale, interpolation=interpolation)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.RandomResizedCrop(image)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchRandomCrop:

    def __init__(self, resize=224):
        self.RandomCrop = transforms.RandomCrop(int(resize))

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.RandomCrop(image)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchColorJitter:

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0):
        self.ColorJitter = transforms.ColorJitter(brightness=brightness,
                                                  contrast=contrast,
                                                  saturation=saturation,
                                                  hue=hue)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.ColorJitter(image)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchResize:

    def __init__(self, resize=224, interpolation=InterpolationMode.BICUBIC):
        self.Resize = transforms.Resize(int(resize),
                                        interpolation=interpolation)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.Resize(image)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchCenterCrop:

    def __init__(self, resize=224):
        self.CenterCrop = transforms.CenterCrop(int(resize))

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.CenterCrop(image)

        sample['image'], sample['caption'] = image, caption

        return sample


class TorchMeanStdNormalize:

    def __init__(self,
                 mean=[0.48145466, 0.4578275, 0.40821073],
                 std=[0.26862954, 0.26130258, 0.27577711]):
        self.to_tensor = transforms.ToTensor()
        self.Normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        '''
        sample must be a dict,contains 'image'、'caption' keys.
        '''
        image, caption = sample['image'], sample['caption']

        image = self.to_tensor(image)
        image = self.Normalize(image)
        # 3 H W ->H W 3
        image = image.permute(1, 2, 0)
        image = image.numpy()

        sample['image'], sample['caption'] = image, caption

        return sample


class ImageCaptionPairCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        captions = [s['caption'] for s in data]

        images = np.array(images).astype(np.float32)
        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2)

        return {
            'image': images,
            'caption': captions,
        }


class HuggingFaceOpenClipModelImageCaptionPairCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        captions = [s['caption'] for s in data]

        # [3, H, W] ->B 3 H W
        images = torch.stack(images, dim=0).float()

        return {
            'image': images,
            'caption': captions,
        }


class HuggingFaceOpenClipModelZeroShortCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        # [3, H, W] ->B 3 H W
        images = torch.stack(images, dim=0).float()

        return {
            'image': images,
            'label': labels,
        }


class HuggingFaceClipModelImageCaptionPairCollater:

    def __init__(self, processor, max_length):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, data):
        images = [s['image'] for s in data]
        captions = [s['caption'] for s in data]

        inputs = self.processor(images=images,
                                text=captions,
                                return_tensors='pt',
                                padding="max_length",
                                max_length=self.max_length)

        return {
            'input': inputs,
        }


class HuggingFaceClipModelZeroShortCollater:

    def __init__(self, processor, max_length):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, data):
        images = [s['image'] for s in data]
        labels = [s['label'] for s in data]

        inputs = self.processor(images=images,
                                return_tensors='pt',
                                padding="max_length",
                                max_length=self.max_length)

        return {
            'input': inputs,
            'label': labels,
        }


class AverageMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccMeter:
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc1_correct_num = 0
        self.acc5_correct_num = 0
        self.sample_num = 0
        self.acc1 = 0
        self.acc5 = 0

    def update(self, acc1_correct_num, acc5_correct_num, sample_num):
        self.acc1_correct_num += acc1_correct_num
        self.acc5_correct_num += acc5_correct_num
        self.sample_num += sample_num

    def compute(self):
        self.acc1 = float(self.acc1_correct_num
                          ) / self.sample_num if self.sample_num != 0 else 0
        self.acc5 = float(self.acc5_correct_num
                          ) / self.sample_num if self.sample_num != 0 else 0


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    '''
    saved_model_path: a saved model.state_dict() .pth file path
    model: a new defined model
    excluded_layer_name: layer names that doesn't want to load parameters
    loading_new_input_size_position_encoding_weight: default False, for vit net, loading a position encoding layer with new input size, set True
    only load layer parameters which has same layer name and same layer weight shape
    '''
    if not saved_model_path:
        print('No pretrained model file!')
        return

    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'),
                                  weights_only=True)

    not_loaded_save_state_dict = []
    filtered_state_dict = {}
    for name, weight in saved_state_dict.items():
        if name in model.state_dict() and not any(
                excluded_name in name for excluded_name in excluded_layer_name
        ) and weight.shape == model.state_dict()[name].shape:
            filtered_state_dict[name] = weight
        else:
            not_loaded_save_state_dict.append(name)

    # resize clip visual position encoding layer with new input size
    if 'visual.positional_embedding' in saved_state_dict.keys(
    ) and 'visual.positional_embedding' not in not_loaded_save_state_dict:
        old_visual_position_embedding = saved_state_dict[
            'visual.positional_embedding']
        grid_size = model.visual.grid_size
        extra_tokens = 1
        new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
        if new_seq_len != old_visual_position_embedding.shape[0]:
            print('resize clip visual position embedding!')

            if extra_tokens:
                pos_emb_tok, pos_emb_img = old_visual_position_embedding[:extra_tokens], old_visual_position_embedding[
                    extra_tokens:]
            else:
                pos_emb_tok, pos_emb_img = None, old_visual_position_embedding
            old_grid_size = [
                int(math.sqrt(len(pos_emb_img))),
                int(math.sqrt(len(pos_emb_img))),
            ]

            pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0],
                                              old_grid_size[1],
                                              -1).permute(0, 3, 1, 2)
            pos_emb_img = F.interpolate(pos_emb_img,
                                        size=grid_size,
                                        mode='bicubic',
                                        align_corners=False)
            pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(
                1, grid_size[0] * grid_size[1], -1)[0]
            if pos_emb_tok is not None:
                new_visual_position_embedding = torch.cat(
                    [pos_emb_tok, pos_emb_img], dim=0)
            else:
                new_visual_position_embedding = pos_emb_img

            filtered_state_dict[
                'visual.positional_embedding'] = new_visual_position_embedding

            not_loaded_save_state_dict.remove('visual.positional_embedding')

    # resize clip text position encoding layer with new input size
    if 'text.positional_embedding' in saved_state_dict.keys(
    ) and 'text.positional_embedding' not in not_loaded_save_state_dict:
        old_text_position_embedding = saved_state_dict[
            'text.positional_embedding']

        old_text_position_width = old_text_position_embedding.shape[1]
        new_text_position_width = model.text.positional_embedding.shape[1]

        assert old_text_position_width == new_text_position_width, 'text positional_embedding width changed!'

        old_text_position_num = old_text_position_embedding.shape[0]
        new_text_position_num = model.text.positional_embedding.shape[0]
        if old_text_position_num != new_text_position_num:
            print('resize clip text position embedding!')

            old_text_position_embedding = old_text_position_embedding.reshape(
                1, old_text_position_num,
                old_text_position_width).permute(0, 2, 1)
            old_text_position_embedding = F.interpolate(
                old_text_position_embedding,
                size=new_text_position_num,
                mode='linear',
                align_corners=False)
            old_text_position_embedding = old_text_position_embedding.permute(
                0, 2, 1)[0]
            new_text_position_embedding = old_text_position_embedding

            filtered_state_dict[
                'text.positional_embedding'] = new_text_position_embedding

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(
            f'load/model weight nums:{len(filtered_state_dict)}/{len(model.state_dict())}'
        )
        print(f'not loaded save layer weight:\n{not_loaded_save_state_dict}')
        model.load_state_dict(filtered_state_dict, strict=False)

    return
