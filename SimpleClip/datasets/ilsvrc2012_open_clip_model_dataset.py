import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import collections
import cv2
import numpy as np

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset

from SimpleClip.datasets.ilsvrc2012_class_info import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES


class ILSVRC2012Dataset(Dataset):
    '''
    ILSVRC2012 Dataset:https://image-net.org/ 
    '''

    def __init__(self,
                 root_dir,
                 class_idx_to_imagenet_classname_dict=IMAGENET_CLASSNAMES,
                 templates=OPENAI_IMAGENET_TEMPLATES,
                 set_name='val',
                 image_processor=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'

        self.image_name_list = []
        self.image_info_dict = collections.OrderedDict()

        self.sub_class_name_list = []
        self.sub_class_imagenet_classname_list = []
        self.sub_class_info_dict = collections.OrderedDict()
        set_dir = os.path.join(root_dir, set_name)
        for per_sub_class_name in tqdm(sorted(os.listdir(set_dir))):
            self.sub_class_name_list.append(per_sub_class_name)
            self.sub_class_imagenet_classname_list.append(
                class_idx_to_imagenet_classname_dict[per_sub_class_name])

            self.sub_class_info_dict[
                per_sub_class_name] = collections.OrderedDict()
            self.sub_class_info_dict[per_sub_class_name][
                'label_name'] = class_idx_to_imagenet_classname_dict[
                    per_sub_class_name]

            per_sub_class_dir = os.path.join(set_dir, per_sub_class_name)
            for per_image_name in sorted(os.listdir(per_sub_class_dir)):
                per_image_path = os.path.join(per_sub_class_dir,
                                              per_image_name)
                if os.path.exists(per_image_path):
                    self.image_name_list.append(per_image_name)
                    self.image_info_dict[
                        per_image_name] = collections.OrderedDict()

                    self.image_info_dict[per_image_name][
                        'path'] = per_image_path
                    self.image_info_dict[per_image_name][
                        'class_name'] = per_sub_class_name
                    self.image_info_dict[per_image_name][
                        'label_name'] = class_idx_to_imagenet_classname_dict[
                            per_sub_class_name]

        assert len(self.image_name_list) == len(self.image_info_dict)

        for idx, per_sub_class_name in enumerate(self.sub_class_name_list):
            self.sub_class_info_dict[per_sub_class_name]['label'] = idx

        for per_image_name in self.image_name_list:
            per_sub_class_name = self.image_info_dict[per_image_name][
                'class_name']
            self.image_info_dict[per_image_name][
                'label'] = self.sub_class_info_dict[per_sub_class_name][
                    'label']

        self.templates = templates
        self.image_processor = image_processor

        print(f'Dataset Size:{len(self.image_name_list)}')

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        path = self.image_info_dict[self.image_name_list[idx]]['path']
        image = self.load_image(idx)

        label = self.image_info_dict[self.image_name_list[idx]]['label']

        if self.image_processor:
            image = self.image_processor(image)

        sample = {
            'path': path,
            'image': image,
            'label': label,
        }

        return sample

    def load_image(self, idx):
        image = Image.open(self.image_info_dict[self.image_name_list[idx]]
                           ['path']).convert('RGB')

        return image


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

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(BASE_DIR)

    from tools.path import ILSVRC2012_path

    from tqdm import tqdm

    from SimpleClip.common import HuggingFaceOpenClipModelZeroShortCollater

    from SimpleClip.huggingface_clip_models.huggingface_open_clip_model import HuggingFaceOpenClipModel
    net = HuggingFaceOpenClipModel(
        hf_model_name='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        use_gradient_checkpoint=False)
    val_processor = net.train_preprocess

    ilsvrc2012traindataset = ILSVRC2012Dataset(
        root_dir=ILSVRC2012_path,
        class_idx_to_imagenet_classname_dict=IMAGENET_CLASSNAMES,
        templates=OPENAI_IMAGENET_TEMPLATES,
        set_name='val',
        image_processor=val_processor)

    count = 0
    for per_sample in tqdm(ilsvrc2012traindataset):
        print('1111', per_sample['path'])
        print('2222', per_sample['image'].shape)
        print('3333', type(per_sample['image']), type(per_sample['label']))

        print('4444', per_sample['label'])

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = per_sample['image']

        image = image.permute((1, 2, 0)).contiguous()
        image = np.asarray(image).astype(np.float32)

        mean = np.expand_dims(np.expand_dims(np.array(
            [0.48145466, 0.4578275, 0.40821073]),
                                             axis=0),
                              axis=0)
        std = np.expand_dims(np.expand_dims(np.array(
            [0.26862954, 0.26130258, 0.27577711]),
                                            axis=0),
                             axis=0)

        image = ((image * std) + mean) * 255.
        image = image.astype(np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = HuggingFaceOpenClipModelZeroShortCollater()
    train_loader = DataLoader(ilsvrc2012traindataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, labels = data['image'], data['label']
        print('1111', images.shape, len(labels))
        print('2222', images.dtype)

        print(labels)

        if count < 2:
            count += 1
        else:
            break
