import os
import collections
import cv2
import json
import numpy as np

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset


class CC3MDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_name=[
                     'cc3m_final',
                 ],
                 set_name='val'):
        assert set_name in ['train', 'val'], 'Wrong set name!'

        self.image_name_list = []
        self.image_path_dict = collections.OrderedDict()
        self.image_caption_dict = collections.OrderedDict()
        for per_dataset_name in tqdm(dataset_name):
            per_dataset_image_dir = os.path.join(root_dir, per_dataset_name,
                                                 set_name)
            per_dataset_caption_json_path = os.path.join(
                root_dir, per_dataset_name,
                f'{per_dataset_name}_{set_name}.json')
            with open(per_dataset_caption_json_path, encoding='utf-8') as f:
                per_dataset_caption_dict = json.load(f)

            for per_image_name in sorted(os.listdir(per_dataset_image_dir)):
                per_image_path = os.path.join(per_dataset_image_dir,
                                              per_image_name)
                if os.path.exists(per_image_path):
                    self.image_name_list.append(per_image_name)
                    self.image_path_dict[per_image_name] = per_image_path
                    self.image_caption_dict[
                        per_image_name] = per_dataset_caption_dict[
                            per_image_name]['web_caption']

        assert len(self.image_name_list) == len(self.image_path_dict) == len(
            self.image_caption_dict)

        print(f'Dataset Size:{len(self.image_name_list)}')

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        path = self.image_path_dict[self.image_name_list[idx]]
        image = self.load_image(idx)
        caption = self.load_caption(idx)

        sample = {
            'path': path,
            'image': image,
            'caption': caption,
        }

        return sample

    def load_image(self, idx):
        image = Image.open(
            self.image_path_dict[self.image_name_list[idx]]).convert('RGB')

        return image

    def load_caption(self, idx):
        caption = self.image_caption_dict[self.image_name_list[idx]]

        return caption


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

    from tools.path import image_caption_pair_dataset_path

    from tqdm import tqdm

    from SimpleClip.common import HuggingFaceClipModelImageCaptionPairCollater

    from SimpleClip.huggingface_clip_models.huggingface_clip_model import HuggingFaceClipModel
    net = HuggingFaceClipModel(
        hf_model_name='openai/clip-vit-base-patch16',
        pretrained=True,
        cache_dir='/root/autodl-tmp/huggingface_clip_pretrained_model',
        local_files_only=False,
        use_gradient_checkpoint=False)
    processor = net.processor

    ilsvrc2012traindataset = CC3MDataset(
        root_dir=image_caption_pair_dataset_path,
        dataset_name=[
            'cc3m_final',
        ],
        set_name='val')

    count = 0
    for per_sample in tqdm(ilsvrc2012traindataset):
        print('1111', per_sample['path'], per_sample['image'].size)
        print('2222', type(per_sample['image']), type(per_sample['caption']))

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = per_sample['image']
        image = np.asarray(image).astype(np.float32)
        image = image.astype(np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        caption = per_sample['caption']
        print('3333', caption)

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = HuggingFaceClipModelImageCaptionPairCollater(
        processor=processor, max_length=77)
    train_loader = DataLoader(ilsvrc2012traindataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        inputs = data['input']
        print('1111', inputs.keys())
        print('2222', inputs['input_ids'].shape,
              inputs['attention_mask'].shape, inputs['pixel_values'].shape)

        if count < 2:
            count += 1
        else:
            break
