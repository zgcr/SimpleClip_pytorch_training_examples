import os
import cv2
import json
import random
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset


class RealSynDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_name=[
                     'RealSyn15M_1_5',
                     'RealSyn15M_6_10',
                     'RealSyn15M_11_15',
                     'RealSyn15M_16_21',
                 ],
                 caption_type='web_caption',
                 random_select_caption=True,
                 transform=None):
        assert caption_type in ['web_caption',
                                'generated_caption'], 'Wrong caption type!'

        self.image_path_list = []
        self.image_caption_list = []
        for per_dataset_name in dataset_name:
            per_dataset_dir = os.path.join(root_dir, per_dataset_name)

            per_dataset_json_name_list = sorted([
                per_file_name for per_file_name in os.listdir(per_dataset_dir)
                if per_file_name.endswith('.json')
            ])

            for per_json_name in tqdm(per_dataset_json_name_list):
                per_sub_set_name = per_json_name[:-len('.json')]
                per_sub_set_image_dir = os.path.join(per_dataset_dir,
                                                     per_sub_set_name)
                if not os.path.isdir(per_sub_set_image_dir):
                    continue

                per_sub_set_json_path = os.path.join(per_dataset_dir,
                                                     per_json_name)
                with open(per_sub_set_json_path, encoding='utf-8') as f:
                    per_sub_set_caption_dict = json.load(f)

                for per_image_name in sorted(
                        os.listdir(per_sub_set_image_dir)):
                    if per_image_name not in per_sub_set_caption_dict:
                        continue

                    per_image_path = os.path.join(per_sub_set_image_dir,
                                                  per_image_name)
                    if not os.path.exists(per_image_path):
                        continue

                    per_image_caption = per_sub_set_caption_dict[
                        per_image_name][caption_type]

                    if isinstance(per_image_caption, str):
                        per_image_caption = [per_image_caption]

                    if not per_image_caption:
                        continue

                    self.image_path_list.append(per_image_path)
                    self.image_caption_list.append(per_image_caption)

                del per_sub_set_caption_dict

        assert len(self.image_path_list) == len(self.image_caption_list)

        self.caption_type = caption_type
        self.random_select_caption = random_select_caption
        self.transform = transform

        print(f'Dataset Size:{len(self.image_path_list)}')

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        path = self.image_path_list[idx]
        image = self.load_image(idx)
        caption = self.load_caption(idx)

        sample = {
            'path': path,
            'image': image,
            'caption': caption,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_list[idx], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_caption(self, idx):
        caption_list = self.image_caption_list[idx]

        if self.random_select_caption:
            caption = random.choice(caption_list)
        else:
            caption = caption_list[0]

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

    from tools.path import realsyn_dataset_path

    from tqdm import tqdm

    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    from SimpleClip.common import Opencv2PIL, PIL2Opencv, TorchRandomResizedCrop, TorchResize, TorchCenterCrop, ImageCaptionPairCollater

    realsyn15mtraindataset = RealSynDataset(
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
            PIL2Opencv(),
        ]))

    count = 0
    for per_sample in tqdm(realsyn15mtraindataset):
        print('1111', per_sample['path'], per_sample['image'].shape)
        print('2222', type(per_sample['image']), type(per_sample['caption']))

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        color = [random.randint(0, 255) for _ in range(3)]
        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
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
    collater = ImageCaptionPairCollater()
    train_loader = DataLoader(realsyn15mtraindataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, captions = data['image'], data['caption']
        print('1111', images.shape, len(captions))
        print('2222', images.dtype)

        for per_image_caption in captions:
            print(per_image_caption)

        if count < 2:
            count += 1
        else:
            break
