import os
import cv2
import json
import random
import numpy as np

from PIL import Image
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
                 random_select_caption=True):
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

        return sample

    def load_image(self, idx):
        image = Image.open(self.image_path_list[idx]).convert('RGB')

        return image

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

    from SimpleClip.common import HuggingFaceClipModelImageCaptionPairCollater

    from SimpleClip.huggingface_clip_models.huggingface_clip_model import HuggingFaceClipModel
    net = HuggingFaceClipModel(hf_model_name='openai/clip-vit-base-patch16',
                               pretrained=True,
                               local_files_only=False,
                               use_gradient_checkpoint=False)
    processor = net.processor

    realsyntraindataset = RealSynDataset(root_dir=realsyn_dataset_path,
                                         dataset_name=[
                                             'RealSyn15M_1_5',
                                             'RealSyn15M_6_10',
                                             'RealSyn15M_11_15',
                                             'RealSyn15M_16_21',
                                         ],
                                         caption_type='web_caption',
                                         random_select_caption=True)

    count = 0
    for per_sample in tqdm(realsyntraindataset):
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
    train_loader = DataLoader(realsyntraindataset,
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
