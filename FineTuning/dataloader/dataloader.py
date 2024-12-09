import sys
sys.path.append("/media/ubuntu/maxiaochuan/myscripts")
from find_all_bounding_box_2d import find_bounding_boxes
from segment_anything_finetune.utils.transforms import ResizeLongestSide
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import image_io

class Png_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, original_size, transform=None):
        """
        初始化函数
        :param image_dir: 存放图像的文件夹路径
        :param label_dir: 存放标签的文件夹路径
        :param transform: 用于图像的预处理和增强
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(label_dir)
        self.sam_trans = ResizeLongestSide(1024)
        self.original_size = original_size

    def __len__(self):
        """
        返回数据集中样本的数量
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取一个样本
        """
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = image_io.read_png(img_name)
        
        label_name = img_name.replace("images", "labels")
        label = image_io.read_png(label_name) 
        # print(f"imageshape: {image.shape}")
        data = {"image": image, "label": label}
        if self.transform:
            data = self.transform(data)
        # print(f"transformed_imageshape: {data['image'].shape}")
        # image = data["image"]
        # # original_size = tuple(image.shape[-2:])
        # image = image.permute(1, 2, 0).detach().cpu().numpy()
        # image = self.sam_trans.apply_image(image)
        # image = torch.as_tensor(image, dtype=torch.float).permute(2, 0, 1)
        # box = find_bounding_boxes(label)[0]
        # box = np.array([box[1], box[0], box[3], box[2]])
        # box = self.sam_trans.apply_boxes(box, original_size=self.original_size)
        # box = torch.Tensor(box)
        # data["box"] = box
        data["idx"] = idx
        data["filename"] = img_name

        return data


def get_data_loader(stage="train", batch_size=8, transform=None, original_size=None, worker_init_fn=None):
    if stage == 'train':
        image_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/data/SAM_imagesTr"
        label_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/data/SAM_labelsTr"

    else: 
        image_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/data/SAM_imagesTs"
        label_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/data/SAM_labelsTs"
    
    Polyp_dataset = Png_Dataset(image_dir=image_dir, label_dir=label_dir, original_size=original_size, transform=transform)
    return DataLoader(Polyp_dataset, batch_size=batch_size, num_workers=4, worker_init_fn=worker_init_fn)


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (tuple): Desired output size (height, width)
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (0, 0)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        h, w, _ = image.shape  # h and w are height and width of the image
        w1 = np.random.randint(0, w - self.output_size[1])  # width random offset
        h1 = np.random.randint(0, h - self.output_size[0])  # height random offset

        image = image[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1], :]
        label = label[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]

        if self.with_sdf:
            sdf = sdf[h1:h1 + self.output_size[0], w1:w1 + self.output_size[1]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Randomly rotate and flip the dataset in a sample
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # Random rotation (90 degrees)
        k = np.random.randint(0, 4)  # rotate 0, 90, 180, or 270 degrees
        image = np.rot90(image, k, axes=(0, 1))  # Rotate on the height and width axes
        label = np.rot90(label, k, axes=(0, 1))

        # Random flip
        axis = np.random.randint(0, 2)  # 0: vertical, 1: horizontal
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # Ensure the image shape is (height, width, channels) and then transpose it to (channels, height, width)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # Convert HWC to CHW format
        
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 
                    'label': torch.from_numpy(sample['label']).float(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).float()}
        else:
            return {'image': torch.from_numpy(image), 
                    'label': torch.from_numpy(sample['label']).float()}

