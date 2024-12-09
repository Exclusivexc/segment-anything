from train import load_sam
from sklearn.preprocessing import binarize
import torch.nn as nn 
import logging
import torch.backends.cudnn as cudnn
import os
import numpy as np
import torch.optim as optim
import argparse
import random
import sys

from tqdm import tqdm
sys.path.append("..")
sys.path.append("/media/ubuntu/maxiaochuan/myscripts/data_augmentation")
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/scripts")
from segment_anything_finetune import sam_model_registry
from find_all_bounding_box_2d import find_bounding_boxes
import image_io
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from torchvision import transforms
import torch


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除零错误，平滑参数

    def forward(self, preds, targets):
        # 将 logits 转为概率（0到1之间的值），应用 sigmoid

        # 计算交集和并集
        intersection = torch.sum(preds * targets)  # 交集
        union = torch.sum(preds * preds) + torch.sum(targets * targets)  # 并集
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # 加上平滑项避免除零

        # Dice Loss = 1 - Dice Coefficient
        dice_loss = 1 - dice
        return dice_loss


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_type', type=str, default='vit_b', help='sam checkpoint type')
    parser.add_argument('--root_path', type=str, default='/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/pencoder_and_decoder', help='Name of Experiment')
    parser.add_argument('--save_model_name', type=str, default='pencoder_and_decoder')
    parser.add_argument('--max_epoch', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=3, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[1024, 1024], help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    
    return parser.parse_args()


sam = load_sam(model_type="/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/model/pencodermdecoder/iter_34800_dice_0.9542.pth")
image_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/nnUNet-master/nnUNet_raw/Dataset501_Polyp/imagesTs_RGB"
pred_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/nnUNet-master/nnUNet_results/Dataset501_Polyp/nnUNetTrainer__nnUNetPlans__2d_aug2_1024_1280/fold_0/test"
label_dir = "/media/ubuntu/maxiaochuan/ISICDM 2024/nnUNet-master/nnUNet_raw/Dataset501_Polyp/labelsTs"
files = os.listdir(pred_dir)
alls = []
alls2 = []
for file in tqdm(files):
    if file.endswith(".png"):
        image_path = os.path.join(image_dir, file)
        pred_path = os.path.join(pred_dir, file)
        label_path = os.path.join(label_dir, file)
        image = image_io.read_png(image_path)
        pred = image_io.read_png(pred_path)
        label = image_io.read_png(label_path)
        batched_input = []


        transform = transforms.Compose([
                transforms.ToTensor(),
            ])


        with torch.no_grad():
            image = transform(image)
            # pred = transform(pred)
            label = transform(label)
            original_size = tuple(image.shape[-2:])
            sam_transform = ResizeLongestSide(sam.image_encoder.img_size)
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = sam_transform.apply_image(image)
            image = torch.as_tensor(image, dtype=torch.float).permute(2, 0, 1)
            box = find_bounding_boxes(pred)
            pred = transform(pred).cuda()
            pred[pred > 0] = 1
            box = np.array([[box[i][1], box[i][0], box[i][3], box[i][2]] for i in range(len(box))])
            box = sam_transform.apply_boxes(box, original_size=original_size)
            box = torch.Tensor(box).cuda()
            image = image.cuda()
            label = label.cuda()
            
            batched_input.append({
                'image': image,
                'original_size': original_size,
                'boxes': box,
            })

            outputs = sam(
                batched_input=batched_input,
                multimask_output=False,
            )
            outputs = torch.cat([x["masks"] for x in outputs], dim=0)
            outputs = torch.sigmoid(outputs)

            out = torch.zeros((1, outputs.shape[1], outputs.shape[2], outputs.shape[3])).cuda()
            
            """ with post """
            for i in range(outputs.shape[0]):
                out[outputs[i].unsqueeze(0) > 0.5] += 1
            out[out != 1] = 0
            
            """ without post """
            # for i in range(outputs.shape[0]):
            #     out[outputs[i].unsqueeze(0) > 0.5] = 1
            

            dice = DiceLoss()
            # print(outputs.shape, label.unsqueeze(0).shape, pred.shape)
            refine_dice = 1 - dice(out, label.unsqueeze(0)) 
            similarity_dice = 1 - dice(pred, out)
            raw_dice = 1 - dice(pred, label.unsqueeze(0))
            print(f"raw_dice: {raw_dice.item(): .4f}, refine_dice: {refine_dice.item(): .4f}, similarity_dice: {similarity_dice.item(): .4f}")
            alls.append([raw_dice.item(), similarity_dice.item(), refine_dice.item(), file]) 
            alls2.append([(label.sum() / (label.shape[1] * label.shape[2])).item(), raw_dice.item()])

alls.sort()
alls2.sort()
with open("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/output.txt", 'w') as f:
    for a, b, c, d in alls:
        f.write(f"{d}, sim: {b: .4f}, raw: {a :.4f}, refine: {c :.4f}\n")

    for i in range(len(alls)):
        alls[i] = alls[i][:3]
    alls = np.array(alls)
    raw = alls[:, 0]
    refine = alls[:, 2]
    f.write(f"mean: {raw.mean()}, std: {raw.std()}, max: {raw.max()}, min: {raw.min()}\n")
    f.write(f"mean: {refine.mean()}, std: {refine.std()}, max: {refine.max()}, min: {refine.min()}\n")
    alls2 = np.array(alls2)
    t1 = alls2[: len(alls2) // 2, 1]
    t2 = alls2[len(alls2) // 2:, 1]
    t3 = alls2[:, 1]
    f.write(f"mean: {t1.mean()}, std: {t1.std()}, max: {t1.max()}, min: {t1.min()}\n")
    f.write(f"mean: {t2.mean()}, std: {t2.std()}, max: {t2.max()}, min: {t2.min()}\n")
    f.write(f"mean: {t3.mean()}, std: {t3.std()}, max: {t3.max()}, min: {t3.min()}\n")


