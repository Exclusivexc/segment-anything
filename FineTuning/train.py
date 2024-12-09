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
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/scripts")
sys.path.append("/media/ubuntu/maxiaochuan/myscripts")
sys.path.append("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning")
from find_all_bounding_box_2d import find_bounding_boxes
from segment_anything_finetune import sam_model_registry
from segment_anything_finetune.utils.transforms import ResizeLongestSide
from dataloader.dataloader import get_data_loader, RandomCrop, RandomRotFlip, ToTensor
from torchvision import transforms
import torch

def get_metrics(pred, mask):
    pred = (pred > 0.5).float()
    pred_positives = pred.sum(dim=(2, 3))
    mask_positives = mask.sum(dim=(2, 3))
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred_positives + mask_positives
    dice = (2 * inter) / (union + 1e-6)
    iou = inter / (union - inter + 1e-6)
    acc = (pred == mask).float().mean(dim=(2, 3))
    recall = inter / (mask_positives + 1e-6)
    precision = inter / (pred_positives + 1e-6)
    f2 = (5 * inter) / (4 * mask_positives + pred_positives + 1e-6)
    mae = (torch.abs(pred - mask)).mean(dim=(2, 3))

    return [dice, iou, acc, recall, precision, f2, mae]


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # 防止除零错误，平滑参数

    def forward(self, preds, targets):
        # 将 logits 转为概率（0到1之间的值），应用 sigmoid

        # 计算交集和并集
        intersection = torch.sum(preds * targets, dim=(2, 3))  # 交集
        union = torch.sum(preds * preds, dim=(2, 3)) + torch.sum(targets * targets, dim=(2, 3))  # 并集
        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # 加上平滑项避免除零

        # Dice Loss = 1 - Dice Coefficient
        dice = torch.sum(dice) / dice.shape[0]
        dice_loss = 1 - dice
        return dice_loss



def load_sam(
    model_type="vit_h",
    train_image_encoder=False,
    train_prompt_encoder=False,
    train_mask_decoder=False,
             ):
    if model_type == "vit_b":
        sam_checkpoint = "/media/ubuntu/maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_b_01ec64.pth" 
    elif model_type == "vit_h":
        sam_checkpoint = "/media/ubuntu//maxiaochuan/CLIP_SAM_zero_shot_segmentation/segment-anything/sam_vit_h_4b8939.pth"
    else: 
        sam_checkpoint = model_type
        model_type = "vit_b"

    print(sam_checkpoint)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.cuda()
    with open("/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/parameters.txt", 'w') as f:
        for i, (param_name, param) in tqdm(enumerate(sam.named_parameters())):
            # if "image_encoder" in param_name or "prompt_encoder" in param_name:
            if not train_image_encoder and "image_encoder" in param_name:
                param.requires_grad = False
            if not train_prompt_encoder and "prompt_encoder" in param_name:
                param.requires_grad = False
            if not train_mask_decoder and "mask_decoder" in param_name:
                param.requires_grad = False
            
            f.write(f"{param_name}, {param.requires_grad} \n")
 
    return sam

def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model_type', type=str, default='vit_b', help='sam checkpoint type')
    # parser.add_argument('--root_path', type=str, default='/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/pencoder_and_decoder', help='Name of Experiment')
    # parser.add_argument('--save_model_name', type=str, default='pencoder_and_decoder')
    parser.add_argument('-tie', '--train_image_encoder', type=bool, default=False)
    parser.add_argument('-tpe', '--train_prompt_encoder', type=bool, default=False)
    parser.add_argument('-tmd', '--train_mask_decoder', type=bool, default=False)
    parser.add_argument('-e', '--max_epoch', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('-lr', '--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
    parser.add_argument('-p', '--patch_size', type=list,  default=[1080, 1080], help='patch size of network input')
    parser.add_argument('-s', '--seed', type=int,  default=2023, help='random seed')
    
    return parser.parse_args()

def train(args, snapshot_path):

    sam = load_sam(
        model_type=args.model_type, 
        train_image_encoder=args.train_image_encoder,
        train_prompt_encoder=args.train_prompt_encoder,
        train_mask_decoder=args.train_mask_decoder
    )
    save_model_name = ""
    if args.train_image_encoder: save_model_name += "iencoder"
    if args.train_prompt_encoder: save_model_name += "pencoder"
    if args.train_mask_decoder: save_model_name += "mdecoder"
    print(save_model_name)

    os.makedirs(os.path.join(snapshot_path, f"{save_model_name}"), exist_ok=True)
    logging.basicConfig(filename=snapshot_path+f"/{save_model_name}/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = get_data_loader(stage="train", batch_size=args.batch_size, original_size=args.patch_size, transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(args.patch_size),
        ToTensor(),
    ]), worker_init_fn=worker_init_fn)
    val_loader = get_data_loader(stage="val", batch_size=1, original_size=args.patch_size, transform=ToTensor())
    sam.train()
    max_iterations = args.max_epoch * len(train_loader)
    # 冻结部分参数后，使用优化器只更新需要训练的参数
    params_to_train = list(filter(lambda p: p.requires_grad, sam.parameters()))
        # 然后定义优化器

    optimizer = optim.SGD(params_to_train, lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    bce_loss = nn.BCELoss(reduction='mean')
    dice_loss = DiceLoss()
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)
    best_performance = 0.0
    iter_num = 0
    for _ in tqdm(range(args.max_epoch)):
        for t, data_batch in enumerate(train_loader):
            image = data_batch["image"].cuda()
            label = data_batch["label"].cuda()
            
            # print(image.shape) # [B, 3, 1024, 1024]
            # print(label.shape) # [B, 1024, 1024]
            # print(box.shape) # [B, 1, 4]
            batched_input = []
            H, W = args.patch_size
            for i in range(image.shape[0]):
                img = image[i].permute(1, 2, 0).detach().cpu().numpy()
                img = sam_trans.apply_image(img)
                img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1).cuda()
                box = find_bounding_boxes(label[i].detach().cpu().numpy())
                box = box[0] if box else [0, 0, H, W]
                box = np.array([box[1], box[0], box[3], box[2]])
                box = sam_trans.apply_boxes(box, original_size=(H, W))
                box = torch.Tensor(box).cuda()

                batched_input.append({
                    'image': img,
                    'original_size': (H, W),
                    'boxes': box,
                    # "mask_inputs": label[i].unsqueeze(0).unsqueeze(0),
                })

            outputs = sam(
                batched_input=batched_input,
                multimask_output=False,
            )
            outputs = torch.cat([x["masks"] for x in outputs], dim=0)
            outputs = torch.sigmoid(outputs)
            loss_bce = bce_loss(outputs, label.unsqueeze(1))
            loss_dice = dice_loss(outputs, label.unsqueeze(1))
            # print(loss_bce.shape)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            # loss = loss_bce
            # loss = loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_ 

                
            logging.info('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_bce.item(), loss_dice.item(), lr_))
            if iter_num and iter_num % 200 == 0: ## 200
                metrics = 0.0
                sam.eval()
                for _, data_batch in enumerate(val_loader):
                    image = data_batch["image"].cuda()
                    label = data_batch["label"].cuda()
                    filename = data_batch["filename"]
                    batched_input = []
                    for i in range(image.shape[0]):
                        img = image[i].permute(1, 2, 0).detach().cpu().numpy()
                        H, W = img.shape[:2]
                        img = sam_trans.apply_image(img)
                        img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1).cuda()
                        box = find_bounding_boxes(label[i].detach().cpu().numpy())
                        box = box[0] if box else [0, 0, H, W]
                        box = np.array([box[1], box[0], box[3], box[2]])
                        box = sam_trans.apply_boxes(box, original_size=(H, W))
                        box = torch.Tensor(box).cuda()
                        batched_input.append({
                            'image': img,
                            'original_size': (H, W),
                            'boxes': box,
                            # "mask_inputs": label[i].unsqueeze(0).unsqueeze(0),
                        })
                    outputs = sam(
                        batched_input=batched_input,
                        multimask_output=False,
                    )
                    outputs = torch.cat([x["masks"] for x in outputs], dim=0)
                    outputs = torch.sigmoid(outputs)
                    
                    dice = 1 - dice_loss(outputs, label.unsqueeze(1))
                    performance = dice.item()
                    print(f"{filename} dice = {performance}")
                    metrics += performance

                metrics /= len(val_loader)
                performance = metrics
                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, f"{save_model_name}", 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, f"{save_model_name}", '{}_best_model.pth'.format(save_model_name))
                    torch.save(sam.state_dict(), save_mode_path)
                    torch.save(sam.state_dict(), save_best)
                

                save_latest_path = os.path.join(snapshot_path, f"{save_model_name}",'{}_latest_model.pth'.format(save_model_name))
                torch.save(sam.state_dict(), save_latest_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, metrics))
                sam.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
  

    """ with distribute mask """ 
    # for _ in tqdm(range(args.max_epoch)):
    #     for t, data_batch in enumerate(train_loader):
    #         image = data_batch["image"].cuda()
    #         label = data_batch["label"].cuda()
            
    #         # print(image.shape) # [B, 3, 1024, 1024]
    #         # print(label.shape) # [B, 1024, 1024]
    #         # print(box.shape) # [B, 1, 4]
    #         batched_input = []
    #         for i in range(image.shape[0]):
    #             img = image[i].permute(1, 2, 0).detach().cpu().numpy()
    #             img = sam_trans.apply_image(img)
    #             img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1).cuda()
    #             box = find_bounding_boxes(label[i].detach().cpu().numpy())
    #             if not box: 
    #                 box = [0, 0, H, W]
    #             else:
    #                 box = np.array([[box[pos][1], box[pos][0], box[pos][3], box[pos][2]] for pos in range(len(box))])
    #             box = sam_trans.apply_boxes(box, original_size=(H, W))
    #             box = torch.Tensor(box).cuda()

    #             batched_input.append({
    #                 'image': img,
    #                 'original_size': (H, W),
    #                 'boxes': box,
    #                 # "mask_inputs": label[i].unsqueeze(0).unsqueeze(0),
    #             })

    #         outputs = sam(
    #             batched_input=batched_input,
    #             multimask_output=False,
    #         )
    #         outputs = torch.cat([x["masks"] for x in outputs], dim=0)
    #         print(outputs.shape)
    #         outputs = torch.sigmoid(outputs)
    #         loss_bce = bce_loss(outputs, label.unsqueeze(1))
    #         loss_dice = dice_loss(outputs, label.unsqueeze(1))
    #         # print(loss_bce.shape)
    #         loss = 0.5 * loss_bce + 0.5 * loss_dice
    #         # loss = loss_bce
    #         # loss = loss_dice
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         iter_num = iter_num + 1
    #         lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = lr_ 

                
    #         logging.info('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_bce.item(), loss_dice.item(), lr_))
    #         if iter_num > 0 and iter_num % 200 == 0: ## 200
    #             metrics = 0.0
    #             sam.eval()
    #             for _, data_batch in enumerate(val_loader):
    #                 image = data_batch["image"].cuda()
    #                 label = data_batch["label"].cuda()
    #                 filename = data_batch["filename"]
    #                 batched_input = []
    #                 for i in range(image.shape[0]):
    #                     img = image[i].permute(1, 2, 0).detach().cpu().numpy()
    #                     img = sam_trans.apply_image(img)
    #                     img = torch.as_tensor(img, dtype=torch.float).permute(2, 0, 1).cuda()
    #                     box = find_bounding_boxes(label[i].detach().cpu().numpy())
    #                     if not box: 
    #                         box = [0, 0, H, W]
    #                     else:
    #                         box = np.array([[box[pos][1], box[pos][0], box[pos][3], box[pos][2]] for pos in range(len(box))])
    #                     box = sam_trans.apply_boxes(box, original_size=(H, W))
    #                     box = torch.Tensor(box).cuda()
    #                     batched_input.append({
    #                         'image': image[i],
    #                         'original_size': image.shape[-2:],
    #                         'boxes': box[i].unsqueeze(0),
    #                         # "mask_inputs": label[i].unsqueeze(0).unsqueeze(0),
    #                     })
    #                 outputs = sam(
    #                     batched_input=batched_input,
    #                     multimask_output=False,
    #                 )
    #                 outputs = torch.cat([x["masks"] for x in outputs], dim=0)
    #                 outputs = torch.sigmoid(outputs)
                    

    #                 dice = 1 - dice_loss(outputs, label.unsqueeze(1))
    #                 performance = dice.item()
    #                 print(f"{filename} dice = {performance}")
    #                 metrics += performance

    #             metrics /= len(val_loader)
    #             performance = metrics
    #             if performance > best_performance:
    #                 best_performance = performance
    #                 save_mode_path = os.path.join(snapshot_path, f"{save_model_name}", 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
    #                 save_best = os.path.join(snapshot_path, f"{save_model_name}", '{}_best_model.pth'.format(save_model_name))
    #                 torch.save(sam.state_dict(), save_mode_path)
    #                 torch.save(sam.state_dict(), save_best)

    #             save_latest_path = os.path.join(snapshot_path, f"{save_model_name}",'{}_latest_model.pth'.format(save_model_name))
    #             torch.save(sam.state_dict(), save_latest_path)

    #             logging.info('iteration %d : mean_dice : %f' % (iter_num, metrics))
    #             sam.train()

    #         if iter_num >= max_iterations:
    #             break
    #     if iter_num >= max_iterations:
    #         break
  

    return "Training Finished!"
    
if __name__ == "__main__":
    args = config()
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"/media/ubuntu/maxiaochuan/ISICDM 2024/SAM_FineTuning/FineTuning/model"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    train(args, snapshot_path)