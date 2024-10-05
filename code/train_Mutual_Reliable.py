import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/home/jwsu/semi/', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='MCNet', help='exp_name')
parser.add_argument('--model', type=str, default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')

args = parser.parse_args()

snapshot_path = "/mnt/imtStu/jwsu/UncertaintyPixel_Rebuttal/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp,
                                                                                   args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Pancreas/'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
 
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=4,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    pixel_criterion = losses.ce_loss_mask
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.Binary_dice_loss

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs, outfeats = model(volume_batch)
            num_outputs = len(outputs)

            y_all = torch.zeros((num_outputs, ) + outputs[0].shape).cuda()

            # loss_seg = 0
            loss_seg_dice = 0
            for idx in range(num_outputs):
                y = outputs[idx]
                y_prob = F.softmax(y, dim=1)
                # loss_seg += F.cross_entropy(y_prob[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:labeled_bs, 1, ...], label_batch[:labeled_bs, ...] == 1)

                y_all[idx] = y_prob

            loss_consist = 0
            pixel_consist = 0

            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        uncertainty_o1 = -1.0 * torch.sum(y_all[i] * torch.log(y_all[i] + 1e-6), dim=1)
                        uncertainty_o2 = -1.0 * torch.sum(y_all[j] * torch.log(y_all[j] + 1e-6), dim=1)
                        mask = (uncertainty_o1 > uncertainty_o2).float()

                        batch_o, c_o, w_o, h_o, d_o = y_all[j].shape
                        batch_f, c_f, w_f, h_f, d_f = outfeats[j].shape

                        teacher_o = y_all[j].reshape(batch_o, c_o, -1)
                        teacher_f = outfeats[j].reshape(batch_f, c_f, -1)
                        stu_f = outfeats[i].reshape(batch_f, c_f, -1)

                        index = torch.argmax(y_all[j], dim=1, keepdim=True)
                        prototype_bank = torch.zeros(batch_f, num_classes, c_f).cuda()
                        for ba in range(batch_f):
                            for n_class in range(num_classes):
                                mask_temp = (index[ba] == n_class).float()
                                top_fea = outfeats[j][ba] * mask_temp
                                prototype_bank[ba, n_class] = top_fea.sum(-1).sum(-1).sum(-1) / (mask_temp.sum() + 1e-6)

                        prototype_bank = F.normalize(prototype_bank, dim=-1)
                        mask_t = torch.zeros_like(y_all[i]).cuda()
                        for ba in range(batch_o):
                            for n_class in range(num_classes):
                                class_prototype = prototype_bank[ba, n_class]
                                mask_t[ba, n_class] = F.cosine_similarity(teacher_f[ba],
                                                                          class_prototype.unsqueeze(1),
                                                                          dim=0).view(w_f, h_f, d_f)

                        weight_pixel_t = (1 - nn.MSELoss(reduction='none')(mask_t, y_all[j])).mean(1)
                        weight_pixel_t = weight_pixel_t * mask

                        loss_t = consistency_criterion(outputs[i], torch.argmax(y_all[j], dim=1).detach())
                        loss_consist += (loss_t * weight_pixel_t.detach()).sum() / (mask.sum() + 1e-6)

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = args.lamda * loss_seg_dice + consistency_weight * (loss_consist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' %
                         (iter_num, loss, loss_seg_dice, loss_consist))

            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            # writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=18,
                                                          stride_z=4,
                                                          dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break

    writer.close()
