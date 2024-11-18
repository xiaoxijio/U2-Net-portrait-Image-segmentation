import argparse
import glob
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import SalObjDataset, RescaleT, RandomCrop, ToTensorLab
from model.u2net import U2NET, U2NETP

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """ 将多个输出的loss相加，类似监督学习 """
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


def train(opt, net, salobj_dataloader):
    for epoch in range(opt.epoch_num):
        net.train()
        ite_num = 0
        running_loss = 0.0
        running_tar_loss = 0.0

        pbar = tqdm(total=len(salobj_dataloader))
        for i, data in enumerate(salobj_dataloader):
            ite_num += 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 总loss
            running_tar_loss += loss2.item()  # 输出loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            s = ("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, opt.epoch_num, (i + 1) * opt.batch_size_train, opt.train_num, ite_num,
                running_loss / ite_num, running_tar_loss / ite_num))
            pbar.set_description(s)
            pbar.update(1)

        torch.save(net.state_dict(), opt.model_dir + opt.model_name + ".pth")
        pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='u2net', help='模型名称')
    parser.add_argument('--epoch_num', default=1, type=int, help='训练回合数')
    parser.add_argument('--batch_size_train', default=6, type=int, help='训练batch_size')
    parser.add_argument('--batch_size_test', default=1, type=int, help='测试batch_size')
    parser.add_argument('--tra_image_dir', type=str, default='data/DUTS-TR/DUTS-TR-Image/', help='图片路径')
    parser.add_argument('--tra_label_dir', type=str, default='data/DUTS-TR/DUTS-TR-Mask/', help='标签路径')
    parser.add_argument('--image_ext', default='.jpg', help='image file extension')
    parser.add_argument('--label_ext', default='.png', help='mask file extension')
    opt = parser.parse_args()

    opt.tra_img_name_list = glob.glob(opt.tra_image_dir + '*' + opt.image_ext)
    opt.tra_lbl_name_list = glob.glob(opt.tra_label_dir + '*' + opt.label_ext)
    opt.model_dir = os.path.join(os.getcwd(), 'saved_models', opt.model_name + os.sep)

    print("---")  # 这里并没有检查label和image是否一一对应，所以要确保你的数据没有问题哦
    print("train images: ", len(opt.tra_img_name_list))
    print("train labels: ", len(opt.tra_lbl_name_list))
    print("---")

    opt.train_num = len(opt.tra_img_name_list)

    salobj_dataset = SalObjDataset(img_name_list=opt.tra_img_name_list, lbl_name_list=opt.tra_lbl_name_list,
                                   transform=transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]))

    salobj_dataloader = DataLoader(salobj_dataset, batch_size=opt.batch_size_train, shuffle=True, num_workers=0)

    if opt.model_name == 'u2net':
        net = U2NET(3, 1)
    elif opt.model_name == 'u2netp':
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.cuda()

    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print("---start training...")
    train(opt, net, salobj_dataloader)
