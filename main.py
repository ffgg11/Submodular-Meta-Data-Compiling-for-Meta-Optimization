import os
from Hyper_paras import args
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.device_idx

import time
import argparse
import random
import copy
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from submodular_batch_sampler import SubmodularBatchSampler
from torchvision_trans import Aug_method
from submodular import Prob_Utility
from data_utils import *
from resnet import *
import shutil




def main():
    best_prec1 = 0.0
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))

    kwargs = {'num_workers': 8, 'pin_memory': True}
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_data_meta,train_data,train_data_ori, test_dataset = build_dataset(args.dataset,args.num_meta)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader_ori = torch.utils.data.DataLoader(
            train_data_ori, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    np.random.seed(42)
    random.seed(42)
    # make imbalanced data
    torch.manual_seed(args.seed)
    classe_labels = range(args.num_classes)

    data_list = {}


    for j in range(args.num_classes):
        data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]


    img_num_list = get_img_num_per_cls(args.dataset,args.imb_factor, args.num_meta*args.num_classes)
    print(img_num_list)
    print(sum(img_num_list))

    #criterion = LDAMLoss(img_num_list)
    #criterion = FocalLoss(gamma = args.gamma)

    im_data = {}
    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        im_data[cls_idx] = img_id_list[img_num:]
        idx_to_del.extend(img_id_list[img_num:])

    print(len(idx_to_del))

    imbalanced_train_dataset = copy.deepcopy(train_data)
    imbalanced_train_dataset_ori = copy.deepcopy(train_data_ori)

    imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
    imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
    imbalanced_train_loader = torch.utils.data.DataLoader(
        imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    imbalanced_train_dataset_ori.targets = np.delete(train_loader_ori.dataset.targets, idx_to_del, axis=0)
    imbalanced_train_dataset_ori.data = np.delete(train_loader_ori.dataset.data, idx_to_del, axis=0)

    print('len(imbalanced_train_loader.dataset.targets) is :', len(imbalanced_train_loader.dataset.targets))



    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    best_prec1 = 0

    beta = 0.9999
    effective_num = 1.0 - np.power(beta, img_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    aug_method = Aug_method(imbalanced_train_dataset_ori, img_num_list)
    get_prob = Prob_Utility(aug_method)

    # create model
    model = build_model()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer

    # print("=> loading checkpoint")
    # checkpoint = torch.load('/home/muhammad/Reweighting_samples/Meta-weight-net_class-imbalance-master/checkpoint/ours/ckpt.best.pth.tar', map_location='cuda:0')
    # args.start_epoch = checkpoint['epoch']
    # best_acc1 = checkpoint['best_acc1']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer_a.load_state_dict(checkpoint['optimizer'])
    # print("=> loaded checkpoint")

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        if epoch < 160:

            train(imbalanced_train_loader, model, optimizer_a,epoch)

        else:
            if epoch % args.meta_data_interval == 0:
                print('start getting meta data!!!')
                get_prob.update(model)
                batch_sampler = SubmodularBatchSampler(get_prob, model, aug_method, 100)
                meta_loader = torch.utils.data.DataLoader(aug_method, batch_sampler=batch_sampler,
                                                            num_workers=10)
                meta_data = []
                meta_target = []

                for bt_id, (img, target, index) in enumerate(meta_loader):
                    if bt_id >= 10:
                        break;
                    meta_data.append(img)
                    meta_target.append(target)
                
                meta_data = torch.cat(meta_data, dim = 0)
                meta_target = torch.cat(meta_target)

                meta_ds = TensorDataset(meta_data, meta_target)            
                meta_loader = DataLoader(meta_ds, batch_size=100, shuffle=True, num_workers=10)

            train_meta(imbalanced_train_loader, meta_loader, model, optimizer_a,epoch, per_cls_weights)

       
        #tr_prec1, tr_preds, tr_gt_labels = validate(imbalanced_train_loader, model, criterion, epoch)
        # evaluate on validation set
        prec1, preds, gt_labels = validate(test_loader, model, nn.CrossEntropyLoss().cuda(), epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)



    print('Best accuracy: ', best_prec1)
    with open('./results.txt', 'a+') as f:
        f.write('imbal factor is :' + str(args.imb_factor) + '\n')
        f.write('best acc is : ' + str(best_prec1) + '\n')
        f.write('++++++++++\n')
        f.write('++++++++++\n')
        f.write('++++++++++\n')
        f.write('++++++++++\n')


def train(train_loader, model,optimizer_a,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    weight_eps_class = [0 for _ in range(int(args.num_classes))]
    total_seen_class = [0 for _ in range(int(args.num_classes))]
    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        #meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        #meta_model.load_state_dict(model.state_dict())

        #meta_model.cuda()

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f, _ = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w) # * w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        #meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        #meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def train_meta(train_loader, validation_loader,model,optimizer_a,epoch, per_cls_weights):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    meta_iter = iter(validation_loader)

    weight_eps_class = [0 for _ in range(int(args.num_classes))]
    total_seen_class = [0 for _ in range(int(args.num_classes))]
    batch_w_eps = []
    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        target_var = target_var.cpu()

        #import pdb; pdb.set_trace()
        y = torch.eye(args.num_classes)

        labels_one_hot = y[target_var].float().cuda()

        weights = torch.tensor(per_cls_weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        #weights = weights.unsqueeze(1)
        #weights = weights.repeat(1,args.num_classes)

        meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
        meta_model.load_state_dict(model.state_dict())

        meta_model.cuda()

        # compute output
        # Lines 4 - 5 initial forward pass to compute the initial weighted loss

        y_f_hat, _ = meta_model(input_var)


        target_var = target_var.cuda()
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
        
        weights = to_var(weights)
        eps = to_var(torch.zeros(cost.size()))

        w_pre = weights + eps

        l_f_meta = torch.sum(cost * w_pre)

        meta_model.zero_grad()



        # Line 6-7 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)
        #del grads

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        #input_validation, target_validation = next(iter(validation_loader))
        try:
            input_validation_var, target_validation_var = next(meta_iter)
        except StopIteration:
            meta_iter = iter(validation_loader)
            input_validation_var, target_validation_var = next(meta_iter)

        input_validation_var = to_var(input_validation_var, requires_grad=False)
        target_validation_var = to_var(target_validation_var, requires_grad=False)

        #import pdb; pdb.set_trace()
        y_g_hat, _ = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_metada = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

  
       # import pdb; pdb.set_trace()
        new_eps = eps - 0.01 * grad_eps
        w = weights + new_eps

        del grad_eps, grads

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f, _ = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)

        l_f = torch.mean(cost_w * w)

        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        # meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        #import pdb; pdb.set_trace()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()#async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output, _ = model(input_var)
        loss = criterion(output, target_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # import pdb; pdb.set_trace()

    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 10 at 160th, and 180th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    #lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate_v1(oargs, optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if epoch <= 5:
#         lr = args.lr * epoch / 5
#     elif epoch > 180:
#         lr = args.lr * 0.0001
#     elif epoch > 160:
#         lr = args.lr * 0.01
#     else:
#         lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % ('checkpoint', 'ours')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

if __name__ == '__main__':
    for imbal in [200, 100, 50, 20, 10]:
        args.imb_factor = 1.0 / imbal
        for i in range(3):
            main()
