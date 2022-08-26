import numpy as np
import time
import copy
import scipy
from multiprocessing.pool import ThreadPool
from operator import itemgetter, mul
from scipy.spatial.distance import cdist
from torch.nn.functional import local_response_norm, normalize
from torch import Tensor
import random
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sampler import Sampler
import time
from Hyper_paras import args

if args.dataset == 'cifar10':
    class_ = 10
elif args.dataset == 'cifar100':
    class_ = 100
import os
import torch.nn as nn
nllloss = torch.nn.NLLLoss(reduction='none')
epsilon = 1e-07


class W_L(object):
    def __init__(self, args, path='./W_L'):
        self.path = path
        self.args = args
        self.W = []
        self.L = []

    def update(self, bt_W, bt_L):
        self.W.append(bt_W)
        self.L.append(bt_L)

    def save(self, mid_p):
        W = np.concatenate(self.W)
        L = np.concatenate(self.L)

        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), mid_p)

        if not os.path.exists(p):
            os.makedirs(p)

        p_w = os.path.join(p, 'weights.npy')
        p_l = os.path.join(p, 'loss.npy')
        np.save(p_w, W)
        np.save(p_l, L)

    def clean(self):
        self.W = []
        self.L = []


class Save_Models(object):
    def __init__(self, args, path='./trained'):
        self.args = args
        self.path = path

    def save(self, main_model, meta_model, mid_p):
        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), mid_p)

        if not os.path.exists(p):
            os.makedirs(p)
        meta_p = os.path.join(p, 'meta.pt')
        main_p = os.path.join(p, 'main.pt')
        torch.save(main_model.state_dict(), main_p)
        torch.save(meta_model.state_dict(), meta_p)


class Noisy_Ten_Sample_Weight_Curve(object):
    def __init__(self, args, path='./ten_noise_w'):
        self.path = path
        self.args = args
        self.weights = []

    def update(self, ten_weights):
        self.weights.append(ten_weights[None, :])

    def save(self, mid_p):
        weights = np.concatenate(self.weights, axis=0)
        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), 'meta nums' + str(args.meta_data_nums), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), 'meta nums' + str(args.meta_data_nums), mid_p)

        if not os.path.exists(p):
            os.makedirs(p)

        p_ten = os.path.join(p, 'ten_mean_ws.npy')
        np.save(p_ten, weights)

    def clean(self):
        self.weights = []


class Loss_Tools(object):
    def __init__(self, args, path='Loss'):
        self.args = args
        self.path = path
        self.train_losses = []
        self.meta_losses = []

    def update(self, train_loss_i, meta_loss_i):
        self.train_losses.append(train_loss_i)
        self.meta_losses.append(meta_loss_i)

    def clean(self):
        self.train_losses = []
        self.meta_losses = []

    def save(self, mid_p):
        self.train_losses = np.array(self.train_losses)
        self.meta_losses = np.array(self.meta_losses)

        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), 'meta nums' + str(args.meta_data_nums), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), 'meta nums' + str(args.meta_data_nums), mid_p)

        if not os.path.exists(p):
            os.makedirs(p)
        p_train_loss = os.path.join(p, 'train_losses.npy')
        p_test_loss = os.path.join(p, 'meta_losses.npy')
        np.save(p_train_loss, self.train_losses)
        np.save(p_test_loss, self.meta_losses)


class Acc_curve_Tools(object):
    def __init__(self, args, path='ACC'):
        self.path = path
        self.args = args
        self.test_Accs = []
        self.train_Accs = []

    def update(self, train_acc_i, test_acc_i):
        self.train_Accs.append(train_acc_i)
        self.test_Accs.append(test_acc_i)

    def save(self, mid_p):

        self.train_Accs = np.array(self.train_Accs)
        self.test_Accs = np.array(self.test_Accs)

        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), 'meta nums' + str(args.meta_data_nums), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), 'meta nums' + str(args.meta_data_nums), mid_p)

        if not os.path.exists(p):
            os.makedirs(p)

        p_train_acc = os.path.join(p, 'train_accs.npy')
        p_test_acc = os.path.join(p, 'test_accs.npy')
        np.save(p_train_acc, self.train_Accs)
        np.save(p_test_acc, self.test_Accs)

    def clean(self):
        self.train_Accs = []
        self.test_Accs = []


class Con_Mat_Tools(object):
    def __init__(self, args, path='./Con_Mat'):
        self.args = args
        self.path = path

    def save(self, pred_label, true_label, mid_p):

        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), 'meta nums' + str(args.meta_data_nums), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), 'meta nums' + str(args.meta_data_nums), mid_p)
        if not os.path.exists(p):
            os.makedirs(p)

        p_pred = os.path.join(p, 'pred_label.npy')
        p_true = os.path.join(p, 'true_label.npy')
        np.save(p_pred, pred_label)
        np.save(p_true, true_label)


class Weight_Tools(object):
    def __init__(self, args, path='./weight'):
        self.path = path
        self.weight = []
        self.ori_labels = []
        self.noi_labels = []
        self.args = args

    def update(self, weight, ori_l, noi_l):
        self.weight.append(weight)
        self.ori_labels.append(ori_l)
        self.noi_labels.append(noi_l)

    def save(self, mid_p):
        self.weight = np.concatenate(self.weight)
        self.ori_labels = np.concatenate(self.ori_labels)
        self.noi_labels = np.concatenate(self.noi_labels)

        if self.args.imbalanced_factor is not None:
            p = os.path.join(self.path, self.args.model, self.args.dataset, 'imbalance', str(
                self.args.imbalanced_factor), 'meta nums' + str(args.meta_data_nums), mid_p)
        else:
            p = os.path.join(self.path, self.args.model, self.args.dataset,
                             self.args.corruption_type, str(self.args.corruption_ratio), 'meta nums' + str(args.meta_data_nums), mid_p)
        if not os.path.exists(p):
            os.makedirs(p)

        p_w = os.path.join(p, 'weights.npy')
        p_o = os.path.join(p, 'ori_label.npy')
        p_n = os.path.join(p, 'noi_label.npy')
        np.save(p_w, self.weight)
        np.save(p_o, self.ori_labels)
        np.save(p_n, self.noi_labels)

    def clean(self):
        self.weight = []
        self.noi_labels = []
        self.ori_labels = []

class MixUp_DS(Dataset):
    def __init__(self, train_data, args, transforms):
        super(MixUp_DS, self).__init__()
        self.datas = train_data.train_data
        self.labels = train_data.train_labels
        self.imbal_list = train_data.imbalanced_num_list
        self.args = args
        self.transforms = transforms
        self.datas = self.transform_img()
        self.train_data, self.train_label = self.mixup(self.imbal_list)        


    def transform_img(self):
        new_data = []
        for i in range(len(self.labels)):
            img = self.datas[i]
            img = Image.fromarray(img)
            img = self.transforms(img)
            img = img.unsqueeze(0)
            new_data.append(img)
        new_data = torch.cat(new_data, dim = 0).detach().numpy()
        return new_data

    def mixup(self, imbal_list, max_c = args.max_c, lam = args.mixup_lambda):
        mixup_data = []
        mixup_label = []
        class_nums = len(self.imbal_list)
        idx = list(range(len(self.labels)))
        pre_S = 0
        for c in range(class_nums):
            nums_c = self.imbal_list[c]
            idx_c = idx[pre_S : pre_S + nums_c]
            pre_S += nums_c
            data_c = self.datas[idx_c]
            if nums_c < max_c:
                mul_factor = max_c // nums_c
                np.random.shuffle(idx_c)
                mixup_idxs1 = idx_c * (mul_factor - 1) + idx_c[:max_c - mul_factor * nums_c]
                mixup_data1_c = data_c[mixup_idxs1]
                mixup_idxs2 = np.random.choice(idx, size = len(mixup_idxs1), replace = False)
                mixup_data2_c = self.datas[mixup_idxs2]
                mixup_data_c = (1 - lam) * mixup_data1_c + lam * mixup_data2_c
                new_data = np.concatenate([data_c, mixup_data_c], axis = 0)
                mixup_data.append(new_data)
                mixup_label += [c] * max_c

            else:
                sub_idx = np.random.choice(nums_c, max_c, replace = True)
                sub_data_c = data_c[sub_idx]
                mixup_data.append(sub_data_c)
                mixup_label += [c] * max_c
        mixup_data = np.concatenate(mixup_data, axis = 0)
        return mixup_data, mixup_label

    def __len__(self):
        return len(self.train_label)

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index], index

class Prob_Utility(object):
    def __init__(self, dataset) -> None:
        self.prob_arr = np.zeros(shape = (len(dataset),), dtype = np.float32)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size = 100, shuffle = False)

    def update(self, net):
        net.eval()
        for bt_idx, (input, target, index) in enumerate(self.dataloader):
            input = input.cuda()
            target = target.cuda()
            outputs, _ = net(input)
            probs = F.softmax(outputs, dim = 1)
            bt_probs = -nllloss(probs, target).detach().cpu().numpy()
            self.prob_arr[index] = bt_probs
        net.train()

    def utilitys(self, index):
        probs = self.prob_arr[index]
        return probs


class SubModSampler(Sampler):
    def __init__(self, get_prob, model, dataset, batch_size, ltl_log_ep=5):
        super(SubModSampler, self).__init__(model, dataset)
        self.batch_size = batch_size
        # It contains the indices of each image of the set.
        self.index_set = list(range(0, len(self.dataset)))
        self.ltl_log_ep = ltl_log_ep
        self.initialize_with_activations()
        self.get_prob = get_prob

    def update_activations(self, model):
        self.final_activations = []
        self.penultimate_activations = []
        self.set_activations_from_model(model)
        self.initialize_with_activations()

    def initialize_with_activations(self):
        # Setup entropy
        f_acts = torch.tensor(self.final_activations)
        p_log_p = F.softmax(f_acts, dim=1) * F.log_softmax(f_acts, dim=1)
        H = -p_log_p.numpy()
        # Compute entropy of all samples for an epoch.
        self.H = np.sum(H, axis=1)

        # Setup penultimate activations
        self.penultimate_activations = np.array(self.penultimate_activations)
        penultimate_activations = torch.tensor(self.penultimate_activations)
        relu = torch.nn.ReLU(inplace=True)
        penultimate_activations = relu(penultimate_activations)

        softmax = torch.nn.Softmax()
        self.normalised_penultimate_activations = softmax(
            penultimate_activations).numpy()

    def get_subset(self):
        set_size = len(self.index_set)
        num_of_partitions = args.num_of_partitions

        if set_size >= num_of_partitions*self.batch_size:
            size_of_each_part = set_size // num_of_partitions
            r_size = (size_of_each_part * self.ltl_log_ep) // self.batch_size

            random.shuffle(self.index_set)
            partitions = [self.index_set[k:k+size_of_each_part]
                          for k in range(0, size_of_each_part * num_of_partitions, size_of_each_part)]

            pool = ThreadPool(processes=len(partitions))
            pool_handlers = []
            for partition in partitions:
                handler = pool.apply_async(get_subset_indices_bal, args=(self.get_prob, partition, self.penultimate_activations, self.normalised_penultimate_activations,
                                                                     self.H, self.batch_size, r_size, self.dataset))
                pool_handlers.append(handler)

            pool.close()
            pool.join()

            intermediate_indices = []
            for handler in pool_handlers:
                intermediate_indices.extend(handler.get())
        else:
            intermediate_indices = self.index_set


        r_size = len(intermediate_indices) / self.batch_size * self.ltl_log_ep
        subset_indices = get_subset_indices_bal(self.get_prob, intermediate_indices, self.penultimate_activations, self.normalised_penultimate_activations, self.H,
                                                self.batch_size, r_size, self.dataset)

        for item in subset_indices:     # Subset selection without replacement.
            self.index_set.remove(item)
        return subset_indices



def get_middle_uncert(entropy, idx_set):

    u_scores = entropy[idx_set.tolist()]
    d = dict(zip(u_scores, idx_set))
    both = [[a, b] for (a, b) in zip(u_scores, idx_set)]

    both = sorted(both, key=lambda x: x[0])
    u_scores = np.array([x[0] for x in both])
    idx_set = np.array([x[1] for x in both])

    len_ = len(idx_set)
    idx_set = idx_set[int(len_ * 1 / 4): int(len_ * 3 / 4)]
    u_scores = u_scores[int(len_ * 1 / 4): int(len_ * 3 / 4)]

    return u_scores, idx_set, d


def get_clean_samples(losses, idx, clean_ratio):
    d = dict(zip(idx, losses))
    all_nums = len(idx)
    num_clean = int(all_nums * clean_ratio)
    loss_map = dict(sorted(d.items(), key=lambda x: x[1]))
    clean_idx = list(loss_map.keys())[:num_clean]
    return np.array(clean_idx)


def get_subset_indices_bal(get_prob,
                           index_set_input,
                           penultimate_activations,
                           normalised_penultimate_activations,
                           entropy,
                           subset_size,
                           r_size,
                           dataset,
                           ):
    '''
    this method is to select a balance Meta Data!!!
    '''

    index_set_input = copy.deepcopy(index_set_input)
    r_size = int(r_size)
    # Subset of indices. Keeping track to improve computational performance.
    subset_indices = []

    class_mean = np.mean(penultimate_activations, axis=0)
    subset_size = min(subset_size, len(index_set_input))

    upper = subset_size // class_

    # record numbers per class
    now_nums = dict(zip(list(range(class_)), [0]*class_))
    now_nums1 = dict(zip(list(range(class_)), [0]*class_))
    for idx in index_set_input:
        data_class = dataset[idx][1]
        now_nums1[data_class] += 1

    while True:
        if r_size < len(index_set_input) and args.use_ltlg:
            index_set = np.random.choice(
                index_set_input, r_size, replace=False)
        else:
            index_set = copy.deepcopy(index_set_input)

        index_set = np.array(index_set)

        sort_u, sort_idx, d = get_middle_uncert(entropy, index_set)

        if len(sort_idx) == 0:
            sort_idx = index_set
        
        prob_scores = get_prob.utilitys(sort_idx)

        coverage_scores = compute_coverage_score(
            normalised_penultimate_activations, subset_indices, sort_idx)

        scores =  args.alpha_4 * normalise(np.array(coverage_scores)) + args.alpha_5 * normalise(prob_scores)

        best_item_index = np.argmax(scores)

        data_idx = index_set[best_item_index]
        data = dataset[data_idx]

        data_class = data[1]
        if now_nums[data_class] >= upper:  
            for idx in index_set_input[:]:
                if dataset[idx][1] == data_class:
                    index_set_input.remove(idx)
        else: 
            subset_indices.append(index_set[best_item_index])
            now_nums[data_class] += 1
            index_set_input.remove(index_set[best_item_index])

        if sum(now_nums.values()) == subset_size or len(index_set_input) == 0:
            break

    return subset_indices


def get_subset_indices(index_set_input,
                       penultimate_activations,
                       normalised_penultimate_activations,
                       entropy,
                       subset_size,
                       r_size,
                       ):

    r_size = int(r_size)
    # Subset of indices. Keeping track to improve computational performance.
    subset_indices = []
    class_mean = np.mean(penultimate_activations, axis=0)
    subset_size = min(subset_size, len(index_set_input))

    for i in range(0, subset_size):

        if r_size < len(index_set_input) and args.use_ltlg:
            index_set = np.random.choice(
                index_set_input, r_size, replace=False)
        else:
            index_set = copy.deepcopy(index_set_input)

        sort_u, sort_idx, d = get_middle_uncert(entropy, index_set)

        u_scores = compute_u_score(entropy, list(sort_idx))

        r_scores = compute_r_score(
            penultimate_activations, list(subset_indices), list(sort_idx))

        md_scores = compute_md_score(
            penultimate_activations, list(sort_idx), class_mean)

        coverage_scores = compute_coverage_score(
            normalised_penultimate_activations, subset_indices, sort_idx)

        scores = args.alpha_1 * normalise(np.array(u_scores)) + args.alpha_2 * normalise(np.array(r_scores)) + args.alpha_3 * normalise(
            np.array(md_scores)) + args.alpha_4 * normalise(np.array(coverage_scores))

        best_item_index = np.argmax(scores)
        subset_indices.append(index_set[best_item_index])
        index_set_input.remove(index_set[best_item_index])

        # log('Processed: {0}/{1} exemplars. Time taken is {2} sec.'.format(i, subset_size, time.time()-now))

    return subset_indices


def normalise(A):
    s = np.sum(A)
    if s == 0:
        s = 1
    return A/s


def compute_u_score(entropy, index_set):
    """
    Compute the Uncertainity Score: The point that makes the model most confused, should be preferred.
    :param final_activations:
    :param subset_indices:
    :param alpha:
    :return: u_score
    """

    if len(index_set) == 0:
        return 0
    else:
        u_scores = entropy[index_set]
        return u_scores


def compute_r_score(penultimate_activations, subset_indices, index_set, distance_metric=args.distance_metric):
    """
    Computes Redundancy Score: The point should be distant from all the other elements in the subset.
    :param penultimate_activations:
    :param subset_indices:
    :param alpha:
    :return:
    """
    if len(subset_indices) == 0:
        return 0
    else:
        index_p_acts = penultimate_activations[np.array(index_set)]
        subset_p_acts = penultimate_activations[np.array(subset_indices)]
        if(distance_metric == 'gaussian'):
            pdist = cdist(index_p_acts, subset_p_acts, metric='sqeuclidean')
            r_score = scipy.exp(-pdist / (0.5) ** 2)
            r_score = np.min(r_score, axis=1)
            return r_score
        else:
            pdist = cdist(index_p_acts, subset_p_acts, metric=distance_metric)
            r_score = np.min(pdist, axis=1)
            return r_score


def compute_md_score(penultimate_activations, index_set, class_mean, distance_metric=args.distance_metric):
    """
    Computes Mean Divergence score: The new datapoint should be close to the class mean
    :param penultimate_activations:
    :param index_set:
    :param class_mean:
    :param alpha:
    :return: list of scores for each index item
    """
    #print('index_set in md is :', len(index_set))
    if(distance_metric == 'gaussian'):
        pen_act = penultimate_activations[np.array(index_set)]
        md_score = cdist(pen_act, np.array(
            [np.array(class_mean)]), metric='sqeuclidean')
        md_score = scipy.exp(-md_score / (0.5) ** 2)
        return md_score
    else:

        pen_act = penultimate_activations[np.array(index_set)]
        md_score = cdist(pen_act, np.array(
            [np.array(class_mean)]), metric=distance_metric)
        md_score = np.squeeze(md_score, 1)
        return -md_score


def compute_coverage_score(normalised_penultimate_activations, subset_indices, index_set):
    """
    :param penultimate_activations:
    :param subset_indices:
    :param index_set:
    :return: g(mu(S))
    """
    if(len(subset_indices) == 0):
        penultimate_activations_index_set = normalised_penultimate_activations[index_set]
        score_feature_wise = np.sqrt(penultimate_activations_index_set)
        scores = np.sum(score_feature_wise, axis=1)
        return scores
    else:
        penultimate_activations_index_set = normalised_penultimate_activations[index_set]
        subset_indices_scores = np.sum(
            normalised_penultimate_activations[subset_indices], axis=0)
        sum_subset_index_set = subset_indices_scores + penultimate_activations_index_set
        score_feature_wise = np.sqrt(sum_subset_index_set)
        scores = np.sum(score_feature_wise, axis=1)
        return scores


def get_batch_meta(model, data_source, batch_size, sampler=None):
    submodular_sampler = SubModSampler(
        model, data_source, batch_size, args.ltl_log_ep)
    batch_meta = submodular_sampler.get_subset()
    return batch_meta


def get_data(idxs, train_data):
    img = []
    target = []
    for idx in idxs:
        img.append(train_data[idx][0].unsqueeze(0))
        target.append(train_data[idx][1])
    img = torch.cat(img, dim=0)
    target = torch.tensor(target).long()
    return img, target


def cal_correct_nums(datalaoder):
    sum_ = 0
    for bt_id, (img, noisy_tar, ori_tar) in enumerate(datalaoder):
        equal_or = (noisy_tar == ori_tar).long()
        sum_ += torch.sum(equal_or).numpy()
    return sum_
