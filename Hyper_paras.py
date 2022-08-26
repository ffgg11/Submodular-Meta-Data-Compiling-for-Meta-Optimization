# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 21:27:38 2021

@author: DELL
"""
import argparse

parser = argparse.ArgumentParser(description='Imbalanced Example')

parser.add_argument('--nums', type=int, default=1)
parser.add_argument('--max_c', type=int, default=1000)
parser.add_argument('--meta_data_interval', type=int, default=4)
parser.add_argument('--meta_data_nums', type=int, default=1)
parser.add_argument('--device_idx', default='0', type=str)



parser.add_argument('--prefetch', type=int, default=10, help='Pre-fetching threads.')
parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)


parser.add_argument('--override_submodular_sampling', type=bool, default=False)
parser.add_argument('--ltl_log_ep', type=int, default=5)
parser.add_argument('--num_of_partitions', type=int, default=1)

parser.add_argument('--gamma', type=float, default=3.0)
parser.add_argument('--mixup_lambda', type=float, default=0.01)

parser.add_argument('--alpha_1', type=float, default=0.0)
parser.add_argument('--alpha_2', type=float, default=0.0)
parser.add_argument('--alpha_3', type=float, default=0.0)
parser.add_argument('--alpha_4', type=float, default=0.9)
parser.add_argument('--alpha_5', type=float, default=0.1)
parser.add_argument('--use_ltlg', type=bool, default=True)

parser.add_argument('--distance_metric', type=str, default='euclidean')

parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.005)
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

args = parser.parse_args()

