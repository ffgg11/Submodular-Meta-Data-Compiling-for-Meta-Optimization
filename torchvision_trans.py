# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:22:50 2021

@author: DELL
"""

import random
import numpy as np
import argparse
from Hyper_paras import args
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageOps

parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--resize_list', type = list, default = [40, 45])
parser.add_argument('--crop_size', type = int, default = 32)
parser.add_argument('--flip_H_prob', type = float, default = 1.0)
aug_args = parser.parse_args()

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

import albumentations as A
trans_color = A.ColorJitter(always_apply=True)

class Aug_method(Dataset):
    def __init__(self, dataset, img_num_list, args = args, aug_args = aug_args, train_transforms = transform_train):
        super(Aug_method, self).__init__()
        self.aug_args = aug_args
        self.args = args
        self.train_transforms = train_transforms
        self.datas = dataset.data
        self.labels = dataset.targets
        self.imbal_list = img_num_list
        self.datas, self.labels = self.sort_datas_by_class()
        
        self.datas, self.labels = self.aug_images()
        self.datas = self.transform()

    def sort_datas_by_class(self):
        sort_imgs = []
        sort_labels = []
        nums_c = len(self.imbal_list)
        for c in range(nums_c):
            idx_c = []
            for idx, l in enumerate(self.labels):
                if l == c:
                    idx_c.append(idx)
            imgs_c = self.datas[idx_c]
            sort_imgs.append(imgs_c)
            sort_labels.extend([c] * self.imbal_list[c])
        sort_imgs = np.concatenate(sort_imgs, axis = 0)
        return sort_imgs, sort_labels

    def resize(self, img):
        resize_imgs = []
        for size in self.aug_args.resize_list:
            resize_op = transforms.Resize(size)
            resize_img = resize_op(img)
            resize_imgs.append(resize_img)
        return resize_imgs
            
    def five_crop(self, img):
        resize_imgs = self.resize(img)
        crop_imgs = []
        five_op = transforms.FiveCrop(self.aug_args.crop_size)
        for resize_imgs_i in resize_imgs:
            five_imgs = five_op(resize_imgs_i)
            for img_i in five_imgs:
                crop_imgs.append(img_i)
        return crop_imgs
    
    def Flip_H(self, imgs):
        flip_h = transforms.RandomHorizontalFlip(p = self.aug_args.flip_H_prob)
        F = []
        for img in imgs:
            img_flip_h = flip_h(img)
            F.append(img_flip_h)
        return F
    
    def Flip_V(self, imgs):
        flip_V = transforms.RandomVerticalFlip(p = self.aug_args.flip_H_prob)
        V = []
        for img in imgs:
            img_flip_v = flip_V(img)
            V.append(img_flip_v)
        return V
    
    def color_jetter(self, imgs):
        color_jetter_imgs = []
        for img in imgs:
            img = np.array(img)
            for i in range(5):
                img_i = trans_color(image = img)['image']
                color_jetter_imgs.append(img_i)
        return color_jetter_imgs
    
    def aug_images(self):
        max_c = self.args.max_c
        images = []
        labels = []
        class_nums = len(self.imbal_list)
        idx = list(range(len(self.labels)))
        pre_S = 0
        
        for c in range(class_nums):
            print('now c is :', c)
            nums_c = self.imbal_list[c]
            idx_c = idx[pre_S : pre_S + nums_c]

            pre_S += nums_c
 
            data_c = self.datas[idx_c]
            if nums_c < max_c:

                data_c_aug = []
                for i in range(nums_c):

                    img_c_i = data_c[i]

                    img_c_i = Image.fromarray(img_c_i)
  
                    FLIP_and_ORI = []
                    five_crop_img_c_i = self.five_crop(img_c_i)
                    F_H = self.Flip_H(five_crop_img_c_i)
                    F_V = self.Flip_V(five_crop_img_c_i)
                    
                    FLIP_and_ORI.extend(five_crop_img_c_i)
                    FLIP_and_ORI.extend(F_H)
                    FLIP_and_ORI.extend(F_V)
                    
                    data_aug_c_i = self.color_jetter(FLIP_and_ORI)

                    data_aug_c_i.extend(FLIP_and_ORI)
                    data_c_aug.extend(data_aug_c_i)

                data_c_aug_np = []
                for i in range(len(data_c_aug)):
                    img_aug_i = data_c_aug[i]
                    img_numpy = np.array(img_aug_i)[None,:,:,:]
                    data_c_aug_np.append(img_numpy)

                data_c_aug_np = np.concatenate(data_c_aug_np, axis = 0)
                sub_idx = np.random.choice(len(data_c_aug_np), size = max_c, replace=False)
                images.append(data_c_aug_np[sub_idx])
                labels.extend([c] * max_c)

            else:

                sub_idx = np.random.choice(nums_c, size = max_c, replace = False)
                sub_data_c = data_c[sub_idx]
                images.append(sub_data_c)
                labels.extend([c] * max_c)
        images = np.concatenate(images, axis = 0)
        return images, labels
    
    def transform(self):
        trans_imgs = []
        for i in range(len(self.labels)):
            img_i = self.datas[i]
            img_i = Image.fromarray(img_i)
            img_i = self.train_transforms(img_i)
            img_i = img_i.unsqueeze(0)
            trans_imgs.append(img_i)
        trans_imgs = torch.cat(trans_imgs,dim = 0).detach().numpy()
        return trans_imgs
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.datas[index], self.labels[index], index


class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
                           'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 231, 10),
            "contrast": np.linspace(0.0, 0.5, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.3, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,           
            "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

    def test_single_operation(self, image, op_name, M=-1):
        '''
        :param image: image
        :param op_name: operation name in   self.transforms
        :param M: -1  stands  for the  max   Magnitude  in  there operation
        :return:
        '''
        operation = self.func[op_name]
        mag = self.ranges[op_name][M]
        image = operation(image, mag)
        return image
    
class Data_Aug_rand(Dataset):
    def __init__(self, train_data, img_num_list, transforms = transform_train, args = args):
        super(Data_Aug_rand, self).__init__()
        self.args = args
        self.datas = train_data.data
        self.labels = train_data.targets
        self.imbal_list = img_num_list
        self.datas, self.labels = self.sort_datas_by_class()
        self.randaug = Rand_Augment()
        self.transforms = transforms
        self.data_imbal, self.label_imbal = self.rand_()
        self.data_imbal = self.transform_img()

    def sort_datas_by_class(self):
        sort_imgs = []
        sort_labels = []
        nums_c = len(self.imbal_list)
        for c in range(nums_c):
            idx_c = []
            for idx, l in enumerate(self.labels):
                if l == c:
                    idx_c.append(idx)
            imgs_c = self.datas[idx_c]
            sort_imgs.append(imgs_c)
            sort_labels.extend([c] * self.imbal_list[c])
        sort_imgs = np.concatenate(sort_imgs, axis = 0)
        return sort_imgs, sort_labels

    def transform_img(self):
        transform_img = []
        for i in range(len(self.label_imbal)):
            img = self.data_imbal[i]
            img = Image.fromarray(img)
            img = self.transforms(img)
            img = img.unsqueeze(0)
            transform_img.append(img)
        transform_img = torch.cat(transform_img, axis = 0).detach().numpy()
        return transform_img

    def rand_(self, max_c = args.max_c):
        data_aug = []
        label_aug = []
        
        pre_S = 0
        class_nums = len(self.imbal_list)
        idx = list(range(len(self.labels)))
        
        for c in range(class_nums):
            data_aug_c = []
            nums_c = self.imbal_list[c]

            idx_c = idx[pre_S : pre_S + nums_c]

            pre_S += nums_c

            data_c = self.datas[idx_c]
            if nums_c < max_c:

                nums_aug_per_img = (max_c - nums_c) // nums_c
                for i in range(nums_c):
                    img_c_i = data_c[i]
                    img_c_i = Image.fromarray(img_c_i)
                    for j in range(nums_aug_per_img):
                        img_j = self.randaug(img_c_i)
                        img_j = np.array(img_j)
                        img_j = img_j[None,:,:,:]
                        data_aug_c.append(img_j)
                        
                for i in range(max_c - nums_c - nums_c * nums_aug_per_img):
                    img_i = data_c[i]
                    img_i = Image.fromarray(img_i)
                    img_i = self.randaug(img_i)
                    img_i = np.array(img_i)[None, :, :, :]
                    data_aug_c.append(img_i)
                    
                data_aug_c = np.concatenate(data_aug_c, axis = 0)
                data_aug_c = np.concatenate([data_aug_c, data_c], axis = 0)
                    
                data_aug.append(data_aug_c)
            else:
                idx_sub = np.random.choice(list(range(len(idx_c))), size = max_c, replace = False)
                data_c_sub = data_c[idx_sub]
                data_aug.append(data_c_sub)
            label_aug.extend([c] * max_c)
            
        data_aug = np.concatenate(data_aug, axis = 0)

        return data_aug, label_aug
    
    def __len__(self):
        return len(self.label_imbal)
    
    def __getitem__(self, index):
        img = self.data_imbal[index]
        label = self.label_imbal[index]
        return img, label, index
