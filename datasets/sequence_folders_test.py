import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import cv2
# from scipy.misc import imresize
from skimage.transform import resize as imresize
from segment.demo_modify import get_reflect, get_illumination, get_enhancment_img, enhancment_illumination, enhancement_reflect

def load_as_float(path):
    image = cv2.imread(path).astype(np.float32)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(img))
    image= imresize(image, (256, 320)).astype(np.float32)
    # img_illumination = get_illumination(image)  # 获得高频分量
    # img_reflect = get_reflect(image, img_illumination)  # 获得反射分量
    # img_enhancement_reflect = enhancement_reflect(img_reflect)  # 增强反射分量
    # img_enhancement_illumination = enhancment_illumination(img_illumination)  # 增强照射分量
    # img_done = get_enhancment_img(img_enhancement_illumination, img_reflect)  # 照射分量与反射分量融合
    # for row in range(256):
    #     for col in range(320):
    #         gray_value = 0.299 * img[row, col, 0] + 0.587 * img[row, col, 1] + 0.114 * img[row, col, 2]
    #         if gray_value > (0.4 * 255):
    #             img[row, col, 0] = 255.
    #             img[row, col, 1] = 255.
    #             img[row, col, 2] = 255.
    # img = np.transpose(img, (2, 0, 1))
    return image

# def load_as_float(path):
#     return imread(path).astype(np.float32)


class SequenceFolderTest(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, img_paths, K_path, transform):
        # np.random.seed(seed)
        # random.seed(seed)
        # self.root = Path(root)
        # scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        # self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        # self.dataset = dataset
        self.K_path = K_path
        # self.k = skip_frames
        self.k = 1
        self.crawl_folders(img_paths, sequence_length=3)

    def crawl_folders(self, img_paths, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        # for img_paths in img_paths:
            # intrinsics = np.genfromtxt(scene/'intrinsics.txt').astype(np.float32).reshape((3, 3))
        intrinsics = np.genfromtxt(self.K_path).astype(np.float32).reshape((3, 3))
        # intrinsics = np.genfromtxt('/home/dokidoki/Datasets/7-Scenes/processing/intrinsics.txt').astype(np.float32).reshape((3, 3))
        # intrinsics = np.genfromtxt('/home/dokidoki/Datasets/mav/Euroc/training_folder/intrinsics.txt').astype(np.float32).reshape((3, 3))
        # intrinsics = np.genfromtxt('/home/dokidoki/Datasets/data_1008/318_v1_1008/intrinsics.txt').astype(np.float32).reshape((3, 3))

        imgs = img_paths
        # print('imgs_name',imgs)
        # stop

        for i in range(demi_length * self.k, len(imgs)-demi_length * self.k-1):
            sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
            for j in shifts:
                sample['ref_imgs'].append(imgs[i+j])

            if i + 1 < len(imgs) - demi_length * self.k:
                sample_next = {'intrinsics': intrinsics, 'tgt': imgs[i+1], 'ref_imgs': []}
                for j in shifts:
                    sample_next['ref_imgs'].append(imgs[i + 1 + j])
            else:
                sample_next = None

            sequence_set.append([sample, sample_next])
            # sequence_set.append(sample)
        # random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample_curr = self.samples[index][0]
        sample_next = self.samples[index][1]
        # if self.dataset == 'mav':
        #     tgt_img_curr = np.expand_dims(load_as_float(sample_curr['tgt']), axis=2)
        #     # print(tgt_img_curr.shape)
        #     # stop
        #     ref_imgs_curr = [np.expand_dims(load_as_float(ref_img), axis=2) for ref_img in sample_curr['ref_imgs']]
        # else:
        tgt_img_curr = load_as_float(sample_curr['tgt'])
        ref_imgs_curr = [load_as_float(ref_img) for ref_img in sample_curr['ref_imgs']]
        if self.transform is not None:
            imgs_curr, intrinsics = self.transform([tgt_img_curr] + ref_imgs_curr, np.copy(sample_curr['intrinsics']))
            tgt_img_curr = imgs_curr[0]
            ref_imgs_curr = imgs_curr[1:]
        else:
            intrinsics = np.copy(sample_curr['intrinsics'])

        # if self.dataset == 'mav':
        #     tgt_img_next = np.expand_dims(load_as_float(sample_next['tgt']), axis=2)
        #     ref_imgs_next = [np.expand_dims(load_as_float(ref_img), axis=2) for ref_img in sample_next['ref_imgs']]
        # else:
        tgt_img_next = load_as_float(sample_next['tgt'])
        ref_imgs_next = [load_as_float(ref_img) for ref_img in sample_next['ref_imgs']]
        if self.transform is not None:
            imgs_next, intrinsics_next = self.transform([tgt_img_next] + ref_imgs_next, np.copy(sample_next['intrinsics']))
            tgt_img_next = imgs_next[0]
            ref_imgs_next = imgs_next[1:]
        else:
            intrinsics_next = np.copy(sample_next['intrinsics'])


        # intrinsics = np.copy(sample['intrinsics'])
        return tgt_img_curr, ref_imgs_curr, tgt_img_next, ref_imgs_next, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
