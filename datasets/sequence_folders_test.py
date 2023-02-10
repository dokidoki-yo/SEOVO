import torch.utils.data as data
import numpy as np
import cv2
from skimage.transform import resize as imresize

def load_as_float(path):
    image = cv2.imread(path).astype(np.float32)
    image = imresize(image, (256, 320)).astype(np.float32)
    return image

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
        self.transform = transform
        self.K_path = K_path
        self.k = 1
        self.crawl_folders(img_paths, sequence_length=3)

    def crawl_folders(self, img_paths, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        intrinsics = np.genfromtxt(self.K_path).astype(np.float32).reshape((3, 3))
        imgs = img_paths

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
        self.samples = sequence_set

    def __getitem__(self, index):
        sample_curr = self.samples[index][0]
        sample_next = self.samples[index][1]
        tgt_img_curr = load_as_float(sample_curr['tgt'])
        ref_imgs_curr = [load_as_float(ref_img) for ref_img in sample_curr['ref_imgs']]
        if self.transform is not None:
            imgs_curr, intrinsics = self.transform([tgt_img_curr] + ref_imgs_curr, np.copy(sample_curr['intrinsics']))
            tgt_img_curr = imgs_curr[0]
            ref_imgs_curr = imgs_curr[1:]
        else:
            intrinsics = np.copy(sample_curr['intrinsics'])

        tgt_img_next = load_as_float(sample_next['tgt'])
        ref_imgs_next = [load_as_float(ref_img) for ref_img in sample_next['ref_imgs']]
        if self.transform is not None:
            imgs_next, intrinsics_next = self.transform([tgt_img_next] + ref_imgs_next, np.copy(sample_next['intrinsics']))
            tgt_img_next = imgs_next[0]
            ref_imgs_next = imgs_next[1:]
        return tgt_img_curr, ref_imgs_curr, tgt_img_next, ref_imgs_next, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
