import numpy as np
import argparse
import torch.optim
import copy
import flow
import custom_transforms
import cv2
import matplotlib
import torch.nn.functional as F
from skimage.transform import resize as imresize
from skimage.measure import label, regionprops
from path import Path
from datasets.sequence_folders_test import SequenceFolderTest
from test1.loss_functions_test2 import compute_photo_and_geometry_loss, compute_smooth_loss
from test1.inverse_warp_test1 import pose_vec2mat
from imageio import imsave
from flow.ransac import reduced_ransac
from models import DispResNet, PoseResNet
# from segment.demo_modify import run as seg
from segment.demo_modify import run as seg, line_detect
from utils import tensor2array
from scipy.spatial.transform import Rotation as R

matplotlib.use('TkAgg')

filter = reduced_ransac(check_num=6000, thres=0.2, dataset='nyu')
show_flag = False
# initial_norm = None

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
# parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
# parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
# parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
# parser.add_argument('--range', default=[1.1, 1.2], type=float, nargs='+', help='path to pre-trained Flow net model')
# parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
# parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
# parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
# parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
# parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
# parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
# parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
# parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
# parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
# parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
# parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
# parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
# parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
# parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
# parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH', help='path to pre-trained Flow net model')
parser.add_argument('--input-folder', dest='input_folder', default=None, metavar='PATH', help='path to the input images')
parser.add_argument('--intrinsics', dest='intrinsics_txt', default=None, metavar='PATH', help='path to the txt file of the camera intrinsics')
# parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
# parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
#                     help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
#                          ' zeros will null gradients outside target image.'
#                          ' border will only null gradients of the coordinate outside (x or y)')
# parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
#                     You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(filename):
    print('filename',filename)
    # img = imread(filename).astype(np.float32)
    if filename.split('/')[-3] == 'depth' or filename.split('/')[-2] == 'depth':
        img = cv2.imread(filename,-1).astype(np.int16)
        # print(img)
        # stop
        # print(np.min(img))
        # print(np.max(img))
        # cv2.imshow('img_ori', img)
        # cv2.waitKey(0)


        # img = cv2.normalize(img, None,0,255, cv2.NORM_MINMAX)
        # img = np.expand_dims(img, axis= 2)
        # print(img)
        # stop

        # img = cv2.resize(img, (320, 256)).astype(np.float32)
        # img = cv2.resize(img, (320, 256)).astype(np.int16)
        # print(img)
        # stop
        img = cv2.resize(img, (320, 256), interpolation=cv2.INTER_NEAREST).astype(np.int16)
        # cv2.imshow('img_resize', img)
        # cv2.waitKey(0)
        # # print(np.min(img))
        # stop

        tensor_img = torch.from_numpy(img).to(device)
        # print(tensor_img)
        # stop

    else:
        img = cv2.imread(filename).astype(np.float32)
        h, w, _ = img.shape
        # if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (256, 320)).astype(np.float32)
        # img = cv2.resize(img, (320, 256)).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
        # tensor_img = (torch.from_numpy(img).unsqueeze(0)).to(device)
    # print(img.shape)
    # for row in range(480):
    #     for col in range(640):
    #         gray_value = 0.299 * img[row, col, 0] + 0.587 * img[row, col, 1] + 0.114 * img[row, col, 2]
    #         if gray_value > (0.4 * 255):
    #             img[row, col, 0] = 255.
    #             img[row, col, 1] = 255.
    #             img[row, col, 2] = 255.
    # white_mask = img > (0.4 * 255)
    # print((white_mask*255).astype(np.uint8))
    # cv2.imshow("tophat", img)
    # cv2.waitKey(0)


    # # Getting the kernel to be used in Top-Hat
    # filterSize = (90, 90)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
    #                                    filterSize)
    #
    # # Reading the image named 'input.jpg'
    # # input_image = cv2.imread("testing.jpg")
    # input_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # Applying the Top-Hat operation
    # # tophat_img = cv2.morphologyEx(input_image,
    # #                               cv2.MORPH_TOPHAT,
    # #                               kernel)
    # tophat_img = cv2.morphologyEx(input_image,
    #                               cv2.MORPH_BLACKHAT,
    #                               kernel)
    # input_image_mask = tophat_img < 10
    # # tophat_img = tophat_img.bool() * input_image_mask
    #
    # # 图片二值化
    #
    # cv2.imshow("original", input_image)
    # # cv2.imshow("tophat", tophat_img)
    # cv2.imshow("tophat", (input_image_mask*255).astype(np.uint8))
    # cv2.waitKey(0)


    # dst = cv2.pyrMeanShiftFiltering(img, 50, 50, None, 2)
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)


    return tensor_img

def meshgrid(h, w):
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    meshgrid = np.transpose(np.stack([xx, yy], axis=-1), [2, 0, 1])  # [2,h,w]
    meshgrid = torch.from_numpy(meshgrid)
    return meshgrid

def rt_from_fundamental_mat_nyu(fmat, K, depth_match):
    # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
    # verify_match = self.rand_sample(depth_match, 5000) # [b,4,100]
    verify_match = depth_match.transpose(1, 2).cpu().detach().numpy()
    # K_inv = torch.inverse(K)
    b = fmat.shape[0]

    # print(K.shape)
    # print('F',fmat)
    # print(K)
    # print(fmat)
    fmat_ = K.transpose(1, 2).bmm(fmat)

    # stop
    essential_mat = fmat_.bmm(K)  # E = K^T * F *K
    # print('K',K)
    iden = torch.cat([torch.eye(3), torch.zeros([3, 1])], -1).unsqueeze(0).repeat(b, 1, 1).to(
        device)  # [b,3,4]
    P1_K = K.bmm(iden)  # P1 with identity rotation and zero translation
    flags = []
    number_inliers = []
    P2 = []
    # P2_ = []
    # print('E', essential_mat[0])
    for i in range(b):
        cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'),
                                        verify_match[i, :, :2].astype('float64'), \
                                        verify_match[i, :, 2:].astype('float64'),
                                        cameraMatrix=K[i, :, :].cpu().detach().numpy().astype('float64'))
        # U,S,VT = np.linalg.svd(essential_mat[i].cpu().detach().numpy().astype('float64'))
        # # print(R.shape)
        # # print(t.shape)
        # W = np.array([[0,-1,0],
        #               [1,0,0],
        #               [0,0,1]])
        # WT = np.array([[0,1,0],
        #               [-1,0,0],
        #               [0,0,1]])
        # # print(WT.shape)
        # # print('E',)
        # # print('U',U)
        # # print('V',VT)
        # # print(W*VT)
        # R1 = np.matmul(U,np.matmul(W,VT))
        # R2 = np.matmul(U,np.matmul(WT,VT))
        # t1 = U[:,2]
        # t2 = -U[:,2]
        # print(t1,t2)
        # print(R1.shape)
        # if t1[0]<0 and np.abs(t1[0])>0.5:
        #     t = t2
        # else:
        #     t = t1
        # t1 = torch.from_numpy(t1).unsqueeze(1)
        # t2 = torch.from_numpy(t2).unsqueeze(1)
        # print(np.array([[t2[0]],[t2[1]],t2[2]]).shape)
        # stop

        # cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'),
        #                                 verify_match[i, :, 2:].astype('float64'),
        #                                 verify_match[i, :, :2].astype('float64'),
        #                                 cameraMatrix=K[i, :, :].cpu().detach().numpy().astype('float64'))
        p2 = torch.from_numpy(np.concatenate([R, t], axis=-1)).float().to(device)
        P2.append(p2)
        # p2_l = torch.from_numpy(np.concatenate([R1, t1], axis=-1)).float().to(device)
        # p2_r = torch.from_numpy(np.concatenate([R1, t2], axis=-1)).float().to(device)
        # p2_b = torch.from_numpy(np.concatenate([R2, t1], axis=-1)).float().to(device)
        # p2_t = torch.from_numpy(np.concatenate([R2, t2], axis=-1)).float().to(device)

        # P2.append(p2_l)
        # P2.append(p2_r)
        # P2.append(p2_b)
        # P2.append(p2_t)
        if cnum > depth_match.shape[-1] / 7.0:
            flags.append(1)
        else:
            flags.append(0)
        number_inliers.append(cnum)
        # print(i,'E',essential_mat[i])
        # print(i,'match', verify_match[i, :, :2],verify_match[i, :, 2:])
        # if i == 0:
        #     print('P before K',torch.stack(P2, axis=0))
    # print(torch.stack(P2, axis=0).shape)
    # P2_K = K.bmm(torch.stack(P2, axis=0))
    # P2_K = K.repeat(4,1,1).bmm(torch.stack(P2, axis=0))
    P2 = torch.stack(P2, axis=0)
    # print('F',P2)
    # pdb.set_trace()

    return P2

import mxnet as mx
def depth2normal(depth):
    w, h =depth.shape
    dx = -(depth[2:h,1:h-1]-depth[0:h-2,1:h-1])*0.5
    dy = -(depth[1:h-1,2:h]-depth[1:h-1,0:h-2])*0.5
    dz = torch.ones((w-2,h-2)).to(device)
    dl = torch.sqrt(torch.mul(dx, dx) + torch.mul(dy, dy) + torch.mul(dz, dz))
    dx = torch.div(dx, dl) * 0.5 + 0.5
    dy = torch.div(dy, dl) * 0.5 + 0.5
    dz = torch.div(dz, dl) * 0.5 + 0.5

    return torch.cat([dy.unsqueeze(0), dx.unsqueeze(0), dz.unsqueeze(0)])
    # return np.concatenate([dy.asnumpy()[np.newaxis,:,:],dx.asnumpy()[np.newaxis,:,:],dz.asnumpy()[np.newaxis,:,:]],axis=0)

def main():
    args = parser.parse_args()

    ### load pretrained weight of flow, pose and depth net trained on NYUv2

    # flownet
    # weights_flow = torch.load('/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner-Release-master/checkpoints/resnet18_flow_NYUv2/01-10-20:05-pyramid1/flow_net_checkpoint.pth.tar')
    weights_flow = torch.load(args.pretrained_flow)
    flow_net = flow.Model_flow_test().to(device)
    flow_net.load_state_dict(weights_flow['state_dict'], strict=False)
    flow_net.eval()

    # posenet
    # weights_pose = torch.load('/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/checkpoints/r18_nyu/04-23-23:32-0.8F/exp_pose_model_best.pth.tar')
    weights_pose = torch.load(args.pretrained_pose)
    pose_net = PoseResNet(18, pretrained=False).to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    # depthnet
    # weights_depth = torch.load('/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/checkpoints/r18_nyu/04-23-23:32-0.8F/dispnet_model_best.pth.tar')
    weights_depth = torch.load(args.pretrained_disp)
    depth_net = DispResNet(18, pretrained=1).to(device)
    depth_net.load_state_dict(weights_depth['state_dict'], strict=False)
    depth_net.eval()

    # read color images from dataset folder and sort them
    # image_dir = Path("/home/dokidoki/Datasets/data_1008/318_v1_1008/color/")
    image_dir = Path(args.input_folder + 'color/')
    test_files =image_dir.files('*.png')
    test_files.sort()
    n = len(test_files)
    print('{} files to test'.format(n))

    # the first image
    tensor_img1 = load_tensor_image(test_files[0])
    tensor_img1_name = test_files[0]

    # some list for future use
    keyframe_pool = []  # to store keyframes
    F_poses_pool = []  # to store the epipolar poses decomposed from the fundamental matrix based on the forward flow
    F_poses_pool_inv = [] # to store the epipolar poses decomposed from the fundamental matrix based on the backward flow
    reproj_matches = []
    reproj_matches_inv = []

    plane_masks = []
    seg_maps = []


    # read intrinsics and inverse
    # K_ms = np.genfromtxt('/home/dokidoki/Datasets/data_preparation/318_v1/intrinsics.txt').astype(np.float32).reshape(
    #     (3, 3))
    K_path = args.input_folder + 'intrinsics.txt'
    K_ms = np.genfromtxt(K_path).astype(np.float32).reshape((3, 3))
    # K_ms_inv_numpy = np.linalg.inv(K_ms)
    K_ms = torch.from_numpy(K_ms).to(device)
    K_ms_inv = K_ms.inverse()

    # initialize keyframe pool
    keyframe_pool.append(tensor_img1_name)

    # normalize the input images and transform them to device
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    gt_depths = []
    # poses_gt_write = []
    # poses_gt = []
    # poses_pool = []
    # pose_ini = np.genfromtxt('/home/dokidoki/Datasets/7-Scenes/processing/head/poses/seq01/000001.txt').astype(
    #     np.float32).reshape(
    #     (4, 4))
    # poses_pool.append(pose_ini)

    # for iter in tqdm(range(n - 1)):
    # for iter in range(1100):
    # skip_frame = 0
    # for iter in range(n-1):
    for iter in range(200):
        # read next image
        tensor_img2_name = test_files[iter+1]
        tensor_img2 = load_tensor_image(test_files[iter+1])
        # skip_frame+=1

        # get the flow between the two frames through flownet
        imgs = torch.cat([tensor_img1, tensor_img2], dim=2)
        return_flow, return_flow_inv = flow_net([imgs, K_ms.unsqueeze(0), K_ms_inv.unsqueeze(0)], is_training=False)
        return_flow_reshape = return_flow.reshape((2,-1))
        average_flow = torch.mean(torch.sqrt(return_flow_reshape[0]**2+return_flow_reshape[1]**2))

        # select keyframes
        flow_th = 20.0
        if average_flow.item() < flow_th:
            print('too close or too far', average_flow)
            continue
        elif average_flow.item() >= flow_th:
            print('in keyframe pool')
            # add the keyframe image and its gt_depth(just for evaluation) in the corresponding pools
            keyframe_pool.append(tensor_img2_name)
            img_name = tensor_img2_name.split('/')[-1]
            # pose_curr = np.genfromtxt(
            #     '/home/dokidoki/Datasets/7-Scenes/processing/head/poses/seq01/000001.txt').astype(
            #     np.float64).reshape(
            #     (4, 4))
            # pose_curr[:3, :3] = R.from_matrix(pose_curr[:3, :3]).as_matrix()
            # pose_curr = np.genfromtxt(
            #     '/home/dokidoki/Datasets/7-Scenes/processing/chess/poses/seq03/000001.txt').astype(
            #     np.float64).reshape(
            #     (4, 4))
            # q = Quaternion(matrix=pose_curr[:3,:3])
            # print(q)
            # pose_curr = np.genfromtxt(
            #     '/home/dokidoki/Datasets/7-Scenes/processing/test/08.txt').astype(
            #     np.float32)[0].reshape(
            #     (3, 4))
            # print(np.linalg.det(pose_curr[:3,:3]))
            # print('pose_curr', K_ms_inv_numpy.shape)
            # pose_k_inv = K_ms_inv_numpy@pose_curr[:3, :]
            # print('pose_k_inv', pose_k_inv)
            # poses_pool.append(pose_curr)
            # print('pose_ini', pose_ini)
            # print('pose_curr', pose_curr)
            # stop
            # poses_gt_write.append(pose_curr[:3,:].reshape((1,12)))
            # print('/home/dokidoki/Datasets/7-Scenes/processing/depth/'+last_name)
            # gt_depth = load_tensor_image('/home/dokidoki/Datasets/7-Scenes/processing/stairs/depth/seq01/'+last_name)
            # gt_depth = load_tensor_image('/home/dokidoki/Datasets/data_1008/318_v1_1008/depth/'+last_name)
            # gt_depth = load_tensor_image('/home/dokidoki/Datasets/data_1008/zoulang_v1_1008/depth/'+last_name)
            gt_depth = load_tensor_image(args.input_folder + 'depth/' + img_name)
            gt_depth[gt_depth ==0] = 5000

            # gt_depth_show = tensor2array(gt_depth, max_value=None, colormap='magma')
            #
            # gt_depth_show = np.transpose(gt_depth_show, (1, 2, 0))
            # # gt_depth_show = cv2.cvtColor(gt_depth_show, cv2.COLOR_BGR2RGB)
            # # cv2.imshow('gt', gt_depth_show)
            # # cv2.waitKey(0)
            # imsave(
            #     '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/gt_depth/' + last_name,
            #     # np.transpose(tgt_img[0].cpu().detach().numpy(), (1, 2, 0)))
            #     gt_depth_show)

            gt_depths.append(gt_depth)


            # get the pixel correspondence by the flow

            grid = meshgrid(256, 320).float().to(device).unsqueeze(0).repeat(1, 1, 1,1)  # [b,2,h,w]

            fwd_corres = torch.cat([(grid[:, 0, :, :] + return_flow[:, 0, :, :]).unsqueeze(1),
                                    (grid[:, 1, :, :] + return_flow[:, 1, :, :]).unsqueeze(1)], 1)
            bwd_corres = torch.cat([(grid[:, 0, :, :] + return_flow_inv[:, 0, :, :]).unsqueeze(1),
                                    (grid[:, 1, :, :] + return_flow_inv[:, 1, :, :]).unsqueeze(1)], 1)

            fwd_match = torch.cat([grid, fwd_corres], 1).detach()  # [b,4,h,w]
            bwd_match = torch.cat([grid, bwd_corres], 1).detach()  # [b,4,h,w]

            fwd_match_reshape = fwd_match.view([1, 4, -1]).transpose(1, 2).reshape([1, 256, 320, -1])[:, :, :,
                                2:]  # b,h,w,2
            bwd_match_reshape = bwd_match.view([1, 4, -1]).transpose(1, 2).reshape([1, 256, 320, -1])[:, :, :, 2:]

            fwd_match_norm = fwd_match_reshape.clone().detach()
            bwd_match_norm = bwd_match_reshape.clone().detach()

            fwd_match_norm[:, :, :, 0] = 2 * (fwd_match_norm[:, :, :, 0]) / (320 - 1) - 1
            fwd_match_norm[:, :, :, 1] = 2 * (fwd_match_norm[:, :, :, 1]) / (256 - 1) - 1

            bwd_match_norm[:, :, :, 0] = 2 * (bwd_match_norm[:, :, :, 0]) / (320 - 1) - 1
            bwd_match_norm[:, :, :, 1] = 2 * (bwd_match_norm[:, :, :, 1]) / (256 - 1) - 1

            reproj_matches.append(fwd_match_norm)
            reproj_matches_inv.append(bwd_match_norm)

            boundary_mask = (fwd_corres[:, 0, :, :] > 10) * (fwd_corres[:, 0, :, :] < 310) * (
                        fwd_corres[:, 1, :, :] > 10) * (fwd_corres[:, 1, :, :] < 250)
            # boundary_mask = torch.ones_like(boundary_mask) * boundary_mask.int()

            boundary_mask_inv = (bwd_corres[:, 0, :, :] > 10) * (bwd_corres[:, 0, :, :] < 310) * (
                    bwd_corres[:, 1, :, :] > 10) * (bwd_corres[:, 1, :, :] < 250)
            # boundary_mask_inv = torch.ones_like(boundary_mask_inv) * boundary_mask_inv.int()

            # seg_maps.append(np.ones([81920]))

            #
            # tgt_depth = 1. / depth_net(tensor_img1)[0]
            # tgt_depth_inv = 1. / depth_net(tensor_img2)[0]
            #
            # mask_depth = (tgt_depth > (torch.mean(tgt_depth) + 3.5 * torch.var(tgt_depth))).int()

            # initial the plane mask with low grayscale
            # img1_gray = 0.299 * tensor_img1[:, 0, :, :] + 0.587 * tensor_img1[:, 1, :, :] + 0.114 * tensor_img1[:, 2, :, :]
            # img2_gray = 0.299 * tensor_img2[:, 0, :, :] + 0.587 * tensor_img2[:, 1, :, :] + 0.114 * tensor_img2[:, 2, :,
            #                                                                                         :]
            # img1_back_mask = img1_gray > 0.4

            # mask_depth = img1_back_mask.unsqueeze(1)

            # if torch.sum(mask_depth) > 10000:  # if the plane is large
            #     mask_gc = mask_depth
            # else:
            #     mask_gc = 1-boundary_mask.int().unsqueeze(0)
            # plane_masks.append(mask_gc)

            # mask_depth_inv = (tgt_depth_inv > (torch.mean(tgt_depth_inv) + 7. * torch.var(tgt_depth_inv))).int()
            # img2_back_mask = img2_gray > 0.35
            # mask_depth_inv = img2_back_mask.unsqueeze(1)
            #
            # if torch.sum(mask_depth_inv) > 10000:
            #     mask_gc_inv = mask_depth_inv
            # else:
            #     mask_gc_inv = 1-boundary_mask_inv.int()

            # calculate the pose according to the flow, the mask is just the boundary mask
            # F_final_1, F_match = filter(fwd_match, 1-mask_gc.int())
            # F_final_2, F_match_inv = filter(bwd_match, 1-mask_gc_inv.int())

            # calculate the fundamental matrix based on the flow
            F_final_1, F_match = filter(fwd_match, boundary_mask.int())
            F_final_2, F_match_inv = filter(bwd_match, boundary_mask_inv.int())

            K = K_ms.unsqueeze(0).to(device)
            P2 = rt_from_fundamental_mat_nyu(F_final_1.detach().to(device), K, F_match.clone())
            P2_inv = rt_from_fundamental_mat_nyu(F_final_2.detach().to(device), K, F_match_inv.clone())

            # P2 = -P2
            # P2[0][0,0] = -P2[0][0,0]
            # P2[0][1,1] = -P2[0][1,1]
            # P2[0][2,2] = -P2[0][2,2]
            # P2_inv = -P2_inv
            # P2_inv[0][0, 0] = -P2_inv[0][0, 0]
            # P2_inv[0][1, 1] = -P2_inv[0][1, 1]
            # P2_inv[0][2, 2] = -P2_inv[0][2, 2]
            # pose_f_gt = np.linalg.inv(pose_ini)@pose_curr
            # pose_f_gt[:3,3] = pose_f_gt[:3,3]/np.sqrt(pose_f_gt[0,3]**2 + pose_f_gt[1,3]**2 + pose_f_gt[2,3]**2)
            # pose_f_gt_inv = np.linalg.inv(pose_curr) @ pose_ini
            # pose_f_gt_inv[:3, 3] = pose_f_gt_inv[:3, 3] / np.sqrt(
            #     pose_f_gt_inv[0, 3] ** 2 + pose_f_gt_inv[1, 3] ** 2 + pose_f_gt_inv[2, 3] ** 2)
            # # print('T', pose_f_gt)
            # # print('P2', P2)
            # poses_gt.append(pose_f_gt)
            # print('T', pose_f_gt_inv)
            # print('P2_inv', P2_inv)
            #  0.99753665  0.02929729 -0.06374847  0.04940725]
            #  [-0.03188675  0.99869239 -0.0399922   0.16343834]
            #  [ 0.06249419  0.04192706  0.99716504 -0.9853156 ]
            #  [ 0.          0.          0.          1.        ]]
            # P2_inv tensor([[[ 0.9983,  0.0309, -0.0486, -0.2460],
            #          [-0.0289,  0.9987, -0.0416,  0.2600],
            #          [ 0.0498,  0.0402,  0.9979, -0.9338]]]
            # stop
            # pose_ini = pose_curr
            # stop
            F_poses_pool.append(P2)
            F_poses_pool_inv.append(P2_inv)
            tensor_img1 = tensor_img2

    # initialize the optimizer
    optim_params = [
        {'params': depth_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    optimizer_SGD = torch.optim.SGD(optim_params, lr=args.lr)

    # prepare the data pairs
    train_loader = torch.utils.data.DataLoader(
        generate_training_set(keyframe_pool, K_path, train_transform), batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)
    # from datasets.validation_folders import ValidationSet
    # valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    # val_set = ValidationSet(
    #     '/home/dokidoki/Datasets/7-Scenes/processing/',
    #     transform=valid_transform,
    #     dataset='mav'
    # )
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    # print(len(keyframe_pool))
    # print(len(poses_gt))
    # print(len(gt_depths))


    # poses_cat = np.concatenate(poses_gt_write[:-2], axis=0)
    # txt_filename = Path("/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/pose_gt.txt")
    # np.savetxt(txt_filename, poses_cat, delimiter=' ', fmt='%1.8e')

    # stop


    gt_depths = torch.stack(gt_depths)# n,1,1,256,320

    output_depths = online_adaption(pose_net, depth_net, optimizer, optimizer_SGD, train_loader, [F_poses_pool, F_poses_pool_inv], [reproj_matches, reproj_matches_inv], plane_masks)

    # print('**********errors************',compute_errors(gt_depths[:-2]/500, output_depths, 'nyu'))


def generate_training_set(keyframes_path, K_path, train_transform):
    return SequenceFolderTest(keyframes_path, K_path, train_transform)

def compute_pose_with_inv(pose_net, tgt_img, ref_imgs, poses_f, poses_f_inv):
    poses = []
    poses_inv = []
    poses_f_align = []
    poses_inv_f_align = []

    for ref_img, pose_f, pose_f_inv in zip(ref_imgs, poses_f, poses_f_inv) :
        # if iter_idx > 15 and iter_idx < 21:
        #     poses.append(pose_net(tgt_img, ref_img, grad_weights).to(device).detach())
        #     poses_inv.append(pose_net(ref_img, tgt_img, grad_weights).to(device).detach())
        # else:
        pose = pose_net(tgt_img, ref_img).to(device)
        pose_inv = pose_net(ref_img, tgt_img).to(device)
        # poses.append(pose)
        # poses_inv.append(pose_inv)
        pose_mat = pose_vec2mat(pose)
        pose_mat_inv = pose_vec2mat(pose_inv)
        poses.append(pose_mat)
        poses_inv.append(pose_mat_inv)


        t_max_idx = torch.max(torch.abs(pose_mat[0, :, 3]), dim=0)[1]
        scale = pose_mat[:, t_max_idx, 3]/pose_f[:, t_max_idx, 3]
        # print(pose_f[:, :, :3].shape)
        # print(pose_f[:, :, 3:].shape)
        pose_mat_new = torch.cat([pose_f[:, :, :3], scale*pose_f[:, :, 3:]], dim=-1)
        # pose_mat_new = torch.cat([pose_mat[:, :, :3], scale*pose_f[:, :, 3:]], dim=-1)

        t_max_idx_inv = torch.max(torch.abs(pose_mat_inv[0, :, 3]), dim=0)[1]
        scale_inv = pose_mat_inv[:, t_max_idx_inv, 3] / pose_f_inv[:, t_max_idx_inv, 3]
        pose_mat_new_inv = torch.cat([pose_f_inv[:, :, :3], scale_inv * pose_f_inv[:, :, 3:]], dim=-1)
        # pose_mat_new_inv = torch.cat([pose_mat_inv[:, :, :3], scale_inv * pose_f_inv[:, :, 3:]], dim=-1)
        # print('pose_mat', pose_mat)
        # print('pose_mat_new', pose_mat_new)
        poses_f_align.append(pose_mat_new)
        poses_inv_f_align.append(pose_mat_new_inv)
    return poses, poses_inv, poses_f_align, poses_inv_f_align

def compute_depth(disp_net, tgt_img, ref_imgs):
    # print(disp_net(tgt_img))
    # print(tgt_img)
    # if iter_idx > 10 and iter_idx < 21:
    #     tgt_depth = [1 / disp.to(device).detach() for disp in disp_net(tgt_img, grad)]
    # else:
    tgt_depth = [(1 / disp.to(device)) for disp in disp_net(tgt_img)]
    # print('max_depth',torch.max(tgt_depth[0]))

    ref_depths = []
    for ref_img in ref_imgs:
        # if iter_idx > 10 and iter_idx < 21:
        #     ref_depth = [1 / disp.to(device).detach() for disp in disp_net(ref_img, grad)]
        # else:
        ref_depth = [1 / disp.to(device) for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths

def compute_F_loss(poses_f, poses, poses_inv):
    loss_R = 0
    loss_t = 0
    for pose_f, pose, pose_inv in zip(poses_f, poses, poses_inv):
        # pose_mat = pose_vec2mat(pose) #b, 3, 4
        pose_mat = pose #b, 3, 4
        # pose_inv_mat = pose_vec2mat(pose_inv) #b, 3, 4
        r_weight = torch.norm(pose_mat[:, :, 3]).detach()
        loss_R = loss_R + torch.mean(torch.abs(pose_f[:, :, :3] - pose_mat[:, :, :3]))

        # print('pose_f', pose_f)
        # print('pose_mat', pose_mat)
        # print(pose_f[:, :, 3])
        # print(pose_mat[:, :, 3]/torch.norm(pose_mat[:, :, 3]))
        t_weight = torch.abs((pose_mat[:, :, 3]/torch.norm(pose_mat[:, :, 3]))).detach()
        # print('t_weight',t_weight)
        # loss_reg = torch.mean(torch.abs(pose_inv_mat[:, :, 3].detach() + pose_mat[:, :, 3]))

        # t_max_idx = torch.max(torch.abs(pose_mat[0, :, 3]), dim=0)[1]
        # scale = pose_mat[:, t_max_idx, 3] / pose_f[:, t_max_idx, 3]
        # pose_mat_new = torch.cat([pose_f[:, :, :3], scale * pose_f[:, :, 3:]], dim=-1)

        loss_t += torch.mean(t_weight*torch.abs(pose_f[:, :, 3]-pose_mat[:, :, 3]/torch.norm(pose_mat[:, :, 3])))
        # loss_t_f = torch.mean(torch.abs(pose_mat-pose_mat_new))
        # loss_t = loss_t_f
    # print('lossR', loss_R)
    # print('lossreg', loss_reg)
    # print('lossttf', loss_t_f)
    return loss_R+loss_t

def largestConnectComponent(bw_img, multiple):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    #!!!! n: not include the background(label 0)

    max_label = 0
    max_num = 0
    # print(labeled_img)
    # print(bw_img)
    # print(np.unique(labeled_img))
    # print('num', num)

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    if multiple:
        connect_regions = []
        bboxs = []
        for i in range(1, num+1):
            # print('i',np.sum(labeled_img == 1))
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        if max_num < 5000:
            return connect_regions
        for i in range(1,num+1):
            # print('i',i)
            # print('ratio',np.sum(labeled_img == i)/max_num)
            mcr = (labeled_img == i)
            # mcr_flatten = torch.nonzero(torch.from_numpy(mcr).to(device).reshape(-1))  # select pixel indices on the plane
            # # print(seg_depth_flatten.shape)  # n,1
            # seg_img = torch.gather(seg_map, index=mcr_flatten[:, 0],
            #                        dim=0)  # select potential plane pixel in segment map
            # # print(seg_img.shape)  # n
            # seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)


            # if show_flag:
            #     cv2.imshow('mcr', 255 * mcr.astype(np.uint8))
            #     cv2.waitKey(0)
            minr, minc, maxr, maxc = regionprops(labeled_img)[i-1].bbox
            bboxs.append([minr, minc, maxr, maxc])
            # print('connect_region_ratio',np.sum(labeled_img == i)/max_num)


            if max_num!= 0 and np.sum(labeled_img == i)/max_num > 0.3:
                # print('current region in')
                connect_regions.append(mcr)
        return connect_regions
    else:
        for i in range(1,num+1):
            # print('i',np.sum(labeled_img == 1))
            if np.sum(labeled_img == i) > max_num and i!=0:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mcr = (labeled_img == max_label)

        minr, minc, maxr, maxc = regionprops(labeled_img)[max_label-1].bbox
        return labeled_img, max_label, [minr, minc, maxr, maxc], mcr

def calculate_all_norm(tgt_depth):
    tgt_depth_normal = torch.zeros([320, 320]).to(device)
    # tgt_depth_normal[:256, :] = (tgt_depth).detach().cpu().numpy() * 300
    tgt_depth_normal[:256, :] = tgt_depth * 300
    # tgt_depth_normal = cv2.GaussianBlur(np.expand_dims(tgt_depth_normal, axis=2), (3, 3), 15)

    normal = depth2normal(tgt_depth_normal) * 255  #3, 318, 318
    # print('normal', normal)

    normal_map = torch.zeros([1, 3, 256, 320]).to(device)
    normal_map[0, :, 1:-1, 1:319] = normal[:, :254, :]
    non_idx = torch.nonzero(normal_map.reshape([1,3,-1]))
    # print(non_idx[0])
    # print('normal_map',normal_map.reshape([1, 3, -1])[:, :, 321])

    # grad_disp_x = torch.abs(normal_map[:, :, :, :-1] - normal_map[:, :, :, 1:])
    # grad_disp_y = torch.abs(normal_map[:, :, :-1, :] - normal_map[:, :, 1:, :])
    #
    # loss_normal = torch.abs(torch.mean(grad_disp_x * mask[:, :, :, :-1])) + torch.abs(torch.mean(grad_disp_y* mask[:, :, :-1, ]))

    # stop
    # normal_show = normal.detach().numpy()

    # normal_image = cv2.cvtColor(np.transpose(normal_show, [1, 2, 0]), cv2.COLOR_BGR2RGB)
    # normal_map = torch.zeros([1, 3, 256, 320])
    # normal_map[0, :, 1:-1, 1:319] = torch.from_numpy(np.transpose(normal_image[:254, :, :], [2, 0, 1]))
    #
    # normal_map_tensor = normal_map[0].detach().cpu().numpy()
    # normal_map_tensor = np.transpose(normal_map_tensor, (1, 2, 0)).astype(np.uint8)
    # cv2.imshow('normal_map', normal_map_tensor)
    # cv2.waitKey(0)

    return normal_map

def calculate_total_seg(mask_depth, plane_mask, seg_map, sobel):

    plane_mask_copy = plane_mask

    seg_depth_flatten = torch.nonzero(plane_mask_copy.reshape(-1))  # select pixel indices on the plane
    # print(seg_depth_flatten.shape)  # n,1
    seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)  # select potential plane pixel in segment map
    # print(seg_img.shape)  # n
    seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)  # count the labels in the segment

    max_idx = torch.max(lab_count, dim=0)[1]  # exact the largest part in the segment
    # seg_labs_max = seg_labs_select[torch.max(lab_count, dim=0)[1]]
    seg_labs_max = seg_labs_select[max_idx]
    # lab_count_copy = lab_count.clone()
    # lab_count_copy[max_idx] = 0
    # seg_labs_second_max = seg_labs_select[torch.max(lab_count_copy, dim=0)[1]]
    # print(seg_labs_max)
    seg_reshape = seg_map.reshape(1, 1, 256, 320)
    # print(torch.sum((seg_reshape == seg_labs_max).int()))
    seg_mask = torch.where(seg_reshape == seg_labs_max, torch.ones_like(mask_depth),
                           torch.zeros_like(mask_depth))
    # print('seg_mask', seg_mask.shape)
    _, _, bbox, mcr = largestConnectComponent(seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8), multiple=False)
    # label_mask_return = (labeled_img == max_label)
    max_mask_return = seg_mask[0,0]

    mask_depth = (mask_depth-seg_mask)>0
    # print('mask_depth_copy', torch.sum(mask_depth))
    total_seg_mask = plane_mask
    if torch.sum(mask_depth)>0:
        seg_depth_flatten = torch.nonzero(mask_depth.int().reshape(-1))
        # print(seg_depth_flatten.shape)  # n,1
        seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)
        # print(seg_img.shape)  # n
        seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)
        print('label_count', lab_count)

        max_idx = torch.max(lab_count, dim=0)[1]
        seg_labs_max = seg_labs_select[max_idx]
        seg_reshape = seg_map.reshape(1, 1, 256, 320)
        seg_mask_second_plane = torch.where(seg_reshape == seg_labs_max, torch.ones_like(mask_depth),
                               torch.zeros_like(mask_depth))

        depth_flatten = torch.nonzero(seg_mask_second_plane.reshape(-1))

        second_depth = torch.gather(torch.from_numpy(sobel).reshape(-1).to(device), index=depth_flatten[:, 0], dim=0)
        # print('var~~~~~~~~~~~~~~~~~~~~~~~', torch.var(second_depth.float()))

        labeled_img, max_label, _ = largestConnectComponent(seg_mask_second_plane[0, 0].detach().cpu().numpy().astype(np.uint8),False)
        label_mask_second = (labeled_img == max_label)
        if np.sum(label_mask_second) > 5000 and torch.var(second_depth.float()) < 1000:
            print('second_plane')
            total_seg_mask = plane_mask + torch.tensor(label_mask_second).unsqueeze(0).unsqueeze(0).to(device)

    return torch.tensor(max_mask_return).unsqueeze(0).unsqueeze(0).to(device), bbox,(total_seg_mask > 0).int()


def maximum_internal_rectangle(img):
    global show_flag
    # print(img_gray)
    # print(img_gray * 255)
    # img = cv2.imread('test_depth/color_0.png')
    # # img = img_gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray)
    #
    # stop


    ret, img_bin = cv2.threshold(img_gray*255, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0].reshape(len(contours[0]), 2)

    rect = []

    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))

    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)

        while not best_rect_found and index_rect < nb_rect:

            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]

            valid_rect = True

            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img[y1, x]) == 0 or any(img[y2, x]) == 0:
                    valid_rect = False
                x += 1

            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img[y, x1]) == 0 or any(img[y, x2]) == 0:
                    valid_rect = False
                y += 1

            if valid_rect:
                best_rect_found = True

            index_rect += 1

        if best_rect_found:
            # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
            rect_mask = torch.zeros([256, 320])
            rect_mask[np.min([y1, y2]):np.max([y1, y2]), np.min([x1, x2]):np.max([x1, x2])] = 1
            # print('x1, x2, y1, y2', x1, x2, y1, y2)
            # print('rect_mask',torch.sum(rect_mask))
            # if show_flag:
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            #     cv2.imshow("rec", img)
            #     cv2.waitKey(0)

            return np.abs((x2 - x1) * (y2-y1)), rect_mask


        else:
            print("No rectangle fitting into the area")
            return 0,0

    else:
        print("No rectangle found")
        return 0,0

def calculate_seg_new(src, plane_mask, seg_map, depth, sobel,grad,flag):

    plane_mask_copy = plane_mask
    max_mask_return = torch.zeros_like(plane_mask).to(device).int()
    class_idx = 1
    # print('plane_mask_num', torch.sum(plane_mask_copy))

    max_area = None

    while torch.sum(plane_mask_copy) > 5000:

        # print('plane_mask_num', torch.sum(plane_mask_copy))
        seg_depth_flatten = torch.nonzero(plane_mask_copy.reshape(-1))  # select pixel indices on the plane
        index_boundary = torch.nonzero(plane_mask_copy[0,0])  #n,2
        seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)  # select potential plane pixel in segment map
        # print(seg_img.shape)  # n
        seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)  # count the labels in the segment

        max_count, max_idx = torch.max(lab_count, dim=0)  # exact the largest part in the segment
        if max_area is None:
            max_area = max_count
        else:
            # print('test ratio', max_count / max_area)
            if max_count/max_area < 0.2:
                break
        # seg_labs_max = seg_labs_select[torch.max(lab_count, dim=0)[1]]
        seg_labs_max = seg_labs_select[max_idx]

        seg_reshape = seg_map.reshape(1, 1, 256, 320)


        seg_mask = torch.where(seg_reshape == seg_labs_max, torch.ones_like(plane_mask),
                               torch.zeros_like(plane_mask))

        _, _, bbox, mcr = largestConnectComponent(seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8),False)
        result = cv2.bitwise_and(src, src, mask=mcr.astype(np.uint8))
        # cv2.imshow('result_mcr', result)
        # cv2.waitKey(0)
        result_seg_mask = cv2.bitwise_and(src, src, mask=seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8))
        max_mask_return += (class_idx * torch.from_numpy(mcr).unsqueeze(0).unsqueeze(0).to(device).int())

        rect_size, rect_mask = maximum_internal_rectangle(result)

        depth_index = torch.nonzero(seg_mask.reshape(-1))
        # grad_index = torch.nonzero(rect_mask.reshape(-1).to(device))
        grad_index = torch.nonzero(torch.from_numpy(mcr).reshape(-1).to(device))
        grad_select = torch.gather(torch.from_numpy(sobel).reshape(-1).to(device), index=grad_index[:, 0], dim=0)
        include_flag = True
        # if grad is not None:
        #     # print('seg_lab', seg_labs_max)
        #     # cv2.imshow('result_mask', result_seg_mask)
        #     # cv2.waitKey(0)
        #     pc_grad_show = np.expand_dims(grad.cpu().detach().numpy(), axis=2).astype(np.uint8)
        #
        #     # result_mask = cv2.bitwise_and(pc_grad_show, pc_grad_show, mask=seg_mask[0,0].cpu().detach().numpy().astype(np.uint8)[:254,:318])
        #     result_mask = cv2.bitwise_and(pc_grad_show, pc_grad_show, mask=mcr.astype(np.uint8)[:254,:318])
        #     # cv2.imshow(' result',result)
        #
        #     lines = np.array(line_detect(result_mask, result, False)) #n,1,4
        #     area = 0
        #     print('num_line', len(lines))
        #     if len(lines) != 0:
        #         x_max = np.max([np.max(lines[:,:,0]), np.max(lines[:,:,2])])
        #         y_max = np.max([np.max(lines[:,:,1]), np.max(lines[:,:,3])])
        #         x_min = np.min([np.min(lines[:,:,0]), np.min(lines[:,:,2])])
        #         y_min = np.min([np.min(lines[:, :, 1]), np.min(lines[:, :, 3])])
        #         # print(x_max,x_min,y_max,y_min)
        #         area = (x_max-x_min)*(y_max-y_min)
        #         # print(area, np.sum(mcr))
        #         print('area_ratio', area/np.sum(mcr))
        #
        #         # print('area_ratio', area/np.sum(mcr))
        #     if len(lines)<10 or area/np.sum(mcr)< 0.3:
        #         print('include1')
        #         include_flag = True
        #     else:
        #         include_flag = False
        #         print('exclude1')
        depth_select = torch.gather(depth.reshape(-1).to(device), index=depth_index[:, 0], dim=0)
        # print('depth_select',torch.var(depth_select))
        # print('mean_total', torch.mean(depth))
        # print('mean_select', torch.mean(depth_select))
        # print('var_depth', torch.var(depth_select))
        # print('median', torch.median(depth))
        # print(grad_select)
        # print('grad_select', torch.mean(grad_select.float()))
        # print('grad_var', torch.var(grad_select.float()))
        # print('^^^^^^^^^^^^^^^^')
        # if torch.mean(depth_select) > torch.max(torch.median(depth), torch.mean(depth)):
        # print('current_plane_num', torch.sum(seg_mask))
        # print(max_mask_return)
        if torch.sum(seg_mask)>5000:
            # print('include2')
            # max_mask_return += (class_idx * seg_mask.to(device).int())
            # max_mask_return += (seg_mask.to(device).int))
            max_mask_return += (class_idx*torch.from_numpy(mcr).unsqueeze(0).unsqueeze(0).to(device).int())
        plane_mask_copy = ((plane_mask_copy.int()-seg_mask.int()) > 0).int()
        if flag == 'tgt':
            class_idx += 1

    # mask_depth = (mask_depth-seg_mask)>0
    # # print('mask_depth_copy', torch.sum(mask_depth))
    # total_seg_mask = plane_mask
    # if torch.sum(mask_depth)>0:
    #     seg_depth_flatten( = torch.nonzero(mask_depth.int().reshape(-1))
    #     # print(seg_depth_flatten.shape)  # n,1
    #     seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)
    #     # print(seg_img.shape)  # n
    #     seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)
    #     print('label_count', lab_count)
    #
    #     max_idx = torch.max(lab_count, dim=0)[1]
    #     seg_labs_max = seg_labs_select[max_idx]
    #     seg_reshape = seg_map.reshape(1, 1, 256, 320)
    #     seg_mask_second_plane = torch.where(seg_reshape == seg_labs_max, torch.ones_like(mask_depth),
    #                            torch.zeros_like(mask_depth))
    #
    #     depth_flatten = torch.nonzero(seg_mask_second_plane.reshape(-1))
    #
    #     second_depth = torch.gather(torch.from_numpy(sobel).reshape(-1).to(device), index=depth_flatten[:, 0], dim=0)
    #     # print('var~~~~~~~~~~~~~~~~~~~~~~~', torch.var(second_depth.float()))
    #
    #     labeled_img, max_label, _ = largestConnectComponent(seg_mask_second_plane[0, 0].detach().cpu().numpy().astype(np.uint8))
    #     label_mask_second = (labeled_img == max_label)
    #     if np.sum(label_mask_second) > 5000 and torch.var(second_depth.float()) < 1000:
    #         print('second_plane')
    #         total_seg_mask = plane_mask + torch.tensor(label_mask_second).unsqueeze(0).unsqueeze(0).to(device)

    return torch.tensor(max_mask_return)

def calculate_grad_for_line(Sobelxy,line):
    x1,y1,x2,y2 = line[0]
    x1,y1,x2,y2 = int(x1), int(y1),int(x2),int(y2)

    mid_pointx = int(0.5*(x1+x2))
    mid_pointy = int(0.5*(y1+y2))
    # grad1 = np.mean(Sobelxy[y1-3:y1+3, x1-3:x1+3])
    # grad2 = np.mean(Sobelxy[y2-3:y2+3, x2-3:x2+3])
    grad_mid = np.mean(Sobelxy[mid_pointy-6:mid_pointy+6, mid_pointx-6:mid_pointx+6])

    return grad_mid

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

def compare_with_tgt(candidate_lines, lines_tgt, matches):
    match_times = []
    proj_last_line = None
    if candidate_lines[0] is None:
        for line_idx_tgt in range(len(lines_tgt)):

            # normal_seg_ref1 = normal_seg_ref1.copy()
            x1_ref, y1_ref, x2_ref, y2_ref = lines_tgt[line_idx_tgt][0]

            if matches is not None:
                x1_proj_inv, y1_proj_inv = matches[0, int(y1_ref), int(x1_ref)].cpu().numpy()
                x2_proj_inv, y2_proj_inv = matches[0, int(y2_ref), int(x2_ref)].cpu().numpy()
                x1_ref = int((x1_proj_inv + 1) * 319 / 2)
                x2_ref = int((x2_proj_inv + 1) * 319 / 2)
                y1_ref = int((y1_proj_inv + 1) * 255 / 2)
                y2_ref = int((y2_proj_inv + 1) * 255 / 2)
                clamp
                if len(lines_tgt) == 1:
                    proj_last_line = [x1_ref, y1_ref, x2_ref, y2_ref]
        return 0, proj_last_line

    for line_idx_candi in range(len(candidate_lines)):
        match_time = 0
        line_candi = candidate_lines[line_idx_candi]
        x1, y1, x2, y2 = line_candi

        # x1_proj, y1_proj = matches[0][0, int(y1), int(x1)].cpu().numpy()
        # x2_proj, y2_proj = matches[0][0, int(y2), int(x2)].cpu().numpy()
        # x1_proj = (x1_proj + 1) * 319 / 2
        # x2_proj = (x2_proj + 1) * 319 / 2
        # y1_proj = (y1_proj + 1) * 255 / 2
        # y2_proj = (y2_proj + 1) * 255 / 2

        # print(x1,y1,x2,y2)
        # print(x1_proj,y1_proj,x2_proj,y2_proj)
        if x2 - x1 == 0:
            angle_candi = 90
        else:
            k_candi = (y2 - y1) / (x2 - x1)
            angle_candi = np.arctan(k_candi) * 57.29577

        # b_proj = y1_proj - k_proj * x1_proj
        # print('process ref1')
        # print('lines_tgt',lines_tgt)
        for line_idx_tgt in range(len(lines_tgt)):

            # normal_seg_ref1 = normal_seg_ref1.copy()
            x1_ref, y1_ref, x2_ref, y2_ref = lines_tgt[line_idx_tgt][0]

            if matches is not None:
                x1_proj_inv, y1_proj_inv = matches[0, int(y1_ref), int(x1_ref)].cpu().numpy()
                x2_proj_inv, y2_proj_inv = matches[0, int(y2_ref), int(x2_ref)].cpu().numpy()
                x1_ref = int((x1_proj_inv + 1) * 319 / 2)
                x2_ref = int((x2_proj_inv + 1) * 319 / 2)
                y1_ref = int((y1_proj_inv + 1) * 255 / 2)
                y2_ref = int((y2_proj_inv + 1) * 255 / 2)
                if len(lines_tgt) == 1:
                    proj_last_line = [clamp(0, x1_ref, 319), clamp(0,y1_ref, 255), clamp(0,x2_ref,319), clamp(0,y2_ref,255)]

            if x2_ref - x1_ref == 0:
                angle_ref = 90
            else:
                k_ref = (y2_ref - y1_ref) / (x2_ref - x1_ref)
                angle_ref = np.arctan(k_ref) * 57.29577
            # b_ref = y1_ref - k_ref * x1_ref
            # print('k_compare', angle_candi,angle_ref)
            # print('x_dis',np.min([x1,x2]),np.min([x1_ref, x2_ref]))

            if np.abs(np.abs(angle_candi)-np.abs(angle_ref)) < 6.5 and np.abs(np.min([x1,x2])-np.min([x1_ref, x2_ref]))<6.5:
                match_time += 1
        match_times.append(match_time)
    return match_times, proj_last_line

def calculate_loss(i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose, Q_last, Q_last_f, src_seg_show):
    # plane_mask_tgt = plane_masks[i + 1].int()
    # plane_mask_last = plane_masks[i]
    # print('i',i)

    # if i < (len(plane_masks) - 2):
    #     plane_mask_next = plane_masks[i + 2]
    # else:
    #     plane_mask_next = torch.ones_like(plane_mask_tgt)

    # last_loss_1 = 10
    # last_loss_2 = 0
    # last_loss_3 = 0
    # last_loss_F = 0
    # last_loss_match = 0
    #
    # max_iter = 100
    # online_iter = 0

    F_poses = [F_poses_with_inv[1][i], F_poses_with_inv[0][i + 1]]
    F_poses_inv = [F_poses_with_inv[0][i], F_poses_with_inv[1][i + 1]]
    # poses_gt_show = F_poses_with_inv[2][i]

    matches = [match_with_inv[1][i], match_with_inv[0][i + 1]]
    matches_inv = [match_with_inv[0][i], match_with_inv[1][i + 1]]

    # src_norm = torch.zeros((1, 3, 256, 320))
    # src_norm_comp = torch.zeros((1, 3, 256, 320))
    # mask_depth = 0
    # global_pose_plane = 0
    # loss_stop = False

    # generate predicted poses
    tgt_img = tgt_img.to(device)
    ref_imgs = [img.to(device) for img in ref_imgs]
    intrinsics = intrinsics.to(device)

    # grad_tmp = 0
    # grad = 0
    # fast_weights = 0

    poses, poses_inv, poses_f, poses_inv_f = compute_pose_with_inv(poseNet, tgt_img, ref_imgs, F_poses, F_poses_inv)
    tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)
    # when the distance between epipolar pose and the predicted on is less than a certain threshold
    # the epipolar pose is considered as reliable
    # and the implicit epipolar constraint is applied
    pred_pose_inv = poses_inv[0].detach()
    # print(poses_inv_show.shape)
    pred_pose_inv[:, :, 3] = pred_pose_inv[:, :, 3]/torch.sqrt(pred_pose_inv[:, 0, 3]**2 + pred_pose_inv[:, 1, 3]**2 + pred_pose_inv[:, 2, 3]**2)
    # print('poses', poses_inv[0])
    # print('poses_f', F_poses_inv[0])
    # print('pose_gt', poses_gt_show)
    # stop

    # tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs, grad_weight_depth, iter_idx)


    # if (Q_last is None) and (Q_last_f is None) and (src_seg_show is None):
    #     loss_1 = 0
    #     loss_3 = 0
    #     loss_match = 0
    #     loss_pc = [0,0
    # else:
    # loss1: photometric_loss
    #loss3: scale consistency loss
    # losspc:global point cloud consistency loss
    # photo_mask0: photometric overfitting coarse mask
    # lossmatch: the reprojection loss
    # photo_loss, geometry_loss, match_loss, photo_masks, pc_loss, Q_curr, fused_grad
    loss_1, loss_3, loss_match, photo_mask, loss_pc, Q_curr_f, _ = compute_photo_and_geometry_loss(tgt_img,
                                                                                                        ref_imgs,
                                                                                                        intrinsics,
                                                                                                        tgt_depth,
                                                                                                        ref_depths,
                                                                                                        poses_f,
                                                                                                        poses_inv_f,
                                                                                                        1, 1, 1, 0,
                                                                                                        'zeros',
                                                                                                        src_seg_show,
                                                                                                        matches,
                                                                                                         matches_inv, global_pose, Q_last_f)
    loss_1_ori, loss_3_ori, loss_match_ori, photo_mask_ori, loss_pc_ori, Q_curr_ori, fused_grad_ori = compute_photo_and_geometry_loss(tgt_img,
                                                                                                        ref_imgs,
                                                                                                        intrinsics,
                                                                                                        tgt_depth,
                                                                                                        ref_depths,
                                                                                                        poses,
                                                                                                        poses_inv,
                                                                                                        1, 1, 1, 0,
                                                                                                        'zeros',
                                                                                                        src_seg_show,
                                                                                                        matches,
                                                                                                        matches_inv, global_pose, Q_last)
    # if show:



    # use_mask = False
    # loss_pc_ori[2]
    loss_2 = compute_smooth_loss(tgt_depth, tgt_img, fused_grad_ori)

    # print('loss_2', loss_2)

    # the explicit epipolar constraint
    loss_F = compute_F_loss(F_poses, poses, poses_inv) + compute_F_loss(F_poses_inv, poses_inv, poses)

    # loss_normal_adj = 0
    # loss_normal_single = 0
    # loss_normal = 0

    w1 = 1.0
    w2 = 0.1
    w_m = 1.5
    # w_f = 5

    # if iter_idx > 0:
    #     w_f = 1
    # else:
    #     w_f = 0
    # # print('tgt_depth', tgt_depth)
    # if iter_idx > 10:
    #     w_pc = 1.0
    # else:
    #     w_pc = 0
    # # if iter_idx > 10:
    # #     w1_new = 2
    # # else:
    # #     w1_new = 0
    # # if i < 9 or (i > 19 and i <24) or (i > 35 and i < 42) and i> 51:
    # if i < 9 or (i > 28 and i <37) or (i > 50 and i < 64) and i> 78:
    # # if i < 10 or (i > 25 and i <35) or (i > 42 and i < 49) and i> 52:
    #     w_new = 0.5
    # else:
    #     w_new = 0
    # if i == 0:
    #     w_pc = 0
    # else:
    #     w_pc = 0.1
    # print('loss_pc',loss_pc_ori[0])
    # stop
    # loss = w1 * loss_1 + 0.5 * loss_3 + w2 * loss_2 + w_m*loss_match + w_f*loss_F + loss_1_ori + 0.5*loss_3_ori
    trans1 = pred_pose_inv[0, :3, 3]
    trans2 = F_poses_inv[0][0, :3, 3]
    cos_theta = (trans1[0]*trans2[0] + trans1[1]*trans2[1] + trans1[2] * trans2[2])/ \
                (torch.sqrt(trans1[0]**2 + trans1[1]**2 + trans1[2]**2)*torch.sqrt(trans2[0]**2 + trans2[1]**2 + trans2[2]**2))
    angle = np.degrees(torch.arccos(cos_theta).detach().cpu().detach())
    # print('cos_theta', angle)
    # if torch.abs(poses_inv_show[:, 2, 3]) < 0.95 or torch.abs(F_poses_inv[0][:, 2, 3]) < 0.95:
    if angle >10:
        w_f = 0
    else:
        w_f = 1
    # print('w_f',w_f)
    loss = w2 * loss_2 + w1*loss_1_ori + 0.5*loss_3_ori +w_f*(0.1*(loss_1 + 0.5 * loss_3))\
           # +0.1 * (loss_pc_ori[0] + loss_pc[0])
           # +w_f*loss_F
           # + loss_match_ori+w_f*(0.1*(loss_1 + 0.5 * loss_3)) +0.1 * (loss_pc_ori[0] + loss_pc[0])
           # +w_f*loss_F
           # +w_f*(0.1*(loss_1 + 0.5 * loss_3))
           # + loss_match_ori+w_f*(0.1*(loss_1 + 0.5 * loss_3)) +0.1 * (loss_pc_ori[0] + loss_pc[0])
           # +w_f*loss_F
           # +w_f*(0.1*(loss_1 + 0.5 * loss_3))\
           # +0.1 * (loss_pc_ori[0] + loss_pc[0])

           # +w_f*loss_F
           # +w_f*(0.1*(loss_1 + 0.5 * loss_3))
           # + loss_match_ori +w_f*(0.1*(loss_1 + 0.5 * loss_3))
           # + loss_match_ori +w_f*(0.1*(loss_1 + 0.5 * loss_3))
           # +w_f*(0.1*(loss_1 + 0.5 * loss_3)) + loss_match_ori
           # + loss_match_ori+w_f*(0.1*(loss_1 + 0.5 * loss_3)) + loss_match_ori
    # +(0.1 * (loss_1 + 0.5 * loss_3))
           # + loss_F\
           # +w_f*(0.1 * (loss_1 + 0.5 * loss_3))
           # + 0.1 * (loss_1 + 0.5 * loss_3)
           # +loss_F
           # +loss_match_ori\
           # +loss_F \
           # + 0.1 * (loss_1 + 0.5 * loss_3 + loss_match)\
           # +0.1 * (loss_pc_ori[0] + loss_pc[0])
           # +loss_F
           # +1.0 * (loss_pc_ori[0] + loss_pc[0])
    # + loss_pc[0])\
           # +1.5 * (loss_1 + 0.5 * loss_3 + loss_match)
           # +loss_match_ori+ 2.5*(loss_1 +0.5*loss_3+loss_match) + 5.5 * (loss_pc_ori[0] + loss_pc[0])
           # + loss_match_ori+ loss_match
           # + w_pc * (loss_pc_ori[0] + loss_pc[0])
           # + (loss_1 +0.5*loss_3+ loss_match)\
    # + w_pc * (loss_pc_ori[0] + loss_pc[0])
    # print('loss_pc', loss_pc_ori[0],loss_pc[0] )
           # + loss_match_ori
           # + 0.1*(loss_1 +0.5*loss_3+ loss_match)\
           # + w_pc * (loss_pc_ori[0] + loss_pc[0])
           # + loss_match_ori + 0.5 * (0.5 * loss_3 + loss_1 + loss_match)
    # + 0.1*(0.5*loss_3 + loss_1 + loss_match)
           # + w_pc*(loss_pc[0])
           # + 0.05*loss_3 + 0.1*loss_1 + 0.1*loss_pc[0] + 0.1*loss_pc_ori[0] + loss_match_ori + 0.1*loss_match


    return loss_1_ori, loss, photo_mask_ori, Q_curr_f, Q_curr_ori, fused_grad_ori
# noraml_ref1_normalize = None
last_z = None
last_x = None
def calculate_loss_norm(i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose_curr, connected_region, Q_last,Q_last_f, src_seg_show):
    # global noraml_ref1_normalize
    # global initial_norm
    # plane_mask_tgt = plane_masks[i + 1].int()
    # plane_mask_tgt = plane_mask
    # boundary_mask = torch.zeros_like(plane_mask_tgt)
    # boundary_mask[:,:,1:-1, 1:319] = 1
    # plane_mask_tgt = plane_mask_tgt*boundary_mask\
                     # *plane_mask
    # plane_mask_last = plane_masks[i]

    # if i < (len(plane_masks) - 2):
    #     plane_mask_next = plane_masks[i + 2]
    # else:
    #     plane_mask_next = torch.ones_like(plane_mask_tgt)

    # last_loss_1 = 10
    # last_loss_2 = 0
    # last_loss_3 = 0
    # last_loss_F = 0
    # last_loss_match = 0
    #
    # max_iter = 100
    # online_iter = 0

    F_poses = [F_poses_with_inv[1][i], F_poses_with_inv[0][i + 1]]
    F_poses_inv = [F_poses_with_inv[0][i], F_poses_with_inv[1][i + 1]]

    matches = [match_with_inv[1][i], match_with_inv[0][i + 1]]
    matches_inv = [match_with_inv[0][i], match_with_inv[1][i + 1]]

    # src_norm = torch.zeros((1, 3, 256, 320))
    # src_norm_comp = torch.zeros((1, 3, 256, 320))
    # mask_depth = 0
    # global_pose_plane = 0
    # loss_stop = False

    # load imgs and intrinsics
    tgt_img = tgt_img.to(device)
    ref_imgs = [img.to(device) for img in ref_imgs]
    intrinsics = intrinsics.to(device)

    # grad_tmp = 0
    # grad = 0
    # fast_weights = 0

    poses, poses_inv, poses_f, poses_inv_f = compute_pose_with_inv(poseNet, tgt_img, ref_imgs, F_poses, F_poses_inv)
    # print('poses',poses)

    tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)

    # loss_normal_single = 0
    # loss_normal_adj = 0
    # tmp_mask = torch.zeros_like(plane_mask_tgt)
    # for n in range(len(connected_region)):
    #     tmp_mask += torch.from_numpy(connected_region[n] > 0).to(device).unsqueeze(0).unsqueeze(0).int()
    #     # if torch.sum(plane_mask_tgt) > 0:
    #     #     plane_mask_tgt *= boundary_mask
    #     #     # if iter_idx == 1 and i > 7:
    #     #     #     src = ((tgt_img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
    #     #     #     src = np.transpose(src, (1, 2, 0)).astype(np.uint8)
    #     #     #     mask_gc_draw = plane_mask_tgt[0, 0].detach().cpu().numpy().astype(np.uint8)
    #     #     #     result = cv2.bitwise_and(src, src, mask=mask_gc_draw)
    #     #     #     # cv2.imshow('result', result)
    #     #     #     # cv2.waitKey(0)
    #     #
    #     #     # print('normal_tgt', normal_tgt.reshape([1,3,-1])[:,:,:100])
    #     #
    #     #     # normal_ref1_flat = projected_normal_map.reshape(1, 3, -1)  # [B, 3, H*W]
    #     #
    #     #     # normal_ref2_flat = normal_ref2.reshape(1, 3, -1)  # [B, 3, H*W]
    #     #     # normal_tgt_flat = normal_tgt.reshape(1, 3, -1)  # [B, 3, H*W]
    #     #     # pose_mat1 = pose_vec2mat(poses_inv[0])
    #     #     # pose_mat2 = pose_vec2mat(poses[0])
    #     #
    #     #     # pose_mat2 = poses[0]
    #     #     # pose_mat2_reshape = pose_mat2.squeeze(0)
    #     #
    #     #     # pose_mat = pose_vec2mat(poses[0]).squeeze(0)
    #     #     #
    #     #     # print('global_pose',global_pose_curr)
    #     #     # print('pose2', pose_mat2_reshape)
    #     #     # pose_mat_h = torch.cat([pose_mat2_reshape, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
    #     #     # global_pose_plane = torch.from_numpy(global_pose_curr).to(device).float() @ pose_mat_h
    #     #     # global_pose_plane = global_pose_plane[:3, :].unsqueeze(0)
    #     #     # # pose_mat2 = pose_vec2mat(poses_inv[1])
    #     #     # # print(torch.mean(normal_ref1_flat))
    #     #     # # print(pose_mat1[0].shape)  ##1,6
    #     #     # # stop
    #     #
    #     #     non0_index = torch.nonzero(plane_mask_tgt.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
    #     #                                                                                                      1)  # 1,3,n
    #     #     norm_select_tgt = torch.gather(normal_tgt_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
    #     #
    #     #     # grad_normal = torch.abs(norm_select_tgt[:, :, :-1] - norm_select_tgt[:, :, 1:])
    #     #     # print('norm_select_tgt',norm_select_tgt)
    #     #     normal_x_var = torch.var(norm_select_tgt[:, 0, :], dim=-1)
    #     #     normal_y_var = torch.var(norm_select_tgt[:, 1, :], dim=-1)
    #     #     normal_z_var = torch.var(norm_select_tgt[:, 2, :], dim=-1)
    #     #
    #     #     if i == 0 and n == 0:
    #     #         non0_index = torch.nonzero(plane_mask_tgt.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
    #     #                                                                                                          1)  # 1,3,n
    #     #         Q_tgt_select_global = torch.gather(loss_pc_ori[1].reshape(1, 3, -1), index=non0_index, dim=2)
    #     #         pc_array = np.array(Q_tgt_select_global[0].transpose(1, 0).detach().cpu().numpy(), dtype=np.float32)
    #     #         cloud = pcl.PointCloud()
    #     #         cloud.from_array(pc_array)
    #     #
    #     #         # filter_vox = cloud.make_voxel_grid_filter()
    #     #         # filter_vox.set_leaf_size(0.005, 0.005, 0.005)
    #     #         # cloud_filtered = filter_vox.filter()
    #     #         # print(cloud_filtered.size)
    #     #
    #     #         seg = cloud.make_segmenter_normals(ksearch=50)  # 平面分割
    #     #         seg.set_optimize_coefficients(True)
    #     #         seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    #     #         seg.set_method_type(pcl.SAC_RANSAC)
    #     #         seg.set_distance_threshold(0.1)
    #     #         seg.set_normal_distance_weight(0.3)
    #     #         seg.set_max_iterations(100)
    #     #         indices, coefficients = seg.segment()
    #     #         # print('indice',indices)
    #     #         # stop
    #     #         # print(-coefficients[0]/coefficients[2],-coefficients[1]/coefficients[2],-coefficients[2]/coefficients[2],-coefficients[3]/coefficients[2])
    #     #         q = cloud.extract(indices, negative=False)
    #     #         # print(q.size)
    #     #         # stop
    #     #         filter_stat = q.make_statistical_outlier_filter()
    #     #         filter_stat.set_mean_k(50)
    #     #         filter_stat.set_std_dev_mul_thresh(1.0)
    #     #         p = filter_stat.filter()
    #     #
    #     #         ne = cloud.make_NormalEstimation()
    #     #         # ne = p.make_NormalEstimation()
    #     #         ne.set_KSearch(20)
    #     #         normals = ne.compute()
    #     #         # print('normals', normals[0])
    #     #         normals_arr = normals.to_array()[:, :3]  # n,3
    #     #         normals_tensor = torch.from_numpy(normals_arr)
    #     #         # print(normals_tensor.shape)
    #     #         normals_tensor = normals_tensor / normals_tensor[:, 2].unsqueeze(1)
    #     #         # print(normals_tensor.shape) #n,3
    #     #         # stop
    #     #         a = torch.mean(normals_tensor[:, 0], dim=0)
    #     #         b = torch.mean(normals_tensor[:, 1], dim=0)
    #     #         c = 1
    #     #
    #     #         # print(initial_norm.shape) #3
    #     #         x_train = Q_tgt_select_global[0, 0, :]
    #     #         y_train = Q_tgt_select_global[0, 1, :]
    #     #         z_train = Q_tgt_select_global[0, 2, :]
    #     #         d = torch.mean(a * x_train + b * y_train + z_train)
    #     #         print('d', d)
    #     #         initial_norm = torch.tensor([a, b, c, d], device=device).detach()
    #     #
    #     #     grad_disp_x = torch.abs(normal_tgt_normalize[:, :, :, :-1] - normal_tgt_normalize[:, :, :, 1:])
    #     #     # grad_disp_x = torch.abs(normal_tgt[:, :, :, :-1] - normal_tgt[:, :, :, 1:])
    #     #     grad_disp_y = torch.abs(normal_tgt_normalize[:, :, :-1, :] - normal_tgt_normalize[:, :, 1:, :])
    #     #     # grad_disp_y = torch.abs(normal_tgt[:, :, :-1, :] - normal_tgt[:, :, 1:, :])
    #     #
    #     #     # loss_normal_single += torch.abs(torch.mean(grad_disp_x * plane_mask_tgt[:, :, :, :-1])) + torch.abs(torch.mean(grad_disp_y* plane_mask_tgt[:, :, :-1, ]))
    #     #     # print('last_norm', last_norm[:,:,0])
    #     #
    #     #     norm_select_ref1 = torch.gather(noraml_ref1_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
    #     #
    #     #     # mean_last_norm = torch.mean(norm_select_ref1, dim=-1)
    #     #     grad_depth_x = torch.abs(tgt_depth[0][:, :, :, :-1] - tgt_depth[0][:, :, :, 1:])
    #     #     grad_depth_y = torch.abs(tgt_depth[0][:, :, :-1, :] - tgt_depth[0][:, :, 1:, :])
    #     #     depth_select = torch.gather(tgt_depth[0].reshape(1, 1, -1), index=non0_index[:, :1, :], dim=2)
    #     #
    #     #     non0_index_x = torch.nonzero(plane_mask_tgt[:, :, :, :-1].reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0)
    #     #     # 1,3,n
    #     #     grad_depth_x_select = torch.gather(grad_depth_x.reshape(1, 1, -1), index=non0_index_x, dim=2)
    #     #
    #     #     non0_index_y = torch.nonzero(plane_mask_tgt[:, :, :-1, :].reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(
    #     #         0)  # 1,3,n
    #     #     grad_depth_y_select = torch.gather(grad_depth_y.reshape(1, 1, -1), index=non0_index_y, dim=2)
    #     #
    #     #     loss_normal_single += (
    #     #                 torch.mean(torch.abs(normal_x_var)) + torch.mean(torch.abs(normal_y_var)) + torch.mean(
    #     #             torch.abs(normal_z_var)))
    #     #     # +torch.abs(torch.mean(grad_disp_x * plane_mask_tgt[:, :, :, :-1])) + torch.abs(
    #     #     #     torch.mean(grad_disp_y * plane_mask_tgt[:, :, :-1, ])))
    #     #     # + 0.01*torch.mean(torch.abs(norm_select_tgt-mean_last_norm.unsqueeze(2).repeat(1,1,norm_select_tgt.shape[2]).detach()))
    #     #     # if n == 0:
    #     #     loss_normal_adj += torch.mean(torch.abs(norm_select_ref1 - norm_select_tgt))
    #     #     # loss_normal_adj += torch.mean(torch.abs(initial_norm.unsqueeze(0).unsqueeze(2).repeat(1,1,norm_select_tgt.shape[2])-norm_select_tgt))
    #     #     print('loss_normal_adj1', loss_normal_adj)
    #     #     loss_depth_grad += torch.mean(torch.abs(grad_depth_x_select)) + torch.mean(torch.abs(grad_depth_y_select)) \
    #     #         # + torch.var(depth_select[0,0])
    #     #
    #     #     # print('loss_normal', loss_normal)
    #     #     # print('loss_normal_single', loss_normal_single)
    #     #     # print('loss_normal_adj', loss_normal_adj)
    #     # print('F', poses, poses_inv)
    #
    #     # else:
    #     #     last_norm_return = last_norm

    print('seg_show', src_seg_show.shape)
    loss_1, loss_3, loss_match, photo_mask, loss_pc, Q_curr_f, _ = compute_photo_and_geometry_loss(tgt_img,
                                                                                                        ref_imgs,
                                                                                                        intrinsics,
                                                                                                        tgt_depth,
                                                                                                        ref_depths,
                                                                                                        poses_f,
                                                                                                        poses_inv_f,
                                                                                                        1, 1, 1, 0,
                                                                                                        'zeros',
                                                                                                        src_seg_show,
                                                                                                        matches,
                                                                                                        matches_inv,global_pose_curr,Q_last_f)
    # if show:
    loss_1_ori, loss_3_ori, loss_match_ori, photo_mask_ori, loss_pc_ori, Q_curr_ori, fused_grad_ori = compute_photo_and_geometry_loss(tgt_img,
                                                                                                        ref_imgs,
                                                                                                        intrinsics,
                                                                                                        tgt_depth,
                                                                                                        ref_depths,
                                                                                                        poses,
                                                                                                        poses_inv,
                                                                                                        1, 1, 1, 0,
                                                                                                        'zeros',
                                                                                                        src_seg_show,
                                                                                                        matches,
                                                                                                        matches_inv, global_pose_curr, Q_last)



    # use_mask = False
    loss_2 = compute_smooth_loss(tgt_depth, tgt_img, fused_grad_ori)

    loss_F = compute_F_loss(F_poses, poses, poses_inv) + compute_F_loss(F_poses_inv, poses_inv, poses)


    # loss_normal_adj = 0
    # loss_normal_single = 0
    # loss_depth_grad = 0
    # loss_normal = 0
    # global last_norm
    last_norm_return = None

    normal_tgt = calculate_all_norm(tgt_depth[0][0, 0])  # 1,3,256,320
    # print('last_norm', last_norm)
    # if iter_idx == 1:
    normal_ref1 = calculate_all_norm(ref_depths[0][0][0, 0])
    projected_normal_map = F.grid_sample(
        normal_ref1.float(), matches[0].float(), padding_mode='reflection', align_corners=False)
    noraml_ref1_normalize = F.normalize(projected_normal_map, dim=1).reshape([1, 3, 256, 320])
    # normal_ref2 = calculate_all_norm(ref_depths[1][0][0, 0])
    normal_tgt_flat = normal_tgt.reshape(1, 3, -1)  # [B, 3, H*W]
    # print('normal_tgt_flat.shape',normal_tgt_flat.shape)

    # # pose_mat1 = poses[0]
    pose_mat1 = poses[0]
    # # print(pose_mat1.shape) #1,3,4
    pose_mat1_reshape = pose_mat1.squeeze(0)
    pose_mat_h = torch.cat([pose_mat1_reshape, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
    global_pose_plane = torch.from_numpy(global_pose_curr).to(device).float() @ pose_mat_h
    # # # print(global_pose_plane.shape) #4,4
    global_pose_plane = global_pose_plane[:3,:].unsqueeze(0)
    # normal_tgt = pose_mat1[:, :, :3] @ normal_tgt_flat \
                 # + pose_mat1[:, :, -1:]
    normal_tgt = global_pose_plane[:, :, :3] @ normal_tgt_flat \
                 # + global_pose_plane[:, :, -1:]
    # # # pcoords_ref2 = pose_mat2[:, :, :3] @ normal_ref2_flat + pose_mat2[:, :, -1:]  #1,3,n
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    # print('initial_norm',initial_norm)
    # print('global_pose_plane',global_pose_plane)
    # # print('normal_tgt_flat',normal_tgt_flat)
    # # print('normal_tgt',normal_tgt)
    # # print(normal_tgt.shape)




    normal_tgt_normalize = F.normalize(normal_tgt, dim=1).reshape([1,3, 256,320])
    # print('normal_tgt_normalize', normal_tgt_normalize.shape) #1,3,n
    # eyop


    loss_normal_single = 0
    loss_normal_per = 0
    loss_normal_adj = 0
    loss_wall = 0
    plane_mask_left = None
    plane_mask_right = None
    src = ((tgt_img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
    src = np.transpose(src, (1, 2, 0)).astype(np.uint8)
    P_d = connected_region[0]
    P_p = connected_region[1]
    for n in range(len(P_d)):
        # print(P_d[n])
        # stop
        plane_mask_left = torch.from_numpy(P_d[n]>0).to(device).unsqueeze(0).unsqueeze(0).int()
        # if np.stack(connected_region[n]).shape[0] != 2:
        #     plane_mask_left = torch.from_numpy(connected_region[n]>0).to(device).unsqueeze(0).unsqueeze(0).int()
        #     # print(plane_mask_left.shape)
        #     # stop
        # else:
        #     plane_mask_left = torch.from_numpy(connected_region[n][0] > 0).to(device).unsqueeze(0).unsqueeze(0).int()
        #     plane_mask_right = torch.from_numpy(connected_region[n][1]> 0).to(device).unsqueeze(0).unsqueeze(0).int()
        #     # mask_gc_draw_right = plane_mask_right[0, 0].detach().cpu().numpy().astype(np.uint8)
        #     # result_right = cv2.bitwise_and(src, src, mask=mask_gc_draw_right)
        #     # cv2.imshow('result_right', result_right)
        #     # cv2.waitKey(0)
        # if torch.sum(plane_mask_left) > 0:
        #     plane_mask_left *= boundary_mask

        # mask_gc_draw_left = plane_mask_left[0, 0].detach().cpu().numpy().astype(np.uint8)
        # result_left = cv2.bitwise_and(src, src, mask=mask_gc_draw_left)
        # # if show_flag:
        # # cv2.imshow('result_left', result_left)
        # # cv2.waitKey(0)

        # loss_instance = 0
        # Q_plane = 0

        # Q_tgt = pixel2cam(tgt_depth[0].squeeze(1), intrinsics.inverse())

        # pose_mat1 = poses[0]
        # # print(pose_mat1.shape) #1,3,4
        # pose_mat1_reshape = pose_mat1.squeeze(0)
        # pose_mat_h = torch.cat([pose_mat1_reshape, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
        # global_pose_plane = torch.from_numpy(global_pose_curr).to(device).float() @ pose_mat_h
        # # # print(global_pose_plane.shape) #4,4
        # global_pose_plane = global_pose_plane[:3, :].unsqueeze(0)

        # Q_tgt_flat = Q_tgt.reshape(tgt_depth[0].shape[0], 3, -1)
        # rot_inv, tr_inv = global_pose_plane[:, :3, :3], global_pose_plane[:, :3, -1:]
        # Q_global_tgt = rot_inv @ Q_tgt_flat + tr_inv

        non0_index = torch.nonzero(plane_mask_left.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
                                                                                                             1)  # 1,3,n
        norm_select_tgt = torch.gather(normal_tgt_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
        normal_x_var = torch.var(norm_select_tgt[:, 0, :], dim=-1)
        normal_y_var = torch.var(norm_select_tgt[:, 1, :], dim=-1)
        normal_z_var = torch.var(norm_select_tgt[:, 2, :], dim=-1)
        loss_normal_single += (torch.mean(torch.abs(normal_x_var)) + torch.mean(torch.abs(normal_y_var)) + torch.mean(
            torch.abs(normal_z_var)))

    for n in range(len(P_p)):
        plane_mask_left = torch.from_numpy(P_p[n][0] > 0).to(device).unsqueeze(0).unsqueeze(0).int()
        plane_mask_right = torch.from_numpy(P_p[n][1] > 0).to(device).unsqueeze(0).unsqueeze(0).int()
        # loss_normal_per = 0

        # print('x_coord', torch.min(non0_index[0, 0, :] % 320))
        # if False:
        # # if torch.min(non0_index[0, 0, :] % 320) >4000:
        #     plane_mask_right = plane_mask_left
        #     mask_gc_draw_right = plane_mask_right[0, 0].detach().cpu().numpy().astype(np.uint8)
        #     result_right = cv2.bitwise_and(src, src, mask=mask_gc_draw_right)
        #     norm_select_tgt = torch.gather(normal_tgt_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
        #     # if show_flag:
        #     # cv2.imshow('result_right', result_right)
        #     # cv2.waitKey(0)
        # else:
        #     mask_gc_draw_left = plane_mask_left[0, 0].detach().cpu().numpy().astype(np.uint8)
        #     result_left = cv2.bitwise_and(src, src, mask=mask_gc_draw_left)
        #     # if show_flag:
        #     # cv2.imshow('result_left', result_left)
        #     # cv2.waitKey(0)
        #     norm_select_tgt = torch.gather(normal_tgt_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
        #
        #     # tgt_non_zero_idx_forpc = torch.nonzero(plane_mask_left.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0)
        #     # Q_tgt_select_global_forpc = torch.gather(Q_global_tgt.reshape(1, 3, -1),
        #     #                                          index=tgt_non_zero_idx_forpc.repeat(1, 3, 1), dim=2)
        #     # z_train = Q_tgt_select_global_forpc[0, 2]
        #
        #     # print(z_train)
        #     # stop
        #     # Q_plane += 10*torch.abs(torch.var(z_train)) \
        #           # + torch.sum(torch.abs(z_train-0.7))
        #     # print('Q_plane_left', Q_plane)
        # if plane_mask_right is not None:
        #     # non0_index_right = torch.nonzero(plane_mask_right.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
        #     #                                                                                                   1)  # 1,3,n
        #     # norm_select_tgt_right = torch.gather(normal_tgt_normalize.reshape(1, 3, -1), index=non0_index_right, dim=2)
        #     # mean_left = torch.mean(norm_select_tgt_right, dim=-1)
        #     # mean_right = torch.mean(norm_select_tgt, dim=-1)
        #     # loss_instance = torch.mean(torch.abs(torch.sum(mean_left*mean_right, dim=1)), dim=-1)
        #
        #     # tgt_non_zero_idx_forpc = torch.nonzero(plane_mask_right.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0)
        #     # Q_tgt_select_global_forpc = torch.gather(Q_global_tgt.reshape(1, 3, -1),
        #     #                                          index=tgt_non_zero_idx_forpc.repeat(1, 3, 1), dim=2)
        #     # print('loss_instance', loss_instance)
        #     # x_train = Q_tgt_select_global_forpc[0, 0]
        #     # y_train = Q_tgt_select_global_forpc[0, 1]
        #     # x_train = Q_tgt_select_global_forpc[0, 0]
        #     # z_train = Q_tgt_select_global_forpc[0, 2]
        #     # x_train = Q_tgt_select_global_forpc[0, 0]
        #     # last_x = torch.mean(x_train)
        #     # print('x_train', x_train)
        #     # Q_plane += 10*torch.abs(torch.var(z_train))
        #     # print('Q_plane', Q_plane)
        #     # else:
        #
        #     normal_x_var = torch.var(norm_select_tgt[:,0,:], dim=-1)
        #     normal_y_var = torch.var(norm_select_tgt[:,1,:], dim=-1)
        #     normal_z_var = torch.var(norm_select_tgt[:,2,:], dim=-1)
        #
        #     # if i == 0 and n == 0:
        #     #     non0_index = torch.nonzero(plane_mask_tgt.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
        #     #                                                                                                      1)  # 1,3,n
        #     #     Q_tgt_select_global = torch.gather(loss_pc_ori[1].reshape(1, 3, -1), index=non0_index, dim=2)
        #     #     pc_array = np.array(Q_tgt_select_global[0].transpose(1, 0).detach().cpu().numpy(), dtype=np.float32)
        #     #     cloud = pcl.PointCloud()
        #     #     cloud.from_array(pc_array)
        #     #
        #     #     # filter_vox = cloud.make_voxel_grid_filter()
        #     #     # filter_vox.set_leaf_size(0.005, 0.005, 0.005)
        #     #     # cloud_filtered = filter_vox.filter()
        #     #     # print(cloud_filtered.size)
        #     #
        #     #     seg = cloud.make_segmenter_normals(ksearch=50)  # 平面分割
        #     #     seg.set_optimize_coefficients(True)
        #     #     seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        #     #     seg.set_method_type(pcl.SAC_RANSAC)
        #     #     seg.set_distance_threshold(0.1)
        #     #     seg.set_normal_distance_weight(0.3)
        #     #     seg.set_max_iterations(100)
        #     #     indices, coefficients = seg.segment()
        #     #     # print('indice',indices)
        #     #     # stop
        #     #     # print(-coefficients[0]/coefficients[2],-coefficients[1]/coefficients[2],-coefficients[2]/coefficients[2],-coefficients[3]/coefficients[2])
        #     #     q = cloud.extract(indices, negative=False)
        #     #     # print(q.size)
        #     #     # stop
        #     #     filter_stat = q.make_statistical_outlier_filter()
        #     #     filter_stat.set_mean_k(50)
        #     #     filter_stat.set_std_dev_mul_thresh(1.0)
        #     #     p = filter_stat.filter()
        #     #
        #     #     ne = cloud.make_NormalEstimation()
        #     #     # ne = p.make_NormalEstimation()
        #     #     ne.set_KSearch(20)
        #     #     normals = ne.compute()
        #     #     # print('normals', normals[0])
        #     #     normals_arr = normals.to_array()[:, :3]  # n,3
        #     #     normals_tensor = torch.from_numpy(normals_arr)
        #     #     # print(normals_tensor.shape)
        #     #     normals_tensor = -normals_tensor / normals_tensor[:, 2].unsqueeze(1)
        #     #     # print(normals_tensor.shape) #n,3
        #     #     # stop
        #     #     a = torch.mean(normals_tensor[:, 0], dim=0)
        #     #     b = torch.mean(normals_tensor[:, 1], dim=0)
        #     #     c = -1
        #     #
        #     #     # print(initial_norm.shape) #3
        #     #     x_train = Q_tgt_select_global[0, 0, :]
        #     #     y_train = Q_tgt_select_global[0, 1, :]
        #     #     z_train = Q_tgt_select_global[0, 2, :]
        #     #     d = torch.mean(z_train-a*x_train-b*y_train)
        #     #     # print('d',d)
        #     #     initial_norm = torch.tensor([a, b,c, d], device=device).detach()
        #
        #
        #
        #     grad_disp_x = torch.abs(normal_tgt_normalize[:, :, :, :-1] - normal_tgt_normalize[:, :, :, 1:])
        #     # grad_disp_x = torch.abs(normal_tgt[:, :, :, :-1] - normal_tgt[:, :, :, 1:])
        #     grad_disp_y = torch.abs(normal_tgt_normalize[:, :, :-1, :] - normal_tgt_normalize[:, :, 1:, :])
        #     # grad_disp_y = torch.abs(normal_tgt[:, :, :-1, :] - normal_tgt[:, :, 1:, :])
        #
        #     # loss_normal_single += torch.abs(torch.mean(grad_disp_x * plane_mask_tgt[:, :, :, :-1])) + torch.abs(torch.mean(grad_disp_y* plane_mask_tgt[:, :, :-1, ]))
        #     # print('last_norm', last_norm[:,:,0])
        #
        #
        #     # norm_select_ref1 = torch.gather(noraml_ref1_normalize.reshape(1, 3, -1), index=non0_index, dim=2)
        #
        #     # mean_last_norm = torch.mean(norm_select_ref1, dim=-1)
        #     grad_depth_x = torch.abs(tgt_depth[0][:, :, :, :-1] - tgt_depth[0][:, :, :, 1:])
        #     grad_depth_y = torch.abs(tgt_depth[0][:, :, :-1, :] - tgt_depth[0][:, :, 1:, :])
        #     depth_select = torch.gather(tgt_depth[0].reshape(1, 1, -1), index=non0_index[:,:1,:], dim=2)
        #
        #     non0_index_x = torch.nonzero(plane_mask_tgt[:, :, :, :-1].reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0)
        #                                                                                                      # 1,3,n
        #     grad_depth_x_select = torch.gather(grad_depth_x.reshape(1, 1, -1), index=non0_index_x, dim=2)
        #
        #     non0_index_y = torch.nonzero(plane_mask_tgt[:, :, :-1, :].reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(
        #         0)  # 1,3,n
        #     grad_depth_y_select = torch.gather(grad_depth_y.reshape(1, 1, -1), index=non0_index_y, dim=2)
        #
        #     loss_normal_single += (torch.mean(torch.abs(normal_x_var))+torch.mean(torch.abs(normal_y_var))+torch.mean(torch.abs(normal_z_var)) )
        #                 # +torch.abs(torch.mean(grad_disp_x * plane_mask_tgt[:, :, :, :-1])) + torch.abs(torch.mean(grad_disp_y * plane_mask_tgt[:, :, :-1, ])))
        #                         # + 0.01*torch.mean(torch.abs(norm_select_tgt-mean_last_norm.unsqueeze(2).repeat(1,1,norm_select_tgt.shape[2]).detach()))
        #     # if n == 0:
        #     # loss_normal_adj += torch.mean(torch.abs(norm_select_ref1-norm_select_tgt))
        #     # division = torch.sqrt(initial_norm[0]**2 + initial_norm[1]**2 +1)
        #     # normalize_initial_norm = initial_norm[:3]/division
        #     # loss_normal_adj += torch.mean(torch.abs(normalize_initial_norm.unsqueeze(0).unsqueeze(2).repeat(1,1,norm_select_tgt.shape[2])-norm_select_tgt))
        #     # # print('loss_normal_adj1', loss_normal_adj)
        #     # loss_depth_grad += torch.mean(torch.abs(grad_depth_x_select)) + torch.mean(torch.abs(grad_depth_y_select)) \
        #                        # + torch.var(depth_select[0,0])
        #
        #
        #     # print('loss_normal', loss_normal)
        #     # print('loss_normal_single', loss_normal_single)
        #     # print('loss_normal_adj', loss_normal_adj)
        # print('F', poses, poses_inv)

        # else:
        #     last_norm_return = last_norm
    w1 = 0.1
    w2 = 0.1
    w_m = 1.5
    # if i > 7:
    #     p_d = 5
    # else:
    #     p_d = 0.1

    # if i < 9 or (i > 19 and i < 24) or (i > 35 and i < 42) and i > 51:
    #     # if i < 12 or (i > 25 and i <35) or (i > 42 and i < 49) and i> 52:
    #     w_new = 1.
    # else:
    #     w_new = 1.

    if i == 0:
        w_adj = 0
    else:
        w_adj = 2.5

        # print('loss_normal_adj2', loss_normal_adj)
    poses_inv_show = poses_inv[0].detach()
    # print(poses_inv_show.shape)
    poses_inv_show[:, :, 3] = poses_inv_show[:, :, 3] / torch.sqrt(
        poses_inv_show[:, 0, 3] ** 2 + poses_inv_show[:, 1, 3] ** 2 + poses_inv_show[:, 2, 3] ** 2)
    # print('poses', poses_inv[0])
    trans1 = poses_inv_show[0, :3, 3]
    trans2 = F_poses_inv[0][0, :3, 3]
    cos_theta = (trans1[0]*trans2[0] + trans1[1]*trans2[1] + trans1[2] * trans2[2])/ \
                (torch.sqrt(trans1[0]**2 + trans1[1]**2 + trans1[2]**2)*torch.sqrt(trans2[0]**2 + trans2[1]**2 + trans2[2]**2))
    angle = np.degrees(torch.arccos(cos_theta).detach().cpu().detach())
    print('cos_theta', angle)
    # if torch.abs(poses_inv_show[:, 2, 3]) < 0.95 or torch.abs(F_poses_inv[0][:, 2, 3]) < 0.95:
    if angle >10:
        w_f = 0
    else:
        w_f = 1

    if i >=1:
        w_p = 10
    else:
        w_p = 0
    # w_p = 10
        # loss = w1 * loss_1 + 0.5 * loss_3 + w2 * loss_2 + w_m*loss_match + w_f*loss_F + loss_1_ori + 0.5*loss_3_ori +5 *loss_normal_single \
    # loss = w2 * loss_2 + w1*loss_1_ori + 0.5*loss_3_ori + 2.5*(loss_match_ori) + 0.01*(loss_pc[0]+loss_pc_ori[0])+ 0.1*(loss_match + 0.5*loss_3 + 0.1*loss_1)+0.1*loss_normal_single
    loss = w2 * loss_2 + w1*loss_1_ori + 0.5*loss_3_ori + loss_match_ori + 1.0*w_f*( (loss_1 + 0.5 * loss_3)) \
           + loss_normal_single + loss_normal_per
           #   + 10*loss_pc[0] \

           # 0.5*( (loss_1 + 0.5 * loss_3)) \
           # + loss_pc_ori[0]
           # + 10 * (loss_pc_ori[0])
           # + loss_normal_single
           # + w_p*Q_plane \
           # + 0.1*loss_pc_ori[0]
           # + 5*loss_normal_single
           # + w_p*Q_plane + 0.1*loss_pc_ori[0]
           # + 0.5 * (loss_pc_ori[0] + loss_pc[0])\
           # + 0.5 * (loss_match + 0.5 * loss_3 + 0.1 * loss_1)\

           # + 0.5 * (loss_pc_ori[0] + loss_pc[0])\
           # + 0.5 * (loss_match + 0.5 * loss_3 + 0.1 * loss_1)

            # 5.5 * (loss_pc_ori[0] + loss_pc[0]) \

        #
           # + 0.5* (loss_match + 0.5 * loss_3 + 0.1 * loss_1) \
           # + 0.5 * (loss_normal_single) +


           # + 1 * (loss_match + 0.5 * loss_3 + 0.1 * loss_1) \

    # +w_aaaa * loss_pc_ori[0]
           # + 1 * (loss_match + 0.5 * loss_3 + 0.1 * loss_1)
           # +5* loss_normal_single
    # + (loss_1 + loss_match)

    # w2 * loss_2 + w1*loss_1_ori + 0.5*loss_3_ori + 0.1*loss_pc_ori[0] + loss_match_ori +0.01*loss_normal_single
           # + w_new * (w1*loss_1 + 0.5 * loss_3) + loss_match \

           # +1*loss_normal_single \
           #  + w_new * (loss_1 + 0.5 * loss_3) + 1.5 * loss_match_ori \
           # + 5*loss_F
    # +0.5 * loss_normal_single +
    # +1.0 * loss_normal_adj
             # + 0.1 * loss_pc_ori[0]\
           # +5*(loss_normal_single + 0.1*loss_normal_adj)
           # + 5*loss_depth_grad
    #+ 2.5*(loss_normal_single) + w_adj*loss_normal_adj

    # loss = w1 * loss_1 + loss_match + 0.5 * loss_3 + loss_2+ 5*(loss_normal_single+loss_normal_adj+ loss_pc[0] )
    # + loss_pc[0] +loss_depth_grad
    # + 10 * (0.1 * loss_normal_single + loss_normal_adj)
           # 5*(loss_normal_single+loss_normal_adj)w1 * loss_1
    # loss = w1 * loss_1 + 0.5 * loss_3 + loss_F + w2 * loss_2 + w_m * loss_match
           # + 5*loss_pc[1]
    # print('loss_pc1',loss_pc[1])
           # 5*(loss_normal_single+loss_normal_adj + loss_pc)
           # + 2.5*loss_normal
           # + 10*loss_normal
           # + loss_plane1
    # print('loss_depth_grad',loss_depth_grad)


    return loss_1, loss, Q_curr_ori, Q_curr_ori

def Sobel_function(img):
    src_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    Sobelx = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0)
    Sobely = cv2.Sobel(src_gray, cv2.CV_64F, 0, 1)
    Sobelx = cv2.convertScaleAbs(Sobelx)
    Sobely = cv2.convertScaleAbs(Sobely)
    Sobelxy = cv2.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)
    return Sobelxy

def img_norm2ori(img):
    img = ((img.detach().cpu().numpy()) * 0.225 + 0.45) * 255
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def online_adaption(poseNet, depthNet, optimizer, optimizer_SGD, train_loader, F_poses_with_inv, match_with_inv, plane_masks):
    poseNet.train()
    depthNet.train()
    # poseNet_copy = deepcopy(poseNet)
    # depthNet_copy = deepcopy(depthNet)

    global_pose = np.eye(4)
    # pose_ini = np.genfromtxt('/home/dokidoki/Datasets/7-Scenes/processing/head/poses/00001.txt').astype(
    #     np.float32).reshape(
    #     (4, 4))
    # print('pose_ini', pose_ini)
    # # stop
    # global_pose = pose_ini
    poses_write = []
    # poses_gt = []
    output_depths = []
    # poses_write.append(global_pose)
    # poses_write.append(global_pose[0:3, :].reshape(1, 12))
    global show_flag

    tpt_tgt_depth = torch.ones([1, 1, 256, 320])
    tpt_ref_depth1 = torch.ones_like(tpt_tgt_depth)
    tpt_ref_depth2 = torch.ones_like(tpt_tgt_depth)
    Q_last = None
    Q_last_f = None

    for i, (tgt_img, ref_imgs, tgt_img_next, ref_imgs_next, intrinsics, intrinsics_inv) in enumerate(train_loader):

        # if i > 25:
        # show_flag = True
        print("**********************************************")
        print('iter', i)
        # last_loss = 10
        # loss1 = 0.
        # # for meta_idx in range(40):
        stop_flag = False
        online_iter = 0
        # # fast_weights_pose = None
        # # fast_weights_depth = None
        # loss_total = 0
        # loss = 0
        # param_last = 0
        max_iteration = 30
        # if i < 12:
        #     max_iteration = 3
        # if i > 49:
        #     max_iteration = 3

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        # tgt_depth, _ = compute_depth(depthNet, tgt_img, ref_imgs, None, online_iter)

        # recover the normalized image
        ori_img = img_norm2ori(tgt_img[0])
        ori_img_ref1 = img_norm2ori(ref_imgs[0][0])
        ori_img_ref2 = img_norm2ori(ref_imgs[1][0])

        # ref_img1_array = ((ref_imgs[0][0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        # ref_img1_array = np.transpose(ref_img1_array, (1, 2, 0)).astype(np.uint8)

        # ref_img2_array = ((ref_imgs[1][0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        # ref_img2_array = np.transpose(ref_img2_array, (1, 2, 0)).astype(np.uint8)

        matches = [match_with_inv[1][i], match_with_inv[0][i + 1]]
        matches_inv = [match_with_inv[0][i], match_with_inv[1][i + 1]]

        # seg_map_tgt, src_smooth = seg(src, [ref_img1_array, ref_img2_array], matches, i)
        # seg_map_ref1, ref1_smooth = seg(src_ref1, [ref_img1_array, ref_img2_array], matches, i-1)
        # seg_map_ref2, ref2_smooth = seg(src_ref2, [ref_img1_array, ref_img2_array], matches, i+1)

        # the SegNet returns the segmentation map, the visualized seg map and the enhanced image
        seg_map_tgt, seg_show_tgt, src_en = seg(ori_img, i)
        seg_map_ref1, seg_show_ref1, ref1_en = seg(ori_img_ref1, i)
        seg_map_ref2, seg_show_ref2, ref2_en = seg(ori_img_ref2, i)

        Sobelxy_tgt = Sobel_function(src_en)
        Sobelxy_ref1 = Sobel_function(ref1_en)
        Sobelxy_ref2 = Sobel_function(ref2_en)

        # last_loss1 = 0
        pc_grad_for_seg = None
        # if i == 0:
        #     Q_last = None
        #     Q_last_f = None
        last_loss1 = 0

        # warm-up training
        while not stop_flag and online_iter < max_iteration:
        # # while online_iter < max_iteration:
        #     param_last = 0
            online_iter += 1
            # if i == 0 and online_iter == 1:
            # optim_params = [
            #     {'params': depthNet.parameters(), 'lr': 1e-4},
            #     {'params': poseNet.parameters(), 'lr': 1e-4}
            # ]
            # optimizer = torch.optim.Adam(optim_params,
            #                              betas=(0.9, 0.999),
            #                              weight_decay=0)

            # i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose, Q_last, Q_last_f, src_seg_show
            # loss_1_ori, loss, photo_mask_ori , Q_curr_f, Q_curr_ori, fused_grad_ori
            loss_pho, loss_curr, photo_masks, Q_curr_f, Q_curr_ori, pc_grad_for_seg = calculate_loss(i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose, Q_last, Q_last_f, seg_show_tgt)
            # _, loss_next = calculate_loss(i + 1, tgt_img_next, ref_imgs_next, plane_masks, F_poses_with_inv,
            #                               match_with_inv, intrinsics, poseNet, depthNet, None, None)
            # print('loss_pho', loss_pho)
            # print(Q_curr_f.shape)
            # print(pc_grad_for_seg.shape)
            # stop
            param_last = copy.deepcopy(optimizer.param_groups[0]['params'])
            # print('param_last',param_last[0][-1])

            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
            # print('middle',optimizer.param_groups[0]['params'][0][0])
            # if i > 7 and i < 14:
            #     loss_stop = torch.abs(loss_parts[0] - last_loss_1) < 0.01
            # else:
            # if i < 15:
            stop_flag = torch.abs(loss_pho - last_loss1) < 0.001
            # else:
            #     loss_stop = torch.abs(loss_parts[0] - last_loss_1) < 0.001
            #
            #     loss_stop = loss1_stop and (torch.abs(loss_2 - last_loss_2) < 0.01) and (
            #                 torch.abs(loss_3 - last_loss_3) < 0.01) \
            last_loss1 = loss_pho

            # online meta-learning
            if online_iter > 0:

                # print('inner loop')
                _, loss_next, _, _, _, _ = calculate_loss(i+1, tgt_img_next, ref_imgs_next, F_poses_with_inv,
                                              match_with_inv, intrinsics, poseNet, depthNet, global_pose, None, None, seg_show_ref2)
                # loss_next = 0
                optimizer.zero_grad()
                loss_next.backward()
                # print('before', optimizer.param_groups[0]['params'][0][-1])
                for param_idx in range(len(optimizer.param_groups[0]['params'])):
                    optimizer.param_groups[0]['params'][param_idx].data = param_last[param_idx].data

                # optimizer.param_groups[0]['params'] = param_last

                optimizer.step()
                # print('param_last', optimizer.param_groups[0]['params'][0][-1])

            del param_last

        tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)

        loss_stop = False
        online_iter = 0

        # the "Semantic Module"

        # coarse extraction based on the "photometric overfitting" phenomenon
        # tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs, None, online_iter)
        # mask_initial_draw = initial_masks[0][0].detach().cpu().numpy().astype(np.uint8)
        # result_point = cv2.bitwise_and(ori_img, ori_img, mask=mask_initial_draw)
        # if show_flag:
        # cv2.imshow('result_point', result_point)
        # # cv2.imshow('result_mask', result)
        # cv2.waitKey(0)

        # coarse extraction
        seg_map_tgt = torch.tensor(seg_map_tgt).to(device)
        seg_map_ref1 = torch.tensor(seg_map_ref1).to(device)
        seg_map_ref2 = torch.tensor(seg_map_ref2).to(device)

        # depth_grad = torch.zeros([1, 1, 256, 320], device=device)
        # depth_grad[0, 0, :255, :319] = grad_disp_x[:, :, :255, :319] + grad_disp_y[:, :, :255, :319]
        # depth_grad[0, 0, :254, :318] = pc_grad_for_seg

        # print(src.shape, initial_masks[0].unsqueeze(0).shape, seg_map_tgt.shape, tgt_depth[0].shape, Sobelxy_tgt.shape)
        # print(src_ref1.shape, initial_masks[1].unsqueeze(0).shape, seg_map_ref1.shape, ref_depths[0][0].shape, Sobelxy_ref1.shape)
        max_seg_mask_tgt = calculate_seg_new(ori_img, photo_masks[0].unsqueeze(0), seg_map_tgt, tgt_depth[0], Sobelxy_tgt,
                                             pc_grad_for_seg, flag='tgt')
        max_seg_mask_ref1 = calculate_seg_new(ori_img_ref1, photo_masks[1].unsqueeze(0), seg_map_ref1, ref_depths[0][0], Sobelxy_ref1, None, flag='ref')
        max_seg_mask_ref2 = calculate_seg_new(ori_img_ref2, photo_masks[3].unsqueeze(0), seg_map_ref2, ref_depths[1][0], Sobelxy_ref2, None, flag='ref')

        # write_mask = 1-(max_seg_mask_tgt>0).int()
        # print(write_mask.shape)

        # # # print(max_seg_mask_tgt.shape)
        # # # stop
        mask_draw_tgt = max_seg_mask_tgt[0, 0].detach().cpu().numpy().astype(np.uint8)
        # mask_draw_ref1 = max_seg_mask_ref1[0, 0].detach().cpu().numpy().astype(np.uint8)
        # mask_draw_ref2 = max_seg_mask_ref2[0, 0].detach().cpu().numpy().astype(np.uint8)

        # result_tgt = cv2.bitwise_and(src, src, mask=mask_draw_tgt)
        # if show_flag:
        # cv2.imshow('final_result', result_tgt)
        # cv2.waitKey(0)

        # result_ref1 = cv2.bitwise_and(src_ref1, src_ref1, mask=mask_draw_ref1)
        # result_ref2 = cv2.bitwise_and(src_ref2, src_ref2, mask=mask_draw_ref2)

        mask_warp_ref1 = F.grid_sample(max_seg_mask_ref1.float(), matches[0].clamp(-1, 1), padding_mode='border', align_corners=False)

        mask_warp_ref2 = F.grid_sample(max_seg_mask_ref2.float(), matches[1].clamp(-1, 1), padding_mode='border', align_corners=False)

        final_mask_tensor = max_seg_mask_tgt \
                            # * mask_warp_ref1.int() * mask_warp_ref2.int()

        mask_draw_ref1_warp = mask_warp_ref1[0, 0].int().detach().cpu().numpy().astype(np.uint8)
        mask_draw_ref2_warp = mask_warp_ref2[0, 0].int().detach().cpu().numpy().astype(np.uint8)

        result_tgt_warp1 = cv2.bitwise_and(ori_img, ori_img, mask=mask_draw_ref1_warp)
        result_tgt_warp2 = cv2.bitwise_and(ori_img, ori_img, mask=mask_draw_ref2_warp)

        # result_tgt_final = cv2.bitwise_and(src, src, mask=mask_draw_tgt*mask_draw_ref1_warp*mask_draw_ref2_warp)

        # print(total_mask.shape)
        # print(mask_draw_ref1_warp.shape)
        result_tgt_final = cv2.bitwise_and(ori_img, ori_img,
                                           mask=final_mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8))
        cv2.imshow('result_final', result_tgt_final)
        cv2.waitKey(0)

        # fine extraction
        connect_regions = largestConnectComponent(final_mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8), True)

        # grad_disp_x = torch.abs(tgt_depth[0][:, :, :, :-1] - tgt_depth[0][:, :, :, 1:])
        # grad_disp_y = torch.abs(tgt_depth[0][:, :, :-1, :] - tgt_depth[0][:, :, 1:, :])

        pc_grad_show = np.expand_dims(pc_grad_for_seg.cpu().detach().numpy(), axis=2).astype(np.uint8)
        # print(pc_grad_show.shape)
        # if show_flag:
        cv2.imshow('pc_grad_for_seg[2]',pc_grad_show)
        cv2.waitKey(0)


        # depth_grad = depth_grad/torch.max(depth_grad)
        #here
        total_mask = np.zeros([256,320])
        total_mask_left = np.zeros([256,320])
        total_mask_right = np.zeros([256,320])
        # print('len(connect_regions)', len(connect_regions))
        final_connect_seg_mask = []
        mask_instance = None
        # process_regions = []
        P_p = []
        P_d = []
        # pair = []
        line_return = None
        # for each connected region, detect the separating line in G_fuse
        for region_idx in range(len(connect_regions)):
            current_region = cv2.bitwise_and(ori_img, ori_img, mask=connect_regions[region_idx].astype(np.uint8))
            # if show_flag:
            cv2.imshow('current_region_ori', current_region)
            cv2.waitKey(0)
            # connect_region_tensor = torch.from_numpy(connect_regions[region_idx]).to(device).unsqueeze(0).unsqueeze(0)
            # connect_region_tensor_copy = connect_region_tensor.clone().int()
            # connect_region_tensor = connect_region_tensor*max_seg_mask_tgt
            #
            # seg_labels, seg_num = torch.unique(connect_region_tensor, return_counts=True)
            #
            # max_num, max_idx = torch.max(seg_num, dim=0)  # exact the largest part in the segment
            #
            # # seg_labs_max = seg_labs_select[torch.max(lab_count, dim=0)[1]]
            # seg_labs_max = seg_labels[max_idx]
            # print('max_num', max_num, seg_labs_max)
            # seg_num[max_idx] = 0
            #
            # max_num_second, max_idx_second = torch.max(seg_num, dim=0)
            #
            #
            # while max_num_second/max_num > 0.3:
            #     print('second_num', max_num_second, seg_labs_max)
            #     seg_labs_second_max = seg_labels[max_idx_second]
            #     seg_second_mask = torch.where(connect_region_tensor == seg_labs_second_max, torch.ones_like(connect_region_tensor),
            #                            torch.zeros_like(connect_region_tensor))
            #     current_region = cv2.bitwise_and(src, src, mask=seg_second_mask[0,0].cpu().detach().numpy().astype(np.uint8))
            #     if show_flag:
            #         cv2.imshow('current_region_second', current_region)
            #         cv2.waitKey(0)
            #     final_connect_seg_mask.append(seg_second_mask)
            #     connect_region_tensor_copy = ((seg_second_mask>0).int())
            #
            #     seg_num[max_idx_second] = 0
            #     max_num_second, max_idx_second = torch.max(seg_num, dim=0)
            #
            # current_region = cv2.bitwise_and(src, src, mask=connect_region_tensor[0,0].cpu().detach().numpy().astype(np.uint8))
            # if show_flag:
            #     cv2.imshow('current_region_largest', current_region)
            #     cv2.waitKey(0)
            # final_connect_seg_mask.append(connect_region_tensor>0)

            #line_detect
            mask_pc_grad = cv2.bitwise_and(pc_grad_show, pc_grad_show,
                                           mask=connect_regions[region_idx].astype(np.uint8)[:254, :318])
            line_return = line_detect(mask_pc_grad, current_region, True)

            if len(line_return) < 7 and len(line_return) > 0:
                x1, y1, x2, y2 = line_return[0][0]
                k = (y1-y2)/(x1-x2)
                b = y1 - k*x1
                color_mask_instance = np.zeros([256, 320, 3])
                color_mask_left = np.zeros([256, 320])
                color_mask_right = np.zeros([256, 320])
                mask_left = np.zeros([256, 320])
                mask_right = np.zeros([256, 320])
                pair = []
                for col in range(256):
                    for row in range(320):
                        if col > k*row+b:
                            color_mask_instance[col, row] = [225, 228, 255]
                            if connect_regions[region_idx][col, row] !=0:
                                color_mask_right[col, row] = connect_regions[region_idx][col, row]
                                mask_left[col, row] = 1
                        else:
                            color_mask_instance[col, row] = [255, 255, 225]
                            if connect_regions[region_idx][col, row] != 0:
                                color_mask_left[col, row] = connect_regions[region_idx][col, row]
                                mask_right[col, row] = 1
                left_mask = cv2.bitwise_and(ori_img, ori_img, mask=mask_left.astype(np.uint8))
                right_mask = cv2.bitwise_and(ori_img, ori_img, mask=mask_right.astype(np.uint8))
                cv2.imshow('left', left_mask)
                cv2.imshow('right', right_mask)
                cv2.waitKey(0)
                # total_mask_left += mask_left
                # total_mask_right += mask_right
                pair.append(mask_left)
                pair.append(mask_right)
                P_p.append(pair)
            else:
                P_d.append(connect_regions[region_idx]>0)

            # pc_grad_show = np.expand_dims(pc_grad_for_seg[2].cpu().detach().numpy(),axis=2).astype(np.uint8)
            # # print(pc_grad_show.shape)
        #     mask_pc_grad = cv2.bitwise_and(pc_grad_show, pc_grad_show,
        #                                    mask=connect_regions[region_idx].astype(np.uint8)[:254,:318])
        #     #
        #     total_mask+=connect_regions[region_idx]
        #     region_tensor = torch.from_numpy(connect_regions[region_idx])
        #     non0_index = torch.nonzero(region_tensor.reshape(1, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3,
        #                                                                                                       1)  # 1,3,n
        #     # print('x_coord', torch.min(non0_index[0, 0, :] % 320))
        #     if torch.min(non0_index[0, 0, :] % 320) > 100:
        #         total_mask_right += connect_regions[region_idx]
        #     else:
        #         total_mask_left += connect_regions[region_idx]
        #     # process_regions.append(connect_regions[region_idx])
        # pair = []
        # pair.append(total_mask_left)
        # pair.append(total_mask_right)
        # process_regions.append(pair)
        #
        # mask_instance = np.zeros([256, 320, 3])
        # for col in range(256):
        #     for row in range(320):
        #         if total_mask_left[col, row]!=0:
        #             mask_instance[col, row] = [255, 228, 225]
        #         elif total_mask_right[col, row]!=0:
        #             mask_instance[col, row] = [225, 255, 255]




            # # cv2.imshow('pc_grad_for_seg[2]', mask_pc_grad)
            #
            # line_return_tmp = line_detect(mask_pc_grad, current_region, True)
            # if len(line_return_tmp)!=0 and line_return is None:
            #     line_return = line_return_tmp

            # print(len(line_return))
            # if len(line_return) < 7 and len(line_return)>0:
            #     x1, y1, x2, y2 = line_return[0][0]
            #     k = (y1-y2)/(x1-x2)
            #     b = y1 - k*x1
            #     mask_instance = np.zeros([256, 320, 3])
            #     mask_left = np.zeros([256, 320])
            #     mask_right = np.zeros([256, 320])
            #     pair = []
            #     for col in range(256):
            #         for row in range(320):
            #             if col > k*row+b:
            #                 mask_instance[col, row] = [225, 228, 255]
            #                 if connect_regions[region_idx][col, row] !=0:
            #                     mask_right[col, row] = connect_regions[region_idx][col, row]
            #             else:
            #                 mask_instance[col, row] = [255, 255, 225]
            #                 if connect_regions[region_idx][col, row] != 0:
            #                     mask_left[col, row] = connect_regions[region_idx][col, row]
            #     # total_mask_left += mask_left
            #     # total_mask_right += mask_right
            #     pair.append(mask_left)
            #     pair.append(mask_right)
            #     process_regions.append(pair)
            #
            # else:
            #     # total_mask_left += connect_regions[region_idx]
            #     process_regions.append(connect_regions[region_idx])
            #     # cv2.imshow('instance_mask', mask_instance)
            #     # cv2.waitKey(0)
                # here

            # print(pc_grad_for_seg[2].shape) #254,318

        # pair.append(total_mask_left)
        # pair.append(total_mask_right)
        # process_regions.append(pair)
        # print('len(line_return)',len(line_return))

        # if line_return is not None and len(line_return) >0 and len(line_return) > 0:
        #     line_len = len(line_return)
        #     mid = int(line_len/2)
        #     x1, y1, x2, y2 = line_return[mid][0]
        #     k = (y1 - y2) / (x1 - x2)
        #     b = y1 - k * x1
        #     mask_instance = np.zeros([256, 320, 3])
        #     mask_left = np.zeros([256, 320])
        #     mask_right = np.zeros([256, 320])
        #     pair = []
        #     for col in range(156):
        #         for row in range(320):
        #             if col > k * row + b:
        #                 mask_instance[col, row] = [225, 228, 255]
        #                 if total_mask[col, row] != 0:
        #                     mask_right[col, row] = total_mask[col, row]
        #             else:
        #                 mask_instance[col, row] = [255, 255, 225]
        #                 if total_mask[col, row] != 0:
        #                     mask_left[col, row] = total_mask[col, row]
        #     # total_mask_left += mask_left
        #     # total_mask_right += mask_right
        #     pair.append(mask_left)
        #     pair.append(mask_right)
        #     process_regions.append(pair)
        # else:
        #     bottom_mask = np.zeros([256, 320])
        #     bottom_mask[:150, : ]=1
        #     process_regions.append(total_mask*bottom_mask)
        # write_mask = write_mask[0,0,:, :] + torch.from_numpy(total_mask>0).to(device)

        #here

        #
        #     # line_detect()
        #
        #     # if show_flag:
        #     # cv2.imshow('current_region', current_region)
        #     # cv2.waitKey(0)

        # class_labels, indx = torch.unique(mask_tensor, return_counts=True)
        # print('class_num', len(class_labels))
        # print('class_num', class_labels)
        # cv2.imshow('pc_grad_for_seg', pc_grad_for_seg)

        # for class_idx in range(1,len(class_labels)):
        #
        #     class_label = class_labels[class_idx]
        #     print('class_label', class_label)
        #     seg_mask = torch.where(mask_tensor == class_label, torch.ones_like(mask_tensor),
        #                            torch.zeros_like(mask_tensor))
        #     # seg_mask[:, :, 254:, 318:] = 0
        #     # seg_mask = seg_mask * mask_tensor
        #     # torch.nonzero(normal_map.reshape([1, 3, -1]))
        #     print(torch.sum(seg_mask))
        #     result_tgt_final_batch = cv2.bitwise_and(src, src, mask=seg_mask[0, 0].int().detach().cpu().numpy().astype(np.uint8))
        #     current_mask_idx = torch.nonzero(seg_mask.reshape(-1))
        #
        #     # pc_grad_for_seg_show = torch.from_numpy(pc_grad_for_seg).to(device)
        #     # pc_grad_for_seg_norm = pc_grad_for_seg_show/torch.max(pc_grad_for_seg_show)
        #     # pc_grad_comp = torch.zeros([1,1,256,320], device=device)
        #     # pc_grad_comp[0,0,:254,:318] = pc_grad_for_seg_norm[:,:,0]
        #     #
        #     # pc_grad_select = torch.gather(pc_grad_comp.float().reshape(-1).to(device), index=current_mask_idx[:, 0], dim=0)
        #
        #     if show_flag and torch.sum(seg_mask)>5000:
        #         # print('mean_pc_grad', torch.mean(pc_grad_select))
        #         # print('var_pc_grad', torch.var(pc_grad_select))
        #         # cv2.imshow('result_point', result_point)
        #         # cv2.imshow('result_tgt', result_tgt)
        #         # cv2.imshow('result_ref1', result_tgt_warp1)
        #         # cv2.imshow('result_ref2', result_tgt_warp2)
        #         cv2.imshow('src', src)
        #         cv2.imshow('result_final_batch', result_tgt_final_batch)
        #         cv2.waitKey(0)
        #
        #     else:
        #         print('jump')

        # cv2.waitKey(0)

        # result_tgt_final = cv2.bitwise_and(src, src, mask=max_seg_mask_tgt[0, 0].detach().cpu().numpy().astype(np.uint8))
        # print(total_mask.shape)
        # stop
        # result_tgt_final = cv2.bitwise_and(src, src, mask=total_mask.astype(np.uint8))
        # if mask_instance is not None:
        #     result_tgt_color = cv2.bitwise_and(mask_instance.astype(np.uint8), mask_instance.astype(np.uint8), mask=total_mask.astype(np.uint8))
        #     result_tgt_final = cv2.bitwise_and(ori_img, ori_img, mask=total_mask.astype(np.uint8))
            # result_tgt_final = cv2.bitwise_and(src, src, mask=mask_draw_tgt)
            # imsave(
            #     '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/seg_mask/mask_' + str(i) + '.png',
            #     result_tgt_color)



            # if show_flag:
            # # cv2.imshow('result_point', result_point)
            # # cv2.imshow('result_tgt', result_tgt)
            # # cv2.imshow('result_ref1', result_tgt_warp1)
            # # cv2.imshow('result_ref2', result_tgt_warp2)
            #     cv2.imshow('result_final', result_tgt_color)
            #     imsave(
            #         '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/seg_mask/mask_' + str(i) + '.png',
            #         result_tgt_final)
            #     cv2.waitKey(0)
        # last_seg_mask = torch.from_numpy(total_mask.astype(np.uint8)*mask_draw_ref1_warp).to(device)
        #
        # # and (i < 12 or i > 38)
        # # idx_flag = i < 16 or (i > 39 and i<44) or i > 52
        # idx_flag = i < 13 or i >52
        # wall_flag = None
        # if i > 39 and i<44:
        #     wall_flag = 'para'
        # if i >52:
        #     wall_flag = 'verti'
        #
        # if (i == 1 or i == 52) and online_iter == 0:
        #     print('in')
        #     last_norm = last_norm_temp

        # the training with the semantic loss
        second_max_iter = 5
        last_loss_1= 0
        while not loss_stop and online_iter < second_max_iter:
            # param_last = 0
            online_iter += 1

                # pose_mat = pose_vec2mat(poses[0]).squeeze(0)
                #
                # pose_mat_h = torch.cat([pose_mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
                # global_pose_plane = torch.from_numpy(global_pose).to(device).float() @ pose_mat_


            loss_pho, loss_curr, Q_curr_f, Q_curr_ori = calculate_loss_norm(i, tgt_img, ref_imgs, F_poses_with_inv,
                                                                 match_with_inv, intrinsics, poseNet, depthNet, global_pose, [P_d, P_p],Q_last, Q_last_f, seg_show_tgt)

            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
            loss_stop = torch.abs(loss_pho - last_loss_1) < 0.01
            #
            #     loss_stop = loss1_stop and (torch.abs(loss_2 - last_loss_2) < 0.01) and (
            #                 torch.abs(loss_3 - last_loss_3) < 0.01) \
            last_loss_1 = loss_pho
            # last_norm_temp = last_norm_return
        Q_last = Q_curr_ori
        Q_last_f = Q_curr_f

        #
        #
        #
        #
        # last_plane = max_seg_mask_tgt
        # tgt_depth, _ = compute_depth(depthNet, tgt_img, ref_imgs, None)
        # src_norm = calculate_all_norm(tgt_depth[0][0, 0])
        # # cv2.imwrite("./normal_ori/" + str(i) + ".png", src_norm.astype(np.uint8))
        #
        #
        #
        # # src_ref1 = ((ref_imgs[0][0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        # # src_ref1 = np.transpose(src_ref1, (1, 2, 0)).astype(np.uint8)
        #
        # # src_ref2 = ((ref_imgs[1][0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        # # src_ref2 = np.transpose(src_ref2, (1, 2, 0)).astype(np.uint8)
        #
        # seg_map_tgt, _ = seg(src, src_norm_comp, i, input_type='image')
        # # seg_map_ref1, _ = seg(src_ref1, src_norm_comp_ref1, i, input_type='image')
        # # seg_map_ref2, _ = seg(src_ref2, src_norm_comp_ref2, i, input_type='image')
        #
        # seg_map_tgt = torch.tensor(seg_map_tgt).to(device)
        #
        # # enhance the color image for better line detection
        # src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        # Sobelx = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0)
        # Sobely = cv2.Sobel(src_gray, cv2.CV_64F, 0, 1)
        # Sobelx = cv2.convertScaleAbs(Sobelx)
        # Sobely = cv2.convertScaleAbs(Sobely)
        # Sobelxy = cv2.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)
        #
        # img = 255 * np.power(Sobelxy / 255, 0.5)
        # img = np.around(img)
        # img[img > 255] = 255
        # img[img < 50] = 0
        # sigma_img = img.astype(np.uint8)
        #
        # # if show_flag:
        # #     cv2.imshow("original", src_gray)
        # #     cv2.imshow("original_sobel", Sobelxy)
        # #     cv2.imshow("xy", sigma_img)
        # #     cv2.waitKey(0)
        # #     # cv2.destroyAllWindows()
        # src_norm_comp = calculate_all_norm(tgt_depth[0][0, 0])
        #
        # max_seg_mask_tgt, bbox_tgt, total_seg_mask = calculate_total_seg(mask_depth, plane_mask_tgt, seg_map_tgt, Sobelxy)
        #
        # ###find line
        # if i > 21:
        #     print('in detect line!!!!!!!!!!!!!!!!!!')
        #     # detect wall line
        #     y_min, x_min, y_max, x_max = bbox_tgt
        #
        #     # find the box in the ref frame through the flow
        #     x_min_ref1, y_min_ref1 = matches[0][0, int(y_min), int(x_min)]
        #     x_max_ref1, y_max_ref1 = matches[0][0, int(y_max-1), int(x_max-1)]
        #
        #     bbox_ref1 = [(y_min_ref1+1) * 255 / 2, (x_min_ref1+1) * 319 / 2, (y_max_ref1+1) * 255 / 2, (x_max_ref1+1) * 319 / 2]
        #
        #     x_min_ref2, y_min_ref2 = matches[1][0, int(y_min), int(x_min)]
        #     x_max_ref2, y_max_ref2 = matches[1][0, int(y_max-1), int(x_max-1)]
        #     bbox_ref2 = [(y_min_ref2+1)* 255 / 2, (x_min_ref2+1)* 319 / 2, (y_max_ref2+1)* 255 / 2, (x_max_ref2+1)* 319 / 2]
        #
        #     max_seg_mask_ref1 = F.grid_sample(max_seg_mask_tgt.float(),
        #                                       matches_inv[0], mode='nearest', padding_mode='zeros',
        #                                       align_corners=False)
        #     line_mask_ref1 = max_seg_mask_ref1[0, 0].int().detach().cpu().numpy().astype(np.uint8)
        #
        #     max_seg_mask_ref2 = F.grid_sample(max_seg_mask_tgt.float(),
        #                                         matches_inv[1], mode='nearest', padding_mode='zeros',
        #                                         align_corners=False)
        #     line_mask_ref2 = max_seg_mask_ref2[0, 0].int().detach().cpu().numpy().astype(np.uint8)
        #     line_mask_tgt = max_seg_mask_tgt[0, 0].detach().cpu().numpy().astype(np.uint8)
        #
        #     if np.sum(line_mask_tgt) > 3000:
        #         normal_tgt = cv2.bitwise_and(src_norm, src_norm, mask=line_mask_tgt)
        #         sigma_normal_tgt = cv2.bitwise_and(sigma_img, sigma_img, mask=line_mask_tgt)
        #         normal_ref1 = cv2.bitwise_and(src_norm_ref1, src_norm_ref1, mask=line_mask_ref1)
        #         normal_ref2 = cv2.bitwise_and(src_norm_ref2, src_norm_ref2, mask=line_mask_ref2)
        #
        #         # 显示seg mask图片
        #         if show_flag:
        #             cv2.imshow('src_norm_tgt',normal_tgt)
        #             cv2.imshow('src_norm_ref1',normal_ref1)
        #             cv2.imshow('src_norm_ref2',normal_ref2)
        #             cv2.waitKey(0)
        #
        #         src = ((tgt_img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        #         src = np.transpose(src, (1, 2, 0)).astype(np.uint8)
        #
        #         _, normal_seg_tgt = seg_only(normal_tgt, input_type='image')
        #         _, normal_seg_ref1 = seg_only(normal_ref1, input_type='image')
        #         _, normal_seg_ref2 = seg_only(normal_ref2, input_type='image')
        #         # 调用函数
        #         src_show_sigma = cv2.bitwise_and(src, src, mask=line_mask_tgt)
        #         _, _, lines_tgt, line_draw_tgt = line_detect(src_show_sigma, normal_seg_tgt, bbox_tgt)
        #         _, _, lines_sigma, line_draw_sigma = line_detect(src_show_sigma, sigma_normal_tgt, bbox_tgt)
        #         _, _, lines_ref1, line_draw_ref1 = line_detect(src_show_sigma, normal_seg_ref1, bbox_ref1)
        #         _, _, lines_ref2, line_draw_ref2 = line_detect(src_show_sigma, normal_seg_ref2, bbox_ref2)
        #
        #         src_copy = src.copy()
        #         candidate_lines = []
        #
        #         for sigma_idx in range(len(lines_sigma)):
        #             grad_line = calculate_grad_for_line(Sobelxy, lines_sigma[sigma_idx])
        #             x1, y1, x2, y2 = lines_sigma[sigma_idx][0]
        #             if grad_line < 15:
        #                 candidate_lines.append(lines_sigma[sigma_idx][0])
        #                 if show_flag:
        #                     cv2.line(src_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #                     src_copy = cv2.cvtColor(src_copy, cv2.COLOR_BGR2RGB)
        #                     cv2.imshow('src_line',src_copy)
        #                     cv2.waitKey(0)
        #
        #         if len(candidate_lines) >= 1:
        #             # print('in2')
        #             match_tgt,_ = compare_with_tgt(candidate_lines, lines_tgt, matches=None)
        #             match_ref1,_ = compare_with_tgt(candidate_lines, lines_ref1, matches=matches_inv[0])
        #             match_ref2,_ = compare_with_tgt(candidate_lines, lines_ref2, matches=matches_inv[1])
        #             if last_line is not None:
        #                 # print('check_last')
        #                 match_last_line,_ = compare_with_tgt(candidate_lines, [[last_line]], matches=matches_inv[0])
        #             else:
        #                 match_last_line = np.zeros_like(match_tgt)
        #             print('match_4_times', match_tgt)
        #             print('match_4_times', match_ref1)
        #             print('match_4_times', match_ref2)
        #             print('match_4_times', match_last_line)
        #             print('lastweight', last_match_weight)
        #             print('match_4_times', last_match_weight*np.array(match_last_line))
        #             matches_max = -1
        #             print('cadidate',candidate_lines)
        #             for idx_candi in range(len(candidate_lines)):
        #                 x1, y1, x2, y2 = candidate_lines[idx_candi]
        #                 if last_line is None and (x1 < 200 or x1 > 290):
        #                     continue
        #                 if last_match_weight != 0:
        #                     match_last_line[idx_candi] = 2*last_match_weight * np.array(match_last_line[idx_candi])
        #                 match_curr = match_tgt[idx_candi] + match_ref1[idx_candi] + match_ref2[idx_candi] + match_last_line[idx_candi]
        #                 if match_curr == matches_max:
        #                     if match_last_line[idx_candi]>0:
        #                         matches_max = match_curr
        #                         final_line = candidate_lines[idx_candi]
        #                 elif match_curr > matches_max:
        #                     matches_max  = match_curr
        #                     final_line = candidate_lines[idx_candi]
        #
        #         if len(candidate_lines) == 1:
        #             final_line = candidate_lines[0]
        #             print('aaa', last_line, final_line[0])
        #             if last_line is None and final_line[0] < 270:
        #                 print('detect not right')
        #                 final_line = None
        #         proj_last_line = None
        #         match_consis = None
        #         print('last_line', last_line, [final_line])
        #         if last_line is not None:
        #             match_consis, proj_last_line = compare_with_tgt([final_line], [[last_line]], matches=matches_inv[0])
        #
        #         if final_line is not None:
        #                 # print(match_consis[0])
        #             if match_consis is not None and match_consis[0] > 0:
        #                 last_match_weight += 1
        #             print('proj_last_line', proj_last_line, final_line)
        #             thresh = 10
        #             if i >= 48:
        #                 thresh = 15
        #             if proj_last_line is not None and (proj_last_line[0] - final_line[0] > thresh or proj_last_line[2] - final_line[2] > 10):
        #                 print('detect_error, keep last line')
        #                 final_line = proj_last_line
        #
        #             x1_draw, y1_draw, x2_draw, y2_draw = final_line
        #             src_select = src.copy()
        #             if show_flag:
        #                 cv2.line(src_select, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 2)
        #                 src_select = cv2.cvtColor(src_select, cv2.COLOR_BGR2RGB)
        #                 cv2.imshow('src_line_draw', src_select)
        #                 cv2.waitKey(0)
        #
        #         else:
        #             final_line = proj_last_line
        #         if final_line is not None:
        #             x1_draw, y1_draw, x2_draw, y2_draw = final_line
        #             src_select = src.copy()
        #             # cv2.line(src_select, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 0, 255), 2)
        #             # cv2.imshow('src_line_draw', src_select)
        #             # cv2.waitKey(0)
        #
        #         last_line = final_line
        #         if final_line is not None and final_line[0] < 50:
        #             last_line = None
        #
        # #optimize again
        # print('optimize again~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # # print('final_line', final_line)
        # loss_stop = False
        # online_iter = 0
        # k_wall_line = 0
        # b_wall_line = 0
        #
        # seg_line_mask_l = torch.ones([1, 1, 256, 320]).to(device)
        # seg_line_mask_r = torch.ones([1, 1, 256, 320]).to(device)
        # grid = meshgrid(256, 320).float().to(device).unsqueeze(0).repeat(1, 1, 1, 1)  # [b,2,h,w]
        # if i < 12 or (i >20 and i < 45) or i > 52:
        # # if i < 11 :
        #     final_line = None
        # if final_line is not None:
        #     print('exits line')
        #     seg_line_mask_l = torch.zeros([1, 1, 256, 320]).to(device)
        #     seg_line_mask_r = torch.zeros([1, 1, 256, 320]).to(device)
        #     x1, y1, x2, y2 = final_line
        #     if x2 - x1 == 0:
        #         print('chuizhi',x1)
        #         seg_line_mask_l[:, :, :, :x1] = 1
        #         seg_line_mask_r[:, :, :, x1:] = 1
        #         k_wall_line = 90
        #         b_wall_line = x1
        #     else:
        #         k = (y1 - y2) / (x1 - x2)
        #         b = y1 - k * x1
        #         k_wall_line = k
        #         b_wall_line = b
        #         x = grid[:, 0, :, :].unsqueeze(0)
        #         y = grid[:, 1, :, :].unsqueeze(1)
        #         # print('grid', grid.shape)
        #         # print('x', torch.max(x))
        #         if k > 0:
        #             plane_mask_idx_l = torch.where(y > (k * x + b), torch.ones_like(grid), torch.zeros_like(grid))
        #             plane_mask_idx_r = torch.where(y < (k * x + b), torch.ones_like(grid), torch.zeros_like(grid))
        #         elif k < 0:
        #             plane_mask_idx_l = torch.where(y < (k * x + b), torch.ones_like(grid), torch.zeros_like(grid))
        #             plane_mask_idx_r = torch.where(y > (k * x + b), torch.ones_like(grid), torch.zeros_like(grid))
        #         # print(plane_mask_idx.shape)
        #         seg_line_mask_l = plane_mask_idx_l[:,:1,:,:]
        #         seg_line_mask_r = plane_mask_idx_r[:,:1,:,:]
        #
        # src_l = ((tgt_img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        # src_l = np.transpose(src_l, (1, 2, 0)).astype(np.uint8)
        # src_r = src_l.copy()
        #
        # # if i == 0  or i >8:
        # #     seg_line_mask_l = torch.ones_like(finetune_seg_mask).to(device)
        # #     seg_line_mask_r = torch.ones_like(finetune_seg_mask).to(device)
        #
        # plane_flag_l = torch.sum(total_seg_mask * seg_line_mask_l) > 5000
        # plane_flag_r = (torch.sum(total_seg_mask * seg_line_mask_r) > 5000) and ((i > 11 and i < 44 )or i > 45)
        # print('l_flag',torch.sum(total_seg_mask * seg_line_mask_l))
        # print('r_flag',torch.sum(total_seg_mask * seg_line_mask_r))
        # mask_plane_with_line_l = (total_seg_mask[0, 0] * seg_line_mask_l[0, 0]).int().detach().cpu().numpy().astype(
        #     np.uint8)
        # mask_plane_with_line_r = (
        #             total_seg_mask[0, 0] * seg_line_mask_r[0, 0]).int().detach().cpu().numpy().astype(
        #     np.uint8)
        # src_show_plane_seg = cv2.bitwise_and(src, src, mask=total_seg_mask[0,0].int().detach().cpu().numpy().astype(
        #     np.uint8))
        # src_show_plane_line_l = cv2.bitwise_and(src, src, mask=mask_plane_with_line_l)
        # src_show_plane_line_r = cv2.bitwise_and(src, src, mask=mask_plane_with_line_r)
        # # if online_iter == 0 and show_flag:
        # #     print('show')
        # #     src_show_plane_seg = cv2.cvtColor(src_show_plane_seg, cv2.COLOR_BGR2RGB)
        # #     cv2.imshow('src_show_plane_seg', src_show_plane_seg)
        # #     cv2.imshow('src_show_plane_line_l', src_show_plane_line_l)
        # #     cv2.imshow('src_show_plane_line_r', src_show_plane_line_r)
        # #     cv2.waitKey(0)
        #
        #
        # second online iteration
        # while ((not loss_stop and online_iter < max_iter) or online_iter < 11):
        #
        #     online_iter += 1
        #
        #     poses, poses_inv = compute_pose_with_inv(poseNet, tgt_img, ref_imgs)
        #
        #     tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)
        #
        #     mask_depth = (tgt_depth[0] > (torch.mean(tgt_depth[0]) + 3. * torch.var(tgt_depth[0]))).int()
        #
        #     pose_mat = pose_vec2mat(poses[0]).squeeze(0)
        #
        #     pose_mat_h = torch.cat([pose_mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
        #     global_pose_plane = torch.from_numpy(global_pose).to(device).float() @ pose_mat_h
        #
        #     pose_mat_next = pose_vec2mat(poses_inv[1]).squeeze(0)
        #     pose_mat_last = pose_vec2mat(poses_inv[0]).squeeze(0)
        #
        #     pose_mat_h_next = torch.cat([pose_mat_next, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
        #     pose_mat_h_last = torch.cat([pose_mat_last, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
        #     global_pose_plane_next = global_pose_plane @ pose_mat_h_next
        #     global_pose_plane_last = global_pose_plane @ pose_mat_h_last
        #     # print('tgt_depth',tgt_depth)
        #
        #     if i == 0 and online_iter == 1:
        #         loss_plane1, norm_mask1, normal_median, coeff = fit_plane(True, tgt_depth[0].detach(),
        #                                                                   global_pose_plane.unsqueeze(0).detach(),
        #                                                                   intrinsics[0],
        #                                                                   total_seg_mask,
        #                                                                   # torch.tensor(label_mask).unsqueeze(0).unsqueeze(0).to(device),
        #                                                                   online_iter,
        #                                                                   a_ini.detach(), b_ini.detach(),
        #                                                                   d_ini.detach(), i, flag='tgt')
        #
        #         a_ini = torch.tensor([-coeff[0] / coeff[2]], requires_grad=True, device=device)
        #         b_ini = torch.tensor([-coeff[1] / coeff[2]], requires_grad=True, device=device)
        #         d_ini = torch.tensor([-coeff[3] / coeff[2]], requires_grad=True, device=device)
        #         print('find new plane___________________')
        #         for no_sth in range(5000):
        #             line = []
        #             # print('no thing', no_sth)
        #             # if no_sth != 0:
        #             # tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)
        #             # poses, poses_inv = compute_pose_with_inv(poseNet, tgt_img, ref_imgs)
        #             #
        #             # pose_mat = pose_vec2mat(poses[0]).squeeze(0)
        #             #
        #             # pose_mat_h = torch.cat([pose_mat, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0)
        #             # global_pose_plane = torch.from_numpy(global_pose).to(device).float() @ pose_mat_h
        #
        #             loss_plane1 = fit_plane_initial(True, tgt_depth[0].detach(), global_pose_plane.unsqueeze(0).detach(),
        #                                             intrinsics[0], total_seg_mask.detach(), i, no_sth,
        #                                             a_ini, b_ini, d_ini, line, a_ini, b_ini, d_ini)
        #             # loss_plane2 = fit_plane(True, ref_depths[1][0].detach(), poses[0].detach(), intrinsics[0], plane_mask_next.detach(), online_iter,
        #             #                         a_ini_next, b_ini_next, d_ini_next)
        #             # loss_norm = torch.abs(a_ini - a_ini_next) + torch.abs(b_ini - b_ini_next)
        #
        #             # loss_norm = a_ini*(-0.1320) + b_ini *0.0147 + 1
        #             loss_ini = loss_plane1 \
        #                 # + loss_norm
        #             # print('loss_plane1', loss_plane1)
        #
        #             # optimizer.zero_grad()
        #             loss_ini.backward()
        #
        #             # print('loss_plane_ini', loss_plane1)
        #             a_ini.data.add_(- 0.001 * a_ini.grad.data)
        #             b_ini.data.add_(- 0.001 * b_ini.grad.data)
        #             d_ini.data.add_(- 0.001 * d_ini.grad.data)
        #
        #             a_ini.grad.data.zero_()
        #             b_ini.grad.data.zero_()
        #             d_ini.grad.data.zero_()
        #             # optimizer.step()
        #
        #     if ( i == 12 or i == 45 or i == 38) and online_iter == 1 :
        #         # if i != 15 :
        #         #     a_ini = a_ini_r
        #         #     b_ini = b_ini_r
        #         #     d_ini = d_ini_r
        #
        #         # print('coeff') loss_plane2, norm_mask2, normal_median, coeff2
        #         coeff2 = fit_plane(True, tgt_depth[0],
        #                                                                    global_pose_plane.unsqueeze(0),
        #                                                                    intrinsics[0],
        #                                                                    total_seg_mask * seg_line_mask_r,
        #                                                                    # torch.tensor(label_mask).unsqueeze(0).unsqueeze(0).to(device),
        #                                                                    online_iter,
        #                                                                    a_ini_r.detach(), b_ini_r.detach(),
        #                                                                    d_ini_r.detach(), i, flag='plane_ini')
        #
        #         # print('coeff2',coeff2)
        #         # stop
        #
        #         one_tensor = torch.tensor([-1], device=device)
        #
        #         # if i == 38:
        #         #     a_ini_r = torch.tensor([-0.1], requires_grad=True, device=device)
        #         #     b_ini_r = torch.tensor([-0.1], requires_grad=True, device=device)
        #         #     d_ini_r = torch.tensor([-0.1], requires_grad=True, device=device)
        #
        #         # else:
        #         a_ini_r = torch.tensor([-coeff2[0] / coeff2[2]], requires_grad=True, device=device)
        #         b_ini_r = torch.tensor([-coeff2[1] / coeff2[2]], requires_grad=True, device=device)
        #         d_ini_r = torch.tensor([-coeff2[3] / coeff2[2]], requires_grad=True, device=device)
        #
        #         # a_ini_r = torch.tensor([-coeff2[0] / coeff2[2]], requires_grad=True, device=device)
        #         # b_ini_r = torch.tensor([-coeff2[1] / coeff2[2]], requires_grad=True, device=device)
        #         # d_ini_r = torch.tensor([-coeff2[3] / coeff2[2]], requires_grad=True, device=device)
        #         print('find new plane right___________________')
        #         # print('a_ini_r_ini',a_ini_r,b_ini_r,d_ini_r)
        #
        #         loss_ini2 = 1
        #         loss_norm = torch.tensor([1])
        #         loss_vertical = torch.tensor([1])
        #         loss_dis = torch.tensor([1])
        #         no_sth = 0
        #
        #         # while(torch.abs(loss_norm)>0.01 and no_sth < 10000):
        #         # while((torch.abs(loss_norm)>0.005 or loss_vertical > 0.01)and no_sth<5000):
        #         while((torch.abs(loss_norm)>0.01 or loss_vertical > 0.1)and no_sth<10000):
        #         # while(no_sth<5000):
        #         # while (no_sth<10):
        #             no_sth+=1
        #             union_line = []
        #
        #             norm_l = torch.cat([a_ini, b_ini, one_tensor]).detach()
        #             norm_r = torch.cat([a_ini_r, b_ini_r, one_tensor])
        #             norm_l_guiyi = norm_l/torch.sqrt(torch.sum(norm_l**2))
        #             norm_r_guiyi = norm_r/torch.sqrt(torch.sum(norm_r**2))
        #             # print('norm_lo', norm_l, norm_r)
        #             cross_norm = torch.cross(norm_l, norm_r)
        #             cross_norm_guiyi = cross_norm/torch.sqrt(torch.sum(cross_norm**2))
        #
        #             loss_plane2 = fit_plane_initial(True, tgt_depth[0].detach(),
        #                                             global_pose_plane.unsqueeze(0).detach(),
        #                                             intrinsics[0], total_seg_mask * seg_line_mask_r, i, no_sth,
        #                                             a_ini_r, b_ini_r, d_ini_r, union_line, a_ini, b_ini, d_ini)
        #
        #             loss_norm = a_ini_r * a_ini.detach() + b_ini.detach() * b_ini_r + 1
        #             loss_vertical = torch.abs(cross_norm[0])
        #             loss_dis =0
        #             # if final_line is not None:
        #             #     loss_dis = torch.abs((b_wall_line-b_3d_proj)/b_wall_line) + torch.abs((k_wall_line-k_3d_proj)/k_wall_line)
        #
        #             w_norm = 2.5
        #             lr = 0.01
        #             w_v = 0.4
        #             # if i == 45 :
        #             #     # w_norm = 5
        #             #     w_v = 0.25
        #             if i == 38 :
        #                 w_norm = 7.5
        #                 w_v = 0.3
        #             if i ==12:
        #                 # w_norm = 5.0
        #                 w_v = 0.4
        #
        #             loss_ini2 = loss_plane2 + w_norm*torch.abs(loss_norm)+ w_v*loss_vertical
        #             loss_ini2.backward()
        #
        #             # print(loss_plane2)
        #             if no_sth % 100 == 0 or no_sth ==1:
        #                 print('nosth', no_sth)
        #                 print('%%%%%%%%%%%%%%%%%%%%%%%%%%', a_ini_r, b_ini_r, d_ini_r)
        #                 print('loss_plane', loss_plane2)
        #                 # print('union', union_line[0])
        #                 # print('b_wall_line', k_wall_line, k_3d_proj, b_wall_line, b_3d_proj)
        #                 print('loss_norm', loss_norm)
        #                 print('loss_v', loss_vertical)
        #                 print('loss_dis', loss_dis)
        #                 # print('point3d', pixel_3d0, pixel_3d1)
        #
        #
        #             # print('loss_plane_ini', loss_plane1)
        #             a_ini_r.data.add_(- lr * a_ini_r.grad.data)
        #             b_ini_r.data.add_(- lr * b_ini_r.grad.data)
        #             d_ini_r.data.add_(- lr * d_ini_r.grad.data)
        #             # print('grad', a_ini_r.grad.data)
        #             # print('grad', b_ini_r.grad.data)
        #             # print('grad', d_ini_r.grad.data)
        #             # print('abd_r_inter', a_ini_r, b_ini_r, d_ini_r)
        #             a_ini_r.grad.data.zero_()
        #             b_ini_r.grad.data.zero_()
        #             d_ini_r.grad.data.zero_()
        #             # optimizer.step()
        #
        #
        #     loss_1, loss_3, loss_trian, loss_pc, write_mask, loss_match = compute_photo_and_geometry_loss_trian(
        #         tgt_img,
        #         ref_imgs,
        #         intrinsics,
        #         tgt_depth,
        #         ref_depths,
        #         plane_mask_tgt.int(),
        #         poses,
        #         poses_inv,
        #         1, 1, 1, 0,
        #         'zeros',
        #         online_iter,
        #         matches,
        #         matches_inv)
        #
        #
        #     use_mask = False
        #     # if i > 11 and i < 16:
        #     #     use_mask = True
        #     loss_2, smooth_mask = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs,
        #                                               1 - plane_mask_tgt.int(),
        #                                               use_mask=use_mask)
        #
        #     depth_write_mask = write_mask.clone().detach()
        #     depth_write_mask[:, :, :255, :319] = smooth_mask * depth_write_mask[:, :, :255, :319]
        #     # print('bug',(finetune_seg_mask*seg_line_mask).shape)
        #
        #     print('online_iter', online_iter)
        #     if final_line is None:
        #         if i > 20 :
        #             last_a = a_ini
        #             last_b = b_ini
        #             # last_d = d_ini
        #             a_ini = a_ini_r
        #             b_ini = b_ini_r
        #             d_ini = d_ini_r
        #             # loss_norm = a_ini * last_a.detach() + last_b.detach() * b_ini + 1
        #
        #
        #         # print('only one plane', a_ini, b_ini, d_ini)
        #         loss_plane1, _, _, _ = fit_plane(True, tgt_depth[0], global_pose_plane.unsqueeze(0), intrinsics[0],
        #                                          # plane_mask_tgt*depth_write_mask,
        #                                          total_seg_mask,
        #                                          online_iter,
        #                                          a_ini.detach(), b_ini.detach(), d_ini.detach(), i, flag='tgt')
        #         # if loss_norm != 0:
        #         #     loss_norm_abs = torch.abs(loss_norm)
        #         # else:
        #         #     loss_norm_abs = loss_norm
        #         loss_plane = loss_plane1
        #     else:
        #         # print('have two planes',a_ini, b_ini,d_ini,a_ini_r, b_ini_r,d_ini_r )
        #
        #         if plane_flag_l:
        #             print('plane left')
        #             loss_plane1, _, _, _ = fit_plane(True, tgt_depth[0], global_pose_plane.unsqueeze(0).detach(), intrinsics[0],
        #                                              # plane_mask_tgt*depth_write_mask,
        #                                              total_seg_mask*seg_line_mask_l,
        #                                              online_iter,
        #                                              a_ini.detach(), b_ini.detach(), d_ini.detach(), i, flag='tgt')
        #         else:
        #             loss_plane1 = 0
        #         # if i == 12:
        #         #     plane_flag_r = plane_flag_r and online_iter >=10
        #         if plane_flag_r:
        #             # print('plane right')
        #             loss_plane2, _, _, _ = fit_plane(True, tgt_depth[0], global_pose_plane.unsqueeze(0).detach(),
        #                                              intrinsics[0],
        #                                              # plane_mask_tgt*depth_write_mask,
        #                                              total_seg_mask * seg_line_mask_r,
        #                                              online_iter,
        #                                              a_ini_r.detach(), b_ini_r.detach(), d_ini_r.detach(), i, flag='next')
        #                                              # a_ini_r.detach(), b_ini_r.detach(), -7.0, i, flag='r')
        #             # loss_norm = a_ini_r * a_ini.detach() + b_ini.detach() * b_ini_r + 1
        #         else:
        #             loss_plane2 = 0
        #         loss_plane = loss_plane1 + loss_plane2
        #
        #     loss_F = compute_F_loss(F_poses, poses) + compute_F_loss(F_poses_inv, poses_inv)
        #
        #     w_pc = 0.1
        #     w_p = 1.0
        #     w1 = 1.
        #     w2 = 0.1
        #     w_m = 1.5
        #
        #
        #     depth_write_mask = write_mask.clone().detach()
        #     # depth_write_mask[:, :, :255, :319] = smooth_mask * depth_write_mask[:, :, :255, :319]
        #
        #     # if i >10 and i <16:
        #     if i > 29 and i <38:
        #     # # if i > 20:
        #     #     # w_p = 2.5 * torch.sum(plane_mask_tgt) / 81920
        #         w_p = 0.
        #
        #     if i >39:
        #         w_p = 1.
        #
        #     loss = w1 * loss_1 + 0.5 * loss_3 + loss_F + w2 * loss_2 + w_m * loss_match + 0.1*w_p * loss_plane + w_pc*loss_pc
        #
        #     if i < 19:
        #         loss1_stop = torch.abs(loss_1 - last_loss_1) < 0.01
        #     else:
        #         loss1_stop = torch.abs(loss_1 - last_loss_1) < 0.01
        #
        #     loss_stop = loss1_stop and (torch.abs(loss_2 - last_loss_2) < 0.01) and (
        #                 torch.abs(loss_3 - last_loss_3) < 0.01) \
        #                 and (torch.abs(loss_F - last_loss_F) < 0.01) and (
        #                             torch.abs(loss_match - last_loss_match) < 0.01) \
        #
        #     last_loss_1 = loss_1
        #     last_loss_2 = loss_2
        #     last_loss_3 = loss_3
        #     last_loss_F = loss_F
        #     last_loss_match = loss_match
        #
        #     if not next_fwd:
        #         # optimizer.zero_grad()
        #         ref_img_last = ref_imgs[0]
        #         tgt_img_last = tgt_img
        #
        #     loss.backward()
        #
        #     if next_fwd:
        #         optimizer.step()
        #         optimizer.zero_grad()

        with torch.no_grad():
            # if i > 0 and next_fwd:
            #     ref_img_test = ref_img_last
            #     tgt_img_test = tgt_img_last
            #     i = i-1
            # else:
            #     continue
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            pose = poseNet(ref_imgs[0], tgt_img)
            tgt_depth = 1. / depthNet(tgt_img)[0]

            pose_write = pose
            print('pose_write!!!!!!!!!!!!!!!!!!!!!', i)

            depth_write = tgt_depth
            pose_mat = pose_vec2mat(pose_write).detach().squeeze(0).cpu().numpy()
            # pose_mat = new_pose_f[0].detach().squeeze(0).cpu().numpy()
            # pose_mat = torch.cat([F_poses_inv[0][0][:3, :3], pose_vec2mat(pose_write)[0][:3, 3]], dim=0).cpu().numpy()
            # print(pose_mat)
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)

            # relative_pose_gt = np.linalg.inv(poses_pool[i])@poses_pool[i+1]
            # print('relative_pose_gt', relative_pose_gt)
            # print('pose_mat', pose_mat)
            #
            # rotation_gt = relative_pose_gt[:3,:3]
            # theta_z_gt = np.arctan2(rotation_gt[1,0], rotation_gt[0,0])
            # theta_y_gt = np.arctan2((-1)*rotation_gt[2,0], np.sqrt(rotation_gt[2,1]**2 + rotation_gt[2,2]**2))
            # theta_x_gt = np.arctan2(rotation_gt[2,1], rotation_gt[2,2])
            # rotation_est = pose_mat[:3,:3]
            # theta_z_est = np.arctan2(rotation_est[1, 0], rotation_est[0, 0])
            # theta_y_est = np.arctan2((-1) * rotation_est[2, 0], np.sqrt(rotation_est[2, 1] ** 2 + rotation_est[2, 2] ** 2))
            # theta_x_est = np.arctan2(rotation_est[2, 1], rotation_est[2, 2])
            # print('gt_angle', theta_x_gt, theta_y_gt, theta_z_gt)
            # print('est_angle', theta_x_est, theta_y_est, theta_z_est)
            # if theta_x_est*theta_x_gt<0:
            #     theta_x_est = -theta_x_est
            #     theta_y_est = -theta_y_est
            #     theta_z_est = -theta_z_est
            # rotation_error = np.abs((theta_x_gt-theta_x_est)/theta_x_gt) \
            #                  + np.abs((theta_y_gt-theta_y_est)/theta_y_gt) \
            #                  + np.abs((theta_z_gt-theta_z_est)/theta_z_gt)
            # global_rotation_error += (rotation_error/3)
            # print('rotation_error', rotation_error/3)
            #
            # trans_gt = relative_pose_gt[:3, 3]
            # trans_est = pose_mat[:3, 3]
            # max_idx = np.argmax(np.abs(trans_gt), axis=-1)
            # scale_align = trans_gt[max_idx]/trans_est[max_idx]
            # # scale_align = np.median(trans_gt[0])/np.median(trans_est[0])
            # print('gt_trans', trans_gt)
            # print('est_trans', scale_align*trans_est)
            # trans_error = np.mean(np.abs((trans_gt-trans_est*scale_align)/trans_gt))
            # print('trans_error', trans_error)
            # global_trans_error += trans_error
            global_pose[:3,:3] = R.from_matrix(global_pose[:3,:3]).as_matrix()

            poses_write.append(global_pose[0:3, :].reshape(1, 12))

            output_depths.append(depth_write[0,0])
            depth_show = tensor2array(1./depth_write, max_value=None, colormap='magma')
            depth_show = np.transpose(depth_show, (1, 2, 0))
            # depth_show = cv2.cvtColor(depth_show, cv2.COLOR_BGR2RGB)
            # cv2.imshow('depth_show', depth_show)
            # cv2.waitKey(0)
            depth_write = depth_write / 2.0 * 65535.0

            # print(depth_show.shape)
            # stop
            # depth_write = depth_write / 20.0 * 65535.0
            depth_write = (depth_write[0]).cpu().numpy().astype(np.uint16)

            # depth_write = depth_write.cpu().detach().numpy()

            # print('depth_write', depth_write.shape)

            tgt_img_write = cv2.cvtColor(np.transpose(tgt_img[0].cpu().detach().numpy(), (1, 2, 0)),cv2.COLOR_BGR2RGB)
            imsave(
                '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/test_depth/depth_' + str(i) + '.png',
                # depth_show)
                np.transpose(depth_write,(1,2,0)))
            imsave(
                '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/test_depth/color_' + str(i) + '.png',
                tgt_img_write)

    # print(poses_write)
        # stop
    # average_rotation_error = global_rotation_error/(len(poses_pool)-3)
    # average_trans_error = global_trans_error/(len(poses_pool)-3)
    # print('average_rotation_error', average_rotation_error)
    # print('average_trans_error', average_trans_error)
    poses_cat = np.concatenate(poses_write, axis=0)
    txt_filename = Path("/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/pose_318.txt")
    np.savetxt(txt_filename, poses_cat, delimiter=' ', fmt='%1.8e')
    return torch.stack(output_depths)


if __name__ == '__main__':
    main()
