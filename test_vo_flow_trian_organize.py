import numpy as np
import argparse
import torch.optim
import copy
import flow
import custom_transforms
import cv2
# import matplotlib
import torch.nn.functional as F
from skimage.transform import resize as imresize
from skimage.measure import label, regionprops
from path import Path
from datasets.sequence_folders_test import SequenceFolderTest
from test1.loss_functions_test2 import compute_photo_and_geometry_loss, compute_smooth_loss, compute_errors
from test1.inverse_warp_test1 import pose_vec2mat
from imageio import imsave
from flow.ransac import reduced_ransac
from models import DispResNet, PoseResNet
from segment.demo_modify import run as seg, line_detect
from utils import tensor2array
from scipy.spatial.transform import Rotation as R

# matplotlib.use('TkAgg')

filter = reduced_ransac(check_num=6000, thres=0.2, dataset='nyu')
show_flag = False

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH', help='path to pre-trained Flow net model')
parser.add_argument('--input-folder', dest='input_folder', default=None, metavar='PATH', help='path to the input images')
parser.add_argument('--output-folder', dest='output_folder', default=None, metavar='PATH', help='path to store the output keyframes, the depth maps and the global poses')
parser.add_argument('--with-gt', dest='with_gt', action='store_true', default=False, help='whether to evaluate the oupt depths')
parser.add_argument('--intrinsics', dest='intrinsics_txt', default=None, metavar='PATH', help='path to the txt file of the camera intrinsics')

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(filename):
    print('filename',filename)
    if filename.split('/')[-3] == 'depth' or filename.split('/')[-2] == 'depth':
        img = cv2.imread(filename,-1).astype(np.int16)
        img = cv2.resize(img, (320, 256), interpolation=cv2.INTER_NEAREST).astype(np.int16)
        tensor_img = torch.from_numpy(img).to(device)
    else:
        img = cv2.imread(filename).astype(np.float32)
        h, w, _ = img.shape
        img = imresize(img, (256, 320)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
    return tensor_img

def meshgrid(h, w):
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    meshgrid = np.transpose(np.stack([xx, yy], axis=-1), [2, 0, 1])  # [2,h,w]
    meshgrid = torch.from_numpy(meshgrid)
    return meshgrid

def rt_from_fundamental_mat_nyu(fmat, K, depth_match):
    # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
    verify_match = depth_match.transpose(1, 2).cpu().detach().numpy()
    b = fmat.shape[0]
    fmat_ = K.transpose(1, 2).bmm(fmat)
    essential_mat = fmat_.bmm(K)  # E = K^T * F *K
    iden = torch.cat([torch.eye(3), torch.zeros([3, 1])], -1).unsqueeze(0).repeat(b, 1, 1).to(
        device)  # [b,3,4]
    P1_K = K.bmm(iden)  # P1 with identity rotation and zero translation
    flags = []
    number_inliers = []
    P2 = []

    for i in range(b):
        cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'),
                                        verify_match[i, :, :2].astype('float64'), \
                                        verify_match[i, :, 2:].astype('float64'),
                                        cameraMatrix=K[i, :, :].cpu().detach().numpy().astype('float64'))
        p2 = torch.from_numpy(np.concatenate([R, t], axis=-1)).float().to(device)
        P2.append(p2)
        if cnum > depth_match.shape[-1] / 7.0:
            flags.append(1)
        else:
            flags.append(0)
        number_inliers.append(cnum)
    P2 = torch.stack(P2, axis=0)
    return P2

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

    for iter in range(100):
        # read next image
        tensor_img2_name = test_files[iter+1]
        tensor_img2 = load_tensor_image(test_files[iter+1])

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
            if args.with_gt:
                print('yes')
                gt_depth = load_tensor_image(args.input_folder + 'depth/' + img_name)
                gt_depth[gt_depth ==0] = 5000
                gt_depths.append(gt_depth)
            else:
                print('no')

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

            boundary_mask_inv = (bwd_corres[:, 0, :, :] > 10) * (bwd_corres[:, 0, :, :] < 310) * (
                    bwd_corres[:, 1, :, :] > 10) * (bwd_corres[:, 1, :, :] < 250)

            F_final_1, F_match = filter(fwd_match, boundary_mask.int())
            F_final_2, F_match_inv = filter(bwd_match, boundary_mask_inv.int())

            K = K_ms.unsqueeze(0).to(device)
            P2 = rt_from_fundamental_mat_nyu(F_final_1.detach().to(device), K, F_match.clone())
            P2_inv = rt_from_fundamental_mat_nyu(F_final_2.detach().to(device), K, F_match_inv.clone())
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
    output_depths = online_adaption(pose_net, depth_net, optimizer, optimizer_SGD, train_loader,
                                    [F_poses_pool, F_poses_pool_inv], [reproj_matches, reproj_matches_inv], plane_masks,
                                    args)
    if args.with_gt:
        gt_depths = torch.stack(gt_depths)# n,1,1,256,320
        print('**********errors************',compute_errors(gt_depths[:-2]/500, output_depths, 'nyu'))


def generate_training_set(keyframes_path, K_path, train_transform):
    return SequenceFolderTest(keyframes_path, K_path, train_transform)

def compute_pose_with_inv(pose_net, tgt_img, ref_imgs, poses_f, poses_f_inv):
    poses = []
    poses_inv = []
    poses_f_align = []
    poses_inv_f_align = []

    for ref_img, pose_f, pose_f_inv in zip(ref_imgs, poses_f, poses_f_inv) :
        pose = pose_net(tgt_img, ref_img).to(device)
        pose_inv = pose_net(ref_img, tgt_img).to(device)
        pose_mat = pose_vec2mat(pose)
        pose_mat_inv = pose_vec2mat(pose_inv)
        poses.append(pose_mat)
        poses_inv.append(pose_mat_inv)


        t_max_idx = torch.max(torch.abs(pose_mat[0, :, 3]), dim=0)[1]
        scale = pose_mat[:, t_max_idx, 3]/pose_f[:, t_max_idx, 3]
        pose_mat_new = torch.cat([pose_f[:, :, :3], scale*pose_f[:, :, 3:]], dim=-1)

        t_max_idx_inv = torch.max(torch.abs(pose_mat_inv[0, :, 3]), dim=0)[1]
        scale_inv = pose_mat_inv[:, t_max_idx_inv, 3] / pose_f_inv[:, t_max_idx_inv, 3]
        pose_mat_new_inv = torch.cat([pose_f_inv[:, :, :3], scale_inv * pose_f_inv[:, :, 3:]], dim=-1)
        poses_f_align.append(pose_mat_new)
        poses_inv_f_align.append(pose_mat_new_inv)
    return poses, poses_inv, poses_f_align, poses_inv_f_align

def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [(1 / disp.to(device)) for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1 / disp.to(device) for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths

def compute_F_loss(poses_f, poses, poses_inv):
    loss_R = 0
    loss_t = 0
    for pose_f, pose, pose_inv in zip(poses_f, poses, poses_inv):
        pose_mat = pose #b, 3, 4
        loss_R = loss_R + torch.mean(torch.abs(pose_f[:, :, :3] - pose_mat[:, :, :3]))
        t_weight = torch.abs((pose_mat[:, :, 3]/torch.norm(pose_mat[:, :, 3]))).detach()
        loss_t += torch.mean(t_weight*torch.abs(pose_f[:, :, 3]-pose_mat[:, :, 3]/torch.norm(pose_mat[:, :, 3])))
    return loss_R+loss_t

def largestConnectComponent(bw_img, multiple):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    #!!!! n: not include the background(label 0)

    max_label = 0
    max_num = 0

    if multiple:
        connect_regions = []
        bboxs = []
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        if max_num < 5000:
            return connect_regions
        for i in range(1, num+1):
            mcr = (labeled_img == i)
            minr, minc, maxr, maxc = regionprops(labeled_img)[i-1].bbox
            bboxs.append([minr, minc, maxr, maxc])
            if max_num!= 0 and np.sum(labeled_img == i)/max_num > 0.3:
                connect_regions.append(mcr)
        return connect_regions
    else:
        for i in range(1,num+1):
            if np.sum(labeled_img == i) > max_num and i!=0:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mcr = (labeled_img == max_label)

        minr, minc, maxr, maxc = regionprops(labeled_img)[max_label-1].bbox
        return labeled_img, max_label, [minr, minc, maxr, maxc], mcr

def calculate_all_norm(tgt_depth):
    tgt_depth_normal = torch.zeros([320, 320]).to(device)
    tgt_depth_normal[:256, :] = tgt_depth * 300
    normal = depth2normal(tgt_depth_normal) * 255  #3, 318, 318
    normal_map = torch.zeros([1, 3, 256, 320]).to(device)
    normal_map[0, :, 1:-1, 1:319] = normal[:, :254, :]
    return normal_map

def calculate_total_seg(mask_depth, plane_mask, seg_map, sobel):

    plane_mask_copy = plane_mask

    seg_depth_flatten = torch.nonzero(plane_mask_copy.reshape(-1))  # select pixel indices on the plane
    seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)  # select potential plane pixel in segment map
    seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)  # count the labels in the segment

    max_idx = torch.max(lab_count, dim=0)[1]  # exact the largest part in the segment
    seg_labs_max = seg_labs_select[max_idx]
    seg_reshape = seg_map.reshape(1, 1, 256, 320)
    seg_mask = torch.where(seg_reshape == seg_labs_max, torch.ones_like(mask_depth),
                           torch.zeros_like(mask_depth))
    _, _, bbox, mcr = largestConnectComponent(seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8), multiple=False)
    max_mask_return = seg_mask[0,0]
    mask_depth = (mask_depth-seg_mask)>0
    total_seg_mask = plane_mask
    if torch.sum(mask_depth)>0:
        seg_depth_flatten = torch.nonzero(mask_depth.int().reshape(-1))
        seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)
        seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)
        max_idx = torch.max(lab_count, dim=0)[1]
        seg_labs_max = seg_labs_select[max_idx]
        seg_reshape = seg_map.reshape(1, 1, 256, 320)
        seg_mask_second_plane = torch.where(seg_reshape == seg_labs_max, torch.ones_like(mask_depth),
                               torch.zeros_like(mask_depth))
        depth_flatten = torch.nonzero(seg_mask_second_plane.reshape(-1))
        second_depth = torch.gather(torch.from_numpy(sobel).reshape(-1).to(device), index=depth_flatten[:, 0], dim=0)

        labeled_img, max_label, _ = largestConnectComponent(seg_mask_second_plane[0, 0].detach().cpu().numpy().astype(np.uint8),False)
        label_mask_second = (labeled_img == max_label)
        if np.sum(label_mask_second) > 5000 and torch.var(second_depth.float()) < 1000:
            # print('second_plane')
            total_seg_mask = plane_mask + torch.tensor(label_mask_second).unsqueeze(0).unsqueeze(0).to(device)

    return torch.tensor(max_mask_return).unsqueeze(0).unsqueeze(0).to(device), bbox,(total_seg_mask > 0).int()

def calculate_seg_new(src, plane_mask, seg_map, depth, sobel,grad,flag):

    plane_mask_copy = plane_mask
    max_mask_return = torch.zeros_like(plane_mask).to(device).int()
    class_idx = 1
    max_area = None

    while torch.sum(plane_mask_copy) > 5000:

        seg_depth_flatten = torch.nonzero(plane_mask_copy.reshape(-1))  # select pixel indices on the plane
        index_boundary = torch.nonzero(plane_mask_copy[0,0])  #n,2
        seg_img = torch.gather(seg_map, index=seg_depth_flatten[:, 0], dim=0)  # select potential plane pixel in segment map
        seg_labs_select, lab_count = torch.unique(seg_img, return_counts=True)  # count the labels in the segment

        max_count, max_idx = torch.max(lab_count, dim=0)  # exact the largest part in the segment
        if max_area is None:
            max_area = max_count
        else:
            if max_count/max_area < 0.2:
                break
        seg_labs_max = seg_labs_select[max_idx]

        seg_reshape = seg_map.reshape(1, 1, 256, 320)


        seg_mask = torch.where(seg_reshape == seg_labs_max, torch.ones_like(plane_mask),
                               torch.zeros_like(plane_mask))

        _, _, bbox, mcr = largestConnectComponent(seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8),False)
        result = cv2.bitwise_and(src, src, mask=mcr.astype(np.uint8))
        result_seg_mask = cv2.bitwise_and(src, src, mask=seg_mask[0, 0].detach().cpu().numpy().astype(np.uint8))
        max_mask_return += (class_idx * torch.from_numpy(mcr).unsqueeze(0).unsqueeze(0).to(device).int())

        depth_index = torch.nonzero(seg_mask.reshape(-1))
        grad_index = torch.nonzero(torch.from_numpy(mcr).reshape(-1).to(device))
        grad_select = torch.gather(torch.from_numpy(sobel).reshape(-1).to(device), index=grad_index[:, 0], dim=0)
        include_flag = True
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
            max_mask_return += (class_idx*torch.from_numpy(mcr).unsqueeze(0).unsqueeze(0).to(device).int())
        plane_mask_copy = ((plane_mask_copy.int()-seg_mask.int()) > 0).int()
        if flag == 'tgt':
            class_idx += 1

    return torch.tensor(max_mask_return)

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

def calculate_loss(i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose, Q_last, Q_last_f, src_seg_show):
    F_poses = [F_poses_with_inv[1][i], F_poses_with_inv[0][i + 1]]
    F_poses_inv = [F_poses_with_inv[0][i], F_poses_with_inv[1][i + 1]]

    matches = [match_with_inv[1][i], match_with_inv[0][i + 1]]
    matches_inv = [match_with_inv[0][i], match_with_inv[1][i + 1]]

    tgt_img = tgt_img.to(device)
    ref_imgs = [img.to(device) for img in ref_imgs]
    intrinsics = intrinsics.to(device)

    poses, poses_inv, poses_f, poses_inv_f = compute_pose_with_inv(poseNet, tgt_img, ref_imgs, F_poses, F_poses_inv)
    tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)
    # when the distance between epipolar pose and the predicted on is less than a certain threshold
    # the epipolar pose is considered as reliable
    # and the implicit epipolar constraint is applied
    pred_pose_inv = poses_inv[0].detach()

    pred_pose_inv[:, :, 3] = pred_pose_inv[:, :, 3]/torch.sqrt(pred_pose_inv[:, 0, 3]**2 + pred_pose_inv[:, 1, 3]**2 + pred_pose_inv[:, 2, 3]**2)

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

    loss_2 = compute_smooth_loss(tgt_depth, tgt_img, fused_grad_ori)

    # the explicit epipolar constraint
    loss_F = compute_F_loss(F_poses, poses, poses_inv) + compute_F_loss(F_poses_inv, poses_inv, poses)

    # loss_normal_adj = 0
    # loss_normal_single = 0
    # loss_normal = 0

    w1 = 1.0
    w2 = 0.1
    w_m = 1.5

    trans1 = pred_pose_inv[0, :3, 3]
    trans2 = F_poses_inv[0][0, :3, 3]
    cos_theta = (trans1[0]*trans2[0] + trans1[1]*trans2[1] + trans1[2] * trans2[2])/ \
                (torch.sqrt(trans1[0]**2 + trans1[1]**2 + trans1[2]**2)*torch.sqrt(trans2[0]**2 + trans2[1]**2 + trans2[2]**2))
    angle = np.degrees(torch.arccos(cos_theta).detach().cpu().detach())

    if angle >10:
        w_f = 0
    else:
        w_f = 1

    loss = w2 * loss_2 + w1*loss_1_ori + 0.5*loss_3_ori +w_f*(0.1*(loss_1 + 0.5 * loss_3))\
           # +0.1 * (loss_pc_ori[0] + loss_pc[0])

    return loss_1_ori, loss, photo_mask_ori, Q_curr_f, Q_curr_ori, fused_grad_ori

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

def online_adaption(poseNet, depthNet, optimizer, optimizer_SGD, train_loader, F_poses_with_inv, match_with_inv, plane_masks, args):
    poseNet.train()
    depthNet.train()

    global_pose = np.eye(4)
    poses_write = []
    output_depths = []

    global show_flag

    tpt_tgt_depth = torch.ones([1, 1, 256, 320])
    tpt_ref_depth1 = torch.ones_like(tpt_tgt_depth)
    tpt_ref_depth2 = torch.ones_like(tpt_tgt_depth)
    Q_last = None
    Q_last_f = None

    for i, (tgt_img, ref_imgs, tgt_img_next, ref_imgs_next, intrinsics, intrinsics_inv) in enumerate(train_loader):

        print("**********************************************")
        print('iter', i)

        stop_flag = False
        online_iter = 0

        max_iteration = 30

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]

        # recover the normalized image
        ori_img = img_norm2ori(tgt_img[0])
        ori_img_ref1 = img_norm2ori(ref_imgs[0][0])
        ori_img_ref2 = img_norm2ori(ref_imgs[1][0])

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

        pc_grad_for_seg = None
        last_loss1 = 0

        # warm-up training
        while not stop_flag and online_iter < max_iteration:
            online_iter += 1
            loss_pho, loss_curr, photo_masks, Q_curr_f, Q_curr_ori, pc_grad_for_seg = calculate_loss(i, tgt_img, ref_imgs, F_poses_with_inv, match_with_inv, intrinsics, poseNet, depthNet, global_pose, Q_last, Q_last_f, seg_show_tgt)

            param_last = copy.deepcopy(optimizer.param_groups[0]['params'])

            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()

            stop_flag = torch.abs(loss_pho - last_loss1) < 0.001
            last_loss1 = loss_pho

            # online meta-learning
            if online_iter > 0:
                _, loss_next, _, _, _, _ = calculate_loss(i+1, tgt_img_next, ref_imgs_next, F_poses_with_inv,
                                              match_with_inv, intrinsics, poseNet, depthNet, global_pose, None, None, seg_show_ref2)
                optimizer.zero_grad()
                loss_next.backward()
                for param_idx in range(len(optimizer.param_groups[0]['params'])):
                    optimizer.param_groups[0]['params'][param_idx].data = param_last[param_idx].data
                optimizer.step()

            del param_last

        tgt_depth, ref_depths = compute_depth(depthNet, tgt_img, ref_imgs)

        loss_stop = False
        online_iter = 0

        seg_map_tgt = torch.tensor(seg_map_tgt).to(device)
        seg_map_ref1 = torch.tensor(seg_map_ref1).to(device)
        seg_map_ref2 = torch.tensor(seg_map_ref2).to(device)

        max_seg_mask_tgt = calculate_seg_new(ori_img, photo_masks[0].unsqueeze(0), seg_map_tgt, tgt_depth[0], Sobelxy_tgt,
                                             pc_grad_for_seg, flag='tgt')
        max_seg_mask_ref1 = calculate_seg_new(ori_img_ref1, photo_masks[1].unsqueeze(0), seg_map_ref1, ref_depths[0][0], Sobelxy_ref1, None, flag='ref')
        max_seg_mask_ref2 = calculate_seg_new(ori_img_ref2, photo_masks[3].unsqueeze(0), seg_map_ref2, ref_depths[1][0], Sobelxy_ref2, None, flag='ref')

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

        result_tgt_final = cv2.bitwise_and(ori_img, ori_img,
                                           mask=final_mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8))
        cv2.imshow('result_final', result_tgt_final)
        cv2.waitKey(0)

        # fine extraction
        connect_regions = largestConnectComponent(final_mask_tensor[0, 0].detach().cpu().numpy().astype(np.uint8), True)

        pc_grad_show = np.expand_dims(pc_grad_for_seg.cpu().detach().numpy(), axis=2).astype(np.uint8)

        # cv2.imshow('pc_grad_for_seg[2]',pc_grad_show)
        # cv2.waitKey(0)

        P_p = []
        P_d = []
        line_return = None
        # for each connected region, detect the separating line in G_fuse
        for region_idx in range(len(connect_regions)):
            current_region = cv2.bitwise_and(ori_img, ori_img, mask=connect_regions[region_idx].astype(np.uint8))
            # if show_flag:
            cv2.imshow('current_region_ori', current_region)
            cv2.waitKey(0)

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

        second_max_iter = 5
        last_loss_1= 0
        while not loss_stop and online_iter < second_max_iter:
            # param_last = 0
            online_iter += 1
            loss_pho, loss_curr, Q_curr_f, Q_curr_ori = calculate_loss_norm(i, tgt_img, ref_imgs, F_poses_with_inv,
                                                                 match_with_inv, intrinsics, poseNet, depthNet, global_pose, [P_d, P_p],Q_last, Q_last_f, seg_show_tgt)

            optimizer.zero_grad()
            loss_curr.backward()
            optimizer.step()
            loss_stop = torch.abs(loss_pho - last_loss_1) < 0.01
            last_loss_1 = loss_pho
        Q_last = Q_curr_ori
        Q_last_f = Q_curr_f


        with torch.no_grad():
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            pose = poseNet(ref_imgs[0], tgt_img)
            tgt_depth = 1. / depthNet(tgt_img)[0]

            pose_write = pose
            print('pose_write!!!!!!!!!!!!!!!!!!!!!', i)

            depth_write = tgt_depth
            pose_mat = pose_vec2mat(pose_write).detach().squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)
            global_pose[:3,:3] = R.from_matrix(global_pose[:3,:3]).as_matrix()

            poses_write.append(global_pose[0:3, :].reshape(1, 12))

            output_depths.append(depth_write[0,0])
            depth_show = tensor2array(1./depth_write, max_value=None, colormap='magma')
            depth_show = np.transpose(depth_show, (1, 2, 0))
            # depth_show = cv2.cvtColor(depth_show, cv2.COLOR_BGR2RGB)
            # cv2.imshow('depth_show', depth_show)
            # cv2.waitKey(0)
            depth_write = depth_write / 2.0 * 65535.0
            depth_write = (depth_write[0]).cpu().numpy().astype(np.uint16)
            output_dir = args.output_folder
            tgt_img_write = cv2.cvtColor(np.transpose(tgt_img[0].cpu().detach().numpy(), (1, 2, 0)),cv2.COLOR_BGR2RGB)
            imsave(
                output_dir+'/depth/depth_' + str(i) + '.png',
                # '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/test_depth/depth_' + str(i) + '.png',
                # # depth_show)
                np.transpose(depth_write,(1,2,0)))
            imsave(
                # '/home/dokidoki/Unsupervised_SfM/Unsupervised_VO/SC/SC-SfMLearner/test_depth/color_' + str(i) + '.png',
                output_dir+'keyframes/color_' + str(i) + '.png',
                tgt_img_write)
    poses_cat = np.concatenate(poses_write, axis=0)
    txt_filename = Path(output_dir+"poses.txt")
    np.savetxt(txt_filename, poses_cat, delimiter=' ', fmt='%1.8e')
    return torch.stack(output_depths)


if __name__ == '__main__':
    main()
