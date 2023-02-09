from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from test1.inverse_warp_test1 import inverse_warp2, pose_vec2mat, pixel2cam, set_id_grid
import math
import numpy as np
import cv2
from torch.autograd import Variable
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from test_vo_flow_trian import Sobel_function
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pcl
import pcl.pcl_visualization
# from segment.demo_modify import run_only as seg_only

# Q_ref = None

# import pandas_profiling
# matplotlib.use('Qt5Agg')

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
x_train_proj = 0
y_train_proj = 0
z_train_proj = 0

x_train_proj2 = 0
y_train_proj2 = 0
z_train_proj2 = 0

x_show = 0
y_show = 0
z_show = 0

boundary = 0


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, matches, matches_inv, global_pose, Q_last):
    # print('o', online_iter)
    photo_loss = 0
    geometry_loss = 0
    trian_loss = 0
    pc_loss = 0
    match_loss = 0
    # print_geo_loss = 0
    # print_pho_loss = 0
    pc_last = 0
    photo_masks = []

    num_scales = min(len(tgt_depth), max_scales)
    pc_enhance = 0
    # print('num_scales',num_scales)
    for ref_img, ref_depth, pose, pose_inv, match, match_inv in zip(ref_imgs, ref_depths, poses, poses_inv, matches, matches_inv):
        # print('match',match)
        for s in range(num_scales):

            # # downsample img
            # b, _, h, w = tgt_depth[s].size()
            # downscale = tgt_img.size(2)/h
            # if s == 0:
            #     tgt_img_scaled = tgt_img
            #     ref_img_scaled = ref_img
            # else:
            #     tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            #     ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            # intrinsic_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            # tgt_depth_scaled = tgt_depth[s]
            # ref_depth_scaled = ref_depth[s]

            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='nearest')
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='nearest')

            # reproj_match = []
            # reproj_match_inv = []
            photo_loss1, geometry_loss1, match_loss1, photo_mask_tgt, pc_loss1, Q_curr, fused_grad= compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, match, global_pose,Q_last)
            photo_loss2, geometry_loss2, match_loss2, photo_mask_ref, _, _, _= compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode, None, match_inv,global_pose,Q_last)
            photo_loss2, geometry_loss2, trian_loss2, pc_loss2, match_loss2 = 0,0,0,0,0
            # print('photo1,2', photo_loss1, photo_loss2)
            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)
            # trian_loss += (trian_loss1 + trian_loss2)
            pc_loss += pc_loss1
            # pc_last = (pc_loss1[1])
            # pc_enhance = pc_loss1[2]

            match_loss += (match_loss1 + match_loss2)

            # print_pho_loss += (mask1[0]+mask2[0])
            # print_geo_loss += (mask1[1]+mask2[1])

            photo_masks.append(photo_mask_tgt)
            photo_masks.append(photo_mask_ref)

            # photo_loss += photo_loss1
            # geometry_loss += geometry_loss1
            # trian_loss += trian_loss1
            # pc_loss += pc_loss1
            #
            # match_loss += match_loss1

    return photo_loss, geometry_loss, match_loss, photo_masks, pc_loss, Q_curr, fused_grad

def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, match, global_pose, Q_last):

    ref_img_warped, ref_coord_warped_norm, _, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)
    # print('ref_coord_warped_norm', ref_coord_warped[0, 100, 100, :])
    # print('match', (match[0, 100, 100, 0]+1)*(320-1)/2)
    # print('match', (match[0, 100, 100, 1]+1)*(256-1)/2)
    # stop
    # mask_depth = (tgt_depth[0] < (torch.mean(tgt_depth[0]) - torch.var(tgt_depth[0]))).int()
    # res_point_cloud = compute_3d_loss(ref_depth, tgt_depth, pose, intrinsic)
    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
    # print(ref_coord_warped_norm.shape)
    # print(match.shape)
    diff_match = (ref_coord_warped_norm - match).abs().clamp(0, 1).reshape(1, -1, 2).transpose(1,2).reshape(1, 2, 256,320)
    # print(ref_coord_warped)
    # boundary_x = (ref_coord_warped[:, :, :, 0] > 0) * (ref_coord_warped[:, :, :, 0] < 320)
    # boundary_y = (ref_coord_warped[:, :, :, 1] > 0) * (ref_coord_warped[:, :, :, 1] < 256)
    # boundary_mask = boundary_x * boundary_y
    # print(ref_coord_warped.shape)

    # b = ref_coord_warped.shape[0]
    # h = ref_coord_warped.shape[1]
    # w = ref_coord_warped.shape[2]
    # i_range = torch.arange(0, h).view(1, h, 1).expand(
    #     1, h, w).to(device) # [1, H, W]
    # j_range = torch.arange(0, w).view(1, 1, w).expand(
    #     1, h, w).to(device)  # [1, H, W]
    # grid = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]
    # print(grid.shape)
    # grid_show = grid.reshape(grid.shape[0],2,-1).transpose(1,2)  # [1, 2, H, W]

    # imshow
    # if online_iter == 1:
    #     tgt_img_show = tgt_img[0].cpu().numpy().transpose((1, 2, 0))
    #     ref_img_show = ref_img[0].cpu().numpy().transpose((1, 2, 0))
    #     depth_match_circle = ref_coord_warped.reshape([ref_img.shape[0], -1, 2])
    #     # print(depth_match_circle.shape)
    #     # stop
    #     b_img = 0
    #     rand_int = torch.randint(0, depth_match_circle.shape[1], [100])
    #     for n_idx in range(100):
    #         # if i%100 == 0:
    #         i = rand_int[n_idx]
    #         # print('i',i)
    #         x1 = np.int(grid_show[b_img, i, 0].cpu().detach().numpy())
    #         y1 = np.int(grid_show[b_img, i, 1].cpu().detach().numpy())
    #         x2 = np.int(depth_match_circle[b_img, i, 0].cpu().detach().numpy())
    #         y2 = np.int(depth_match_circle[b_img, i, 1].cpu().detach().numpy())
    #         if x1 > 0 and x1 < 320 and y1 >0 and y1 < 256 and x2 > 0 and x2 < 320 and y2 >0 and y2 < 256:
    #         # print(x1, y1)
    #         # print(x2, y2)
    #             tgt_img_show = cv2.circle(np.ascontiguousarray(tgt_img_show), (x1, y1), 5, (0, 0, 255))
    #             ref_img_show = cv2.circle(np.ascontiguousarray(ref_img_show), (x2, y2), 5, (0, 0, 255))
    #     cv2.imshow("img1", tgt_img_show)
    #     cv2.imshow("img2", ref_img_show)
    #     cv2.waitKey(0)

    # grid_reshape = grid.transpose(1,3).transpose(1,2).repeat(b,1,1,1) # [b, H, W, 2]
    # print(pixel_coords.shape)
    # flow_diff = ref_coord_warped - grid_reshape #b,h,w,2

    # loss_flow_smooth = compute_loss_flow_smooth(flow_diff.transpose(1,3).transpose(2,3), tgt_img)

    # print('grid_reshape', grid_reshape)
    # flow_diff_xy = torch.sqrt(flow_diff[:, :, :, 0]**2 + flow_diff[:, :, :, 1]**2)
    # flow_mask = flow_diff_xy > 20.0
    # loss_trian = trian_constraint_loss(pose, grid_reshape, ref_coord_warped, boundary_mask*no_plane_mask, intrinsic.inverse(), ref_depth)
    # loss_trian = 0
    # print(ref_img_warped)
    # stop
    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask
    #
    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        # if torch.var(tgt_depth[0]) < 0.003:
        #     # print('var', torch.var(tgt_depth[0]))
        #     weight_mask = (1 - diff_depth)*mask_depth
        # else:
        weight_mask = (1 - diff_depth)
        # print(weight_mask.shape) # b,1,h,w
        # diff_coord = diff_coord * weight_mask
        diff_img = diff_img * weight_mask


    # compute all loss
    # reconstruction_loss = mean_on_mask(diff_coord, valid_mask)
    # print(valid_mask.shape)
    # print(no_plane_mask.shape)
    # valid_mask_pho = valid_mask * no_plane_mask
    # valid_mask_boundary = valid_mask * plane_mask

    # no_mask = torch.ones_like(valid_mask_pho) - valid_mask_pho

    photometric_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    # photometric_loss_print = mean_on_mask(diff_img, valid_mask_boundary)
    # geometry_consistency_loss_print = mean_on_mask(diff_depth, valid_mask_boundary)
    match_loss = mean_on_mask(diff_match, valid_mask)
    # match_loss = mean_on_mask(diff_match,valid_mask)
    # res_pc_loss = 0
    # print('online_iter', online_iter)
    # if Q_last is not None:
    #     if online_iter[1] is None:
    #         global_pc_loss, Q_curr, fused_grad = [0,0,0]
    #     else:
    #         global_pc_loss, Q_curr, fused_grad = compute_3d_loss(tgt_depth,tgt_img, ref_depth, pose, intrinsic, match, global_pose, Q_last, src_seg_show_seg)
    # else:
    global_pc_loss, Q_curr, fused_grad = compute_3d_loss(tgt_depth, tgt_img, ref_depth, pose, intrinsic, match, global_pose, Q_last, src_seg_show_seg)
    mean_value = diff_img[:, 0, :, :] + diff_img[:, 1, :, :] + diff_img[:, 2, :, :]
    # print('mean_value', )
    pho_mask0 = mean_value < 1.5*photometric_loss
    # pho_mask1 = diff_img[:, 1, :, :] < torch.mean(diff_img[:, 1, :, :]) + 5.5*torch.var(diff_img[:, 1, :, :])
    # pho_mask2 = diff_img[:, 2, :, :] < photometric_loss
    # geometry_mask = diff_depth < torch.median(diff_depth) + 2.5*torch.var(diff_depth)
    # pho_mask = pho_mask0* pho_mask1*pho_mask2

    # print(pho_mask.shape)
    # print(geometry_mask.shape)
    # print(valid_mask_booundary.shape)
    # stop



    #gc_mask
    # photometric_loss = mean_on_mask(diff_img, valid_mask_pho)
    # geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask_pho)
    # res_pc_loss = compute_3d_loss(ref_depth, tgt_depth, pose, intrinsic, ref_coord_warped, valid_mask_pho)
    # match_loss = mean_on_mask(diff_match, valid_mask_pho)

    # print('pho', photometric_loss)

    # return photometric_loss, geometry_consistency_loss, loss_trian, loss_flow_smooth, reconstruction_loss
    # return photometric_loss, geometry_consistency_loss, loss_trian, res_pc_loss, valid_mask_boundary*geometry_mask*pho_mask.unsqueeze(1), match_loss
    return photometric_loss, geometry_consistency_loss, match_loss, pho_mask0, global_pc_loss, Q_curr, fused_grad

# def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode, flow_match):
#
#     ref_img_warped, ref_coord_warped_norm, ref_coord_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)
#
#     mask_depth = (tgt_depth[0] < (torch.mean(tgt_depth[0]) - torch.var(tgt_depth[0]))).int()
#     # res_point_cloud = compute_3d_loss(ref_depth, tgt_depth, pose, intrinsic)
#     diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)
#     # print(flow_match)
#     # print(ref_coord_warped)
#     # stop
#     X_mask = ((flow_match[:, :, :, 0] <= 1) * (flow_match[:, :, :, 0] >= -1)).detach()
#     # make sure that no point in warped image is a combinaison of im and gray
#     flow_match[:, :, :, 0][X_mask] = 2
#     Y_mask = ((flow_match[:, :, :, 1] <= 1) + (flow_match[:, :, :, 1] >= -1)).detach()
#     flow_match[:, :, :, 0][Y_mask] = 2
#
#     flow_mask = X_mask*Y_mask
#     # print('flow_mask',torch.sum(flow_mask[0]))
#     # print('flow_mask',flow_mask.shape)
#
#     diff_coord = (flow_match - ref_coord_warped_norm).abs().clamp(0, 1).transpose(3,1).transpose(2,3)
#
#     diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)
#
#     # print(ref_coord_warped)
#     boundary_x = (ref_coord_warped[:, :, :, 0] >0 ) * ( ref_coord_warped[:, :, :, 0] <320 )
#     boundary_y = (ref_coord_warped[:, :, :, 1] >0 ) * ( ref_coord_warped[:, :, :, 1] <256 )
#     boundary_mask = boundary_x * boundary_y
#     # print(ref_coord_warped.shape)
#
#     b = ref_coord_warped.shape[0]
#     h = ref_coord_warped.shape[1]
#     w = ref_coord_warped.shape[2]
#     i_range = torch.arange(0, h).view(1, h, 1).expand(
#         1, h, w).to(device) # [1, H, W]
#     j_range = torch.arange(0, w).view(1, 1, w).expand(
#         1, h, w).to(device)  # [1, H, W]
#     grid = torch.stack((j_range, i_range), dim=1)  # [1, 2, H, W]
#     # print(grid.shape)
#     grid_show = grid.reshape(grid.shape[0],2,-1).transpose(1,2)  # [1, 2, H, W]
#     # print(grid_show[0,:,:])
#     # stop
#     # grid = set_id_grid(ref_coord_warped[:,:,:,0])
#
#     # imshow
#     tgt_img_show = tgt_img[0].cpu().numpy().transpose((1, 2, 0))
#     ref_img_show = ref_img[0].cpu().numpy().transpose((1, 2, 0))
#     depth_match_circle = ref_coord_warped.reshape([ref_img.shape[0], -1, 2])
#     # print(depth_match_circle.shape)
#     # stop
#     b_img = 0
#     rand_int = torch.randint(0, depth_match_circle.shape[1], [100])
#     # for n_idx in range(100):
#     #     # if i%100 == 0:
#     #     i = rand_int[n_idx]
#     #     # print('i',i)
#     #     x1 = np.int(grid_show[b_img, i, 0].cpu().detach().numpy())
#     #     y1 = np.int(grid_show[b_img, i, 1].cpu().detach().numpy())
#     #     x2 = np.int(depth_match_circle[b_img, i, 0].cpu().detach().numpy())
#     #     y2 = np.int(depth_match_circle[b_img, i, 1].cpu().detach().numpy())
#     #     # print(x1, y1)
#     #     # print(x2, y2)
#     #     tgt_img_show = cv2.circle(np.ascontiguousarray(tgt_img_show), (x1, y1), 5, (0, 0, 255))
#     #     ref_img_show = cv2.circle(np.ascontiguousarray(ref_img_show), (x2, y2), 5, (0, 0, 255))
#     # cv2.imshow("img1", tgt_img_show)
#     # cv2.imshow("img2", ref_img_show)
#     # cv2.waitKey(0)
#
#     grid_reshape = grid.transpose(1,3).transpose(1,2).repeat(b,1,1,1) # [b, H, W, 2]
#     # print(pixel_coords.shape)
#     flow_diff = ref_coord_warped - grid_reshape #b,h,w,2
#
#     # print(flow_diff.shape)
#     # print(flow_diff.transpose(1,3).transpose(2,3).shape)
#     loss_flow_smooth = compute_loss_flow_smooth(flow_diff.transpose(1,3).transpose(2,3), tgt_img)
#
#     # print('grid_reshape', grid_reshape)
#     flow_diff_xy = torch.sqrt(flow_diff[:, :, :, 0]**2 + flow_diff[:, :, :, 1]**2)
#     # flow_mask = flow_diff_xy > 20.0
#     loss_trian = trian_constraint_loss(pose, grid_reshape, ref_coord_warped, boundary_mask, intrinsic.inverse(), ref_depth)
#
#     # print(ref_img_warped)
#     # stop
#     if with_auto_mask == True:
#         auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
#         valid_mask = auto_mask
#     #
#     if with_ssim == True:
#         ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
#         diff_img = (0.15 * diff_img + 0.85 * ssim_map)
#
#     if with_mask == True:
#         # if torch.var(tgt_depth[0]) < 0.003:
#         #     # print('var', torch.var(tgt_depth[0]))
#         #     weight_mask = (1 - diff_depth)*mask_depth
#         # else:
#         weight_mask = (1 - diff_depth)
#         # print(weight_mask.shape) # b,1,h,w
#         diff_coord = diff_coord * weight_mask
#         diff_img = diff_img * weight_mask
#
#
#     # compute all loss
#     reconstruction_loss = mean_on_mask(diff_coord, valid_mask)
#     photometric_loss = mean_on_mask(diff_img, valid_mask)
#     geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
#     res_pc_loss = compute_3d_loss(ref_depth, tgt_depth, pose, intrinsic)
#     # print('pho', photometric_loss)
#
#     # return photometric_loss, geometry_consistency_loss, loss_trian, loss_flow_smooth, reconstruction_loss
#     return photometric_loss, geometry_consistency_loss, loss_trian, res_pc_loss

def trian_constraint_loss(pose, p2, p1, flow_mask, K_inv, ref_depth):
    # loss = 0
    # for pose, flow, ref_depth, flow_mask in zip(poses, flows, ref_depths, flows_mask):
        # pose_mat = pose_vec2mat(pose,rotation_mode='quat')  # b, 3, 4
    pose_mat = pose_vec2mat(pose)  # b, 3, 4
    # pose_mat = pose  # b, 3, 4
    R = pose_mat[:, :3, :3]
    t = pose_mat[:, :3, 3]
    points_norm1 = p1.reshape([p1.shape[0], 2, -1]).transpose(1, 2)  # b,n,2
    points_norm2 = p2.reshape([p2.shape[0], 2, -1]).transpose(1, 2)
    loss =single_trian_constraint(points_norm1,points_norm2, K_inv, R, t, ref_depth, flow_mask)
        # loss = loss + single_tian_constraint(match, K_inv, R, t, ref_depth)
    return loss

def single_trian_constraint(points_norm1, points_norm2, K_inv, R, t, depth2, flow_mask):
    b = points_norm1.shape[0]
    p_2 = points_norm1.transpose(1,2) #b, 2, n
    p_1 = points_norm2.transpose(1,2)

    x_2 = K_inv.bmm(torch.cat([p_2, torch.ones([b, 1, p_2.shape[2]]).to(device)], dim=1)) #b, 3, n
    x_1 = K_inv.bmm(torch.cat([p_1, torch.ones([b, 1, p_2.shape[2]]).to(device)], dim=1))

    Rx_2 = R.bmm(x_2) #b, 3, n

    x_1_hat_Rx_2 = torch.cross(x_1, Rx_2)

    # x_1_hat_t = torch.cross(x_1, t.unsqueeze(2).expand_as(x_1))
    # print(t.shape)
    x_1_hat_t = torch.cross(x_1, t.unsqueeze(2).repeat(1, 1, x_1.shape[-1]))
    # print(len(depth2))
    # print(depth2[0].shape)
    loss_trian = depth2.reshape([b,1,-1]).repeat(1,3,1) * x_1_hat_Rx_2 + x_1_hat_t
    # print(loss_trian.shape) b,3,n
    return torch.mean(torch.abs(loss_trian*flow_mask.reshape(b,1,-1).repeat(1,3,1)))

# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img, pc_grad):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # print(grad_disp_x.shape)
        # print(grad_disp_y.shape)
        # stop

        # mask_x = grad_disp_x > torch.mean(grad_disp_x) + 1.5*torch.var(grad_disp_x)
        # mask_y = grad_disp_y > torch.mean(grad_disp_y) + 1.5*torch.var(grad_disp_y)

        # mask_x_img = grad_img_x < torch.mean(grad_img_x) + 1.5 * torch.var(grad_img_x)
        # mask_y_img = grad_img_y < torch.mean(grad_img_y) + 1.5 * torch.var(grad_img_y)

        # if not use_mask:
        # return grad_disp_x.mean() + grad_disp_y.mean(), mask_x[:, :, :-1, :]*mask_y[:, :, :, :-1] *mask_x_img[:, :, :-1, :]*mask_y_img[:, :, :, :-1]
        # else:
        return grad_disp_x.mean() + grad_disp_y.mean()\
            # , mask_x[:, :, :-1, :]*mask_y[:, :, :, :-1] *mask_x_img[:, :, :-1, :]*mask_y_img[:, :, :, :-1]

    def get_smooth_loss_pc(disp, grad):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        # grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        # grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_img_x = grad[0]
        grad_img_y = grad[1]


        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        # print(grad_disp_x.shape)
        # print(grad_disp_y.shape)
        # stop

        # mask_x = grad_disp_x > torch.mean(grad_disp_x) + 1.5*torch.var(grad_disp_x)
        # mask_y = grad_disp_y > torch.mean(grad_disp_y) + 1.5*torch.var(grad_disp_y)
        #
        # mask_x_img = grad_img_x < torch.mean(grad_img_x) + 1.5 * torch.var(grad_img_x)
        # mask_y_img = grad_img_y < torch.mean(grad_img_y) + 1.5 * torch.var(grad_img_y)

        # if not use_mask:
        # return grad_disp_x.mean() + grad_disp_y.mean(), mask_x[:, :, :-1, :]*mask_y[:, :, :, :-1] *mask_x_img[:, :, :-1, :]*mask_y_img[:, :, :, :-1]
        # else:
        return grad_disp_x.mean() + grad_disp_y.mean()

    # if pc_grad is not None and pc_grad != 0:
    pc_grad_tensor_x = torch.zeros([1,1,256,319]).to(device)
    pc_grad_tensor_y = torch.zeros([1,1,255,320]).to(device)
    # pc_grad_tensor_x[:,:,:254,:318] = torch.from_numpy(pc_grad[:,:,0]).unsqueeze(0).unsqueeze(0).to(device).float()
    pc_grad_tensor_x[:,:,:254,:318] = pc_grad[0].float()
    pc_grad_tensor_y[:,:,:254,:318] = pc_grad[1].float()
    loss_smooth = get_smooth_loss_pc(tgt_depth[0], [pc_grad_tensor_x,pc_grad_tensor_y])
    # print('have grad')
        # stop
    # else:
    #     loss, mask = get_smooth_loss(tgt_depth[0], tgt_img)

    # for ref_depth, ref_img in zip(ref_depths, ref_imgs):
    #     loss += get_smooth_loss(ref_depth[0], ref_img)[0]

    return loss_smooth

def compute_loss_flow_smooth(optical_flows, img_pyramid):
    loss_list = []
    flow, img = optical_flows, img_pyramid
    error = cal_grad2_error(flow/20.0, img)
    loss_list.append(error[:,None])
    loss = torch.cat(loss_list, 1).sum(1)
    return loss

def cal_grad2_error(flow, img):
    img_grad_x, img_grad_y = gradients(img)
    w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
    w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

    dx, dy = gradients(flow)
    dx2, _ = gradients(dx)
    _, dy2 = gradients(dy)
    error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
    return error / 2.0

def gradients(img):
    dy = img[:,:,1:,:] - img[:,:,:-1,:]
    dx = img[:,:,:,1:] - img[:,:,:,:-1]
    return dx, dy

# def compute_3d_loss(tgt_depth, ref_depth, pose_inv, intrinsic):
#
#     Q_tgt = pixel2cam(tgt_depth.squeeze(1),intrinsic)
#     Q_ref = pixel2cam(ref_depth.squeeze(1),intrinsic)
#
#     pose_inv_mat = pose_vec2mat(pose_inv)
#     rot_inv, tr_inv = pose_inv_mat[:, :, :3], pose_inv_mat[:, :, -1:]
#     Q_ref_flat = Q_ref.reshape(tgt_depth.shape[0], 3, -1)
#
#     Q_hat_tgt = rot_inv@Q_ref_flat+tr_inv
#
#     Q_hat_tgt = Q_hat_tgt.reshape(Q_tgt.shape)
#
#     error = (Q_hat_tgt - Q_tgt).abs()
#
#     normalized_error = error/torch.max(error)
#
#     return normalized_error

# def compute_3d_loss(tgt_depth, img, ref_depth, pose_inv, intrinsic, ref_coord, mask_tgt_boundary):
#     b = ref_coord.shape[0]
#     h = ref_coord.shape[1]
#     w = ref_coord.shape[2]
#     i_range = torch.arange(0, h).view(h, 1).expand(
#         h, w).to(device)  # [H, W]
#     j_range = torch.arange(0, w).view(1, w).expand(
#         h, w).to(device)  # [H, W]
#     grid = torch.stack((j_range, i_range))  # [1, H, W, 2]
#     # print(grid.shape)
#     # print(grid[:, 5,6])
#     # print(grid.reshape(2, -1)[:, 5*320+6])
#     # stop
#     # print(ref_coord.shape) #1, 256,320,2
#
#     # mask_tgt_boundary = (ref_coord[:, :, :, 0] > 0) * (ref_coord[:, :, :, 0] < 320)\
#     #                     * (ref_coord[:, :, :, 1] > 0)*(ref_coord[:, :, :, 1] < 256)
#     # mask_tgt_boundary = valid_mask
#
#     # tgt_valid_points = mask_tgt_boundary*grid
#     # print(mask_tgt_boundary.shape)
#     # print(torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:,1])
#     # print(torch.sum(mask_tgt_boundary))
#
#
#     tgt_non_zero_idx = torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
#     ref_non_zero_idx = torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
#     ref_coord_select = torch.gather(ref_coord.reshape(b, -1, 2).transpose(1, 2), index=ref_non_zero_idx, dim=2)
#     # print(ref_coord_select)
#     # print(ref_coord_select.shape) #b,2 ,n
#     # print(torch.max(ref_coord_select[:, 0, :]))
#     # print(torch.max(ref_coord_select[:, 1, :]))
#     # stop
#     ref_select_idx = ref_coord_select[:, 1, :].type(torch.long) * w + ref_coord_select[:, 0, :].type(torch.long)
#     # print(ref_coord_select[:, 0, :])
#     # print(ref_select_idx) #b,n
#     # stop
#
#     Q_tgt = pixel2cam(tgt_depth.squeeze(1), intrinsic.inverse())
#     Q_ref = pixel2cam(ref_depth.squeeze(1), intrinsic.inverse()) #b, 3 ,h,w
#
#     # print(tgt_depth[0, 0, :, : ])
#     Q_tgt_copy = torch.zeros_like(Q_tgt)
#     Q_tgt_copy[0, 0, :, :] = (Q_tgt[0, 0, :, :]-torch.min(Q_tgt[0, 0, :, :]))/(torch.max(Q_tgt[0, 0, :, :])-torch.min(Q_tgt[0, 0, :, :]))
#     Q_tgt_copy[0, 1, :, :] = (Q_tgt[0, 1, :, :]-torch.min(Q_tgt[0, 1, :, :]))/(torch.max(Q_tgt[0, 1, :, :])-torch.min(Q_tgt[0, 1, :, :]))
#     Q_tgt_copy[0, 2, :, :] = (Q_tgt[0, 2, :, :]-torch.min(Q_tgt[0, 2, :, :]))/(torch.max(Q_tgt[0, 2, :, :])-torch.min(Q_tgt[0, 2, :, :]))
#     grad_pc_x = torch.mean(torch.abs(Q_tgt_copy[:, :, :, :-1] - Q_tgt_copy[:, :, :, 1:]), 1, keepdim=True)
#     grad_pc_y = torch.mean(torch.abs(Q_tgt_copy[:, :, :-1, :] - Q_tgt_copy[:, :, 1:, :]), 1, keepdim=True)
#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
#     grad_pc_x *= torch.exp(-grad_img_x)
#     grad_pc_y *= torch.exp(-grad_img_y)
#     # print(Q_tgt[0, 0, :, :].clamp(0,1)*255)
#     # print(Q_tgt[0, 1, :, :].clamp(0,1)*255)
#     # print(Q_tgt[0, 2, :, :].clamp(0,1)*255)
#     # print(grad_img_x)
#     pc_img_grad = ((grad_pc_x[:, :, :-1, :]+grad_pc_y[:, :, :, :-1])*100*255).cpu().detach().numpy().astype(np.uint8)
#     pc_img = (Q_tgt_copy*255).cpu().detach().numpy().astype(np.uint8)
#     img_c1 = np.transpose(pc_img[0], (1, 2, 0))[:, :, 0]
#     img_c2 = np.transpose(pc_img[0], (1, 2, 0))[:, :, 1]
#     img_c3 = np.transpose(pc_img[0], (1, 2, 0))[:, :, 2]
#     clahe1 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
#     cl1 = clahe1.apply(img_c1)
#     cl2 = clahe1.apply(img_c2)
#     cl3 = clahe1.apply(img_c3)
#
#     img_grad = np.transpose(pc_img_grad[0], (1, 2, 0))
#     cl4 = clahe1.apply(img_grad)
#     # print(cl4.shape)
#     # cv2.imshow('pc_img_grad_ori', img_grad)
#     # cv2.imshow('pc_img_grad', cl4)
#     # viewer = pcl.pcl_visualization.PCLVisualizering()
#     # viewer.SetBackgroundColor(0.0, 0.0, 0.5)
#     # cloud = pcl.PointCloud()
#     # pc_array = np.array(Q_tgt.reshape(3, -1).transpose(1, 0).detach().cpu().numpy(), dtype=np.float32)
#     # cloud.from_array(pc_array)
#     #
#     # viewer.AddPointCloud(cloud)
#     # flag = True
#     # while flag:
#     #     flag = not (viewer.WasStopped())
#     #     viewer.SpinOnce()
#     # cv2.waitKey(0)
#
#
#     Q_tgt_copy_equ = torch.from_numpy(np.stack([cl1, cl2, cl3])).unsqueeze(0).float().to(device)
#     loss_equ = torch.mean(((F.normalize(Q_tgt_copy_equ, dim=1)-F.normalize(Q_tgt_copy, dim=1))*mask_tgt_boundary)**2)
#     # grad_pc_x = torch.mean(torch.abs(Q_tgt_copy[:, :, :, :-1] - Q_tgt_copy[:, :, :, 1:]), 1, keepdim=True)
#     # grad_pc_y = torch.mean(torch.abs(Q_tgt_copy[:, :, :-1, :] - Q_tgt_copy[:, :, 1:, :]), 1, keepdim=True)
#     # print(grad_pc_x)
#     # pc_img_grad = ((grad_pc_x[:, :, :-1, :] + grad_pc_y[:, :, :, :-1])*30).cpu().detach().numpy().astype(
#     #     np.uint8)
#     # # print(pc_img)
#     # # cv2.imshow('pc_img', np.transpose(pc_img[0], (1, 2, 0)))
#     # cv2.imshow('pc_img_grad', np.transpose(pc_img_grad[0], (1, 2, 0)))
#     # # print(np.stack([cl1, cl2, cl3]).shape)
#     # # stop
#     # cv2.imshow('cl1', np.transpose(np.stack([cl1, cl2, cl3]), (1,2,0)))
#     # cv2.waitKey(0)
#     # stop
#     # print(Q_tgt.reshape(b, 3, -1).shape)
#     # print(tgt_non_zero_idx)
#     Q_tgt_select = torch.gather(Q_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx, dim=2)
#
#     # fit_plane(Q_tgt_select.reshape(3, -1))
#     # stop
#     # print(Q_tgt_select)
#
#     # print(tgt_non_zero_idx[0, :, 0])
#     # stop
#
#     pose_inv_mat = pose_vec2mat(pose_inv)
#     # rot, tr = pose_mat[:, :, :3], pose_mat[:, :, -1:]
#     rot_inv, tr_inv = pose_inv_mat[:, :, :3], pose_inv_mat[:, :, -1:]
#     # Q_tgt_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)
#     Q_ref_flat = Q_ref.reshape(tgt_depth.shape[0], 3, -1)
#     # print(pose_inv_mat.shape)
#     Q_hat_tgt = rot_inv@Q_ref_flat+tr_inv
#     # Q_hat_ref = rot@Q_ref_flat+tr
#     # print(Q_hat_tgt.shape)#16,3,81920
#     # print(Q_tgt.shape)#16,3,256,320
#     # Q_hat_tgt = torch.transpose(Q_hat_tgt,2,1)
#
#     Q_hat_tgt = Q_hat_tgt.reshape(Q_tgt.shape)
#     # print('Q_hat', Q_hat_tgt.reshape(b, 3, -1).shape)
#     # print(torch.max(ref_select_idx.unsqueeze(1).repeat(1, 3, 1)))
#     # print(tgt_non_zero_idx)
#     # print(Q_hat_tgt)
#     # Q_hat_tgt_select = torch.gather(Q_hat_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx, dim=2)
#     Q_hat_tgt_select = torch.gather(Q_hat_tgt.reshape(b, 3, -1), index=ref_select_idx.unsqueeze(1).repeat(1, 3, 1), dim=2)
#
#     rsme_loss = torch.mean(F.normalize(torch.abs(Q_hat_tgt_select.reshape(3, -1)-Q_tgt_select.reshape(3, -1)), dim=0))
#     # print(mask_tgt_boundary)
#     # stop
#     rmse = 0
#     res_T = 0
#     res_point_cloud = 0
#     # print(Q_hat_tgt.shape) 4,3,81920
#     # for b in range(Q_hat_tgt.shape[0]):
#     #     print('processing', b)
#     #     Q_hat_tgt_b = Q_hat_tgt[b].reshape(3, -1)
#     #     Q_tgt_b = Q_tgt[b]
#     #     Q_hat_tgt_icp = Q_hat_tgt_b.permute(1, 0)
#     #
#     #     Q_tgt_icp = Q_tgt_b.permute(1, 2, 0)
#     #     Q_tgt_icp = Q_tgt_icp.reshape(-1, 3)
#     #     icp = ICP6DoF(differentiable=True, solver_type='lm')
#     #     rigid_pose = icp(Q_tgt_icp[:100, :], Q_hat_tgt_icp[:100, :])
#     #
#     #     res_T = res_T+torch.mean(torch.abs(rigid_pose[0] - torch.eye(4).to(device)))
#     #
#     #     res_point_cloud += torch.mean(torch.abs(rigid_pose[0][:3, :3]@Q_tgt_icp.permute(1,0)+rigid_pose[0][:3, -1:] - Q_hat_tgt_icp.permute(1, 0)))
#
#     # return rsme_loss, grad_pc_x.mean() + grad_pc_y.mean()
#     return rsme_loss, loss_equ

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

def Sobel_function(img):
    src_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('src_gray',src_gray)
    Sobelx = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0)
    Sobely = cv2.Sobel(src_gray, cv2.CV_64F, 0, 1)
    Sobelx = cv2.convertScaleAbs(Sobelx)
    Sobely = cv2.convertScaleAbs(Sobely)
    Sobelxy = cv2.addWeighted(Sobelx, 0.5, Sobely, 0.5, 0)
    return Sobelxy, Sobelx, Sobely

def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)

def img_enhance_seg(img_grad):
    # img_grad = np.transpose(img_grad[0], (1, 2, 0))
    img_grad = 255 * np.power(img_grad / 255, 0.5)
    img_grad = np.around(img_grad)
    img_grad[img_grad > 255] = 255
    img_grad[img_grad < 50] = 0
    sigma_img = img_grad.astype(np.uint8)
    return sigma_img

def img_enhance(img_grad):
    # img_grad = np.transpose(img_grad[0], (1, 2, 0))
    img_grad = 255 * np.power(img_grad / 255, 0.5)
    img_grad = np.around(img_grad)
    img_grad[img_grad > 255] = 255
    img_grad[img_grad < 50] = 0

    sigma_img = img_grad.astype(np.uint8)
    return sigma_img

def img_enhance_final(img_grad):
    # img_grad = np.transpose(img_grad[0], (1, 2, 0))
    img_grad = 255 * np.power(img_grad / 255, 0.45)
    img_grad = np.around(img_grad)
    img_grad[img_grad > 255] = 255
    img_grad[img_grad< 50] = 0
    sigma_img = img_grad.astype(np.uint8)
    return sigma_img

def compute_3d_loss(tgt_depth, img, ref_depth, pose, intrinsic, p_match, global_pose, Q_last, src_seg_show_seg):

    # global Q_last

    # img_gray = 0.3 * img[:, 0] + 0.3 * img[:, 1] + 0.11 * img[:, 2] #1,h,w
    # img_normlize = (img_gray.reshape([-1])-torch.min(img_gray.reshape([-1])))/(torch.max(img_gray.reshape([-1]))-torch.min(img_gray.reshape([-1])))

    #here
    b = 1
    # h = ref_coord.shape[1]
    w = 320
    p_match_copy = p_match.clone()
    p_match_copy[:, :, :, 0] = (p_match[:, :, :, 0] + 1) * (320 - 1) / 2
    p_match_copy[:, :, :, 1] = (p_match[:, :, :, 1] + 1) * (256 - 1) / 2
    Q_tgt = pixel2cam(tgt_depth.squeeze(1), intrinsic.inverse())
    mask_tgt_boundary = (p_match_copy[:, :, :, 0] > 0) * (p_match_copy[:, :, :, 0] < 320) \
                        * (p_match_copy[:, :, :, 1] > 0) * (p_match_copy[:, :, :, 1] < 256)

    tgt_non_zero_idx = torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0)
    ref_non_zero_idx = torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    ref_coord_select = torch.gather(p_match_copy.reshape(b, -1, 2).transpose(1, 2), index=ref_non_zero_idx, dim=2)
    ref_select_idx = ref_coord_select[:, 1, :].type(torch.long) * w + ref_coord_select[:, 0, :].type(torch.long)

    if Q_last is None:
        # print(ref_depth)
        # print(intrinsic)
        Q_tgt = pixel2cam(ref_depth.squeeze(1), intrinsic.inverse()) #b, 3 ,h,w
        rsme_loss_global = 0
        # print(Q_last)
        # stop
    else:
        # Q_ref = pixel2cam(ref_depth.squeeze(1), intrinsic.inverse()) #b, 3 ,h,w
        # rot_inv_last, tr_inv_last = torch.from_numpy(global_pose).float().to(device).unsqueeze(0)[:, :3, :3], torch.from_numpy(global_pose).float().to(device).unsqueeze(0)[:, :3, -1:]
        #
        # Q_ref_flat = Q_last.reshape(tgt_depth.shape[0], 3, -1)
        #
        # Q_global_ref = rot_inv_last @ Q_ref_flat + tr_inv_last

        # pose_mat = pose_vec2mat(pose)
        # pose_mat_ref = pose
        # print('global_pose', global_pose)

        # pose_mat = pose_vec2mat(pose).squeeze(0)
        # pose_mat = pose.squeeze(0)

        #
        pose_mat = torch.cat([pose.squeeze(0), torch.tensor([[0, 0, 0, 1]], device=device)])
        global_pose = (torch.from_numpy(global_pose).float().to(device) @ pose_mat).unsqueeze(0)

        # rot_inv_ref, tr_inv_ref = pose[:, :, :3], pose[:, :, -1:]
        # print(global_pose.shape)
        rot_inv, tr_inv = global_pose[:, :3, :3], global_pose[:, :3, -1:]

        Q_tgt_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)

        Q_global_tgt = rot_inv@Q_tgt_flat+tr_inv
        # Q_ref_tgt = rot_inv_ref@Q_tgt_flat+tr_inv_ref

        # plane_pc_mask = plane_boundary*mask_tgt_boundary
        # tgt_non_zero_idx_forpc = torch.nonzero(plane_pc_mask.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0)
        # Q_tgt_select_global_forpc = torch.gather(Q_global_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx_forpc.repeat(1,3,1), dim=2)
        Q_tgt_select_global = torch.gather(Q_global_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx.repeat(1,3,1), dim=2)
        # if initial_norm[0] is not None:
        #     print('initial_norm', initial_norm[0])
        #     #calculate_norm = pc_array = np.array(cloud_tensor.transpose(1, 0).detach().cpu().numpy(), dtype=np.float32)
        #
        #
        #     # a_plane = initial_norm[0][0]
        #     # b_plane = initial_norm[0][1]
        #     # d_plane = initial_norm[0][3]
        #     # a_plane = -normals_tensor[0]
        #     # b_plane = -normals_tensor[1]
        #     # print(Q_tgt_select_global_forpc.shape)
        #     # x_train = Q_tgt_select_global_forpc[0,0]
        #     # y_train = Q_tgt_select_global_forpc[0,1]
        #     # z_train = Q_tgt_select_global_forpc[0,2]
        #     # print('in initial_norm_repeat')
        #     # print(initial_norm, initial_norm.shape)
        #     # print(Q_tgt_select_global.shape)
        #     # initial_norm_repeat = initial_norm[0].unsqueeze(1).repeat(1, Q_tgt_select_global_forpc.shape[2])
        #     # print('initial_norm_repeat',initial_norm_repeat)
        #     # Q_plane = torch.mean(torch.abs(torch.sum(Q_tgt_select_global_forpc[0] * initial_norm_repeat[:3,:], dim=0)-initial_norm_repeat[3,:]))
        #     # Q_plane = torch.mean(torch.abs(a_plane*x_train + b_plane*y_train +d_plane- z_train))
        #     # Q_plane = torch.mean(torch.abs(torch.var(z_train)))
        #     # print('Q_plane',Q_plane)
        #     # stop
        # else:
        #     Q_plane = 0


        # Q_tgt_select_ref = torch.gather(Q_ref_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx.repeat(1,3,1), dim=2)

        # Q_tgt_select_global_non_plane = torch.gather(Q_global_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx_non_plane.repeat(1, 3, 1), dim=2)
        # Q_tgt_select_ref_non_plane = torch.gather(Q_ref_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx_non_plane.repeat(1, 3, 1), dim=2)

        # tgt_depth_select = torch.gather(tgt_depth.reshape(b, 1, -1), index=tgt_non_zero_idx, dim=2)
        # ref_depth_select = torch.gather(ref_depth.reshape(b, 1, -1), index=ref_select_idx.unsqueeze(1), dim=2)
        # print(Q_hat_tgt.shape)

        # Q_hat_tgt = Q_hat_tgt.reshape(Q_tgt.shape)
        # print(Q_last)
        # print(ref_select_idx)
        # stop

        Q_ref_select_global = torch.gather(Q_last.reshape(b, 3, -1), index=ref_select_idx.unsqueeze(1).repeat(1, 3, 1), dim=2)
        # Q_ref_select_ref = torch.gather(Q_ref.reshape(b, 3, -1), index=ref_select_idx.unsqueeze(1).repeat(1, 3, 1), dim=2)

        # Q_ref_select_global_non_plane = torch.gather(Q_last.reshape(b, 3, -1), index=ref_select_idx_non_plane.unsqueeze(1).repeat(1, 3, 1), dim=2)
        # Q_ref_select_ref_non_plane = torch.gather(Q_ref.reshape(b, 3, -1), index=ref_select_idx_non_plane.unsqueeze(1).repeat(1, 3, 1), dim=2)

        # Q_tgt_reg = torch.gather(Q_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx.repeat(1,3,1), dim=2)
        # print(Q_ref_select[:,0])
        # print(Q_tgt_flat[:,0])
        # ref_depth_select = torch.gather(ref_depth.reshape(b, 1, -1), index=ref_select_idx.unsqueeze(1), dim=2)
        # tgt_depth_select = torch.gather(tgt_depth.reshape(b, 1, -1), index=tgt_non_zero_idx, dim=2)
        # loss_reg = torch.mean(torch.abs(ref_depth_select.reshape(1, -1).detach()-tgt_depth_select.reshape(1, -1)))
        # print(tgt_depth_select)
        # print(ref_depth_select)
        # print()

        rsme_loss_global = torch.mean(torch.abs(F.normalize(Q_ref_select_global.reshape(3, -1).detach()-Q_tgt_select_global.reshape(3, -1), dim=1)))
        # rsme_loss_global2 = torch.mean(torch.abs(F.normalize(Q_ref_select_global.reshape(3, -1).detach()-Q_tgt_select_global.reshape(3, -1), dim=0)))
        # weight = rsme_loss_global/rsme_loss_global2
        # rsme_loss_ref = torch.mean(torch.abs(F.normalize(Q_ref_select_ref.reshape(3, -1)-Q_tgt_select_ref.reshape(3, -1), dim=1)))

        # rsme_loss_global_non_plane = torch.mean(torch.abs(Q_ref_select_global_non_plane.reshape(3, -1).detach()-Q_tgt_select_global_non_plane.reshape(3, -1)))
        # rsme_loss_ref_non_plane = torch.mean(torch.abs(Q_ref_select_ref_non_plane.reshape(3, -1)-Q_tgt_select_ref_non_plane.reshape(3, -1)))
                    # + torch.mean((torch.abs(tgt_depth_select - ref_depth_select.detach())))
        # print('reg_depth', torch.mean(torch.abs(tgt_depth_select - ref_depth_select)))
                    # + torch.mean(torch.abs(Q_ref_select.reshape(3, -1)-Q_tgt_reg.reshape(3, -1)))
        # print('pc_loss', rsme_loss)
        # pc_img_grad = 0
        # +10 * Q_plane
        # return 1.5*Q_plane, Q_tgt.detach(), [pc_img_grad_x,pc_img_grad_y, pc_img_grad]
        # return rsme_loss_ref+rsme_loss_global, Q_tgt.detach(), [pc_img_grad_x,pc_img_grad_y, pc_img_grad]

    if src_seg_show_seg is not None:
        src = ((img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        src = np.transpose(src, (1, 2, 0)).astype(np.uint8)
        # _, src_show, _ = seg_only(src, 'image', 0)
        # src_seg_show_seg = initial_norm[1][0]
        # print('src_seg_show_seg', src_seg_show_seg.shape)
        sobel_seg, _, _ = Sobel_function(src_seg_show_seg)
        src_show_sobel_seg = img_enhance_seg(sobel_seg)
        Sobelxy_tgt, _, _ = Sobel_function(src)
        # cv2.imshow('sobel_seg', sobel_seg)
        # cv2.imshow('sobel_tgt', Sobelxy_tgt)
        #
        # cv2.waitKey(0)


        # print(img_grad[:,:,0])
        # stop
        # img_grad[:,:,1] = img_grad[:,:,1]/img_grad[:,:,1].max()*255
        # img_grad[:,:,2] = img_grad[:,:,2]/img_grad[:,:,2].max()*255
        # print('2',img_grad)
        # img_grad = np.expand_dims(img_grad, axis=0)
        # cv2.imshow('img_gray',img_grad)

        # _,seg_show,_ = seg_only(img_rainbow, 'image',0)
        # cv2.imshow('Sobel_tgt_grad', img_rainbow)
        # cv2.imshow('seg', seg_show)
        # cv2.waitKey(0)

        # i_range = torch.arange(0, h).view(h, 1).expand(
        #     h, w).to(device)  # [H, W]
        # j_range = torch.arange(0, w).view(1, w).expand(
        #     h, w).to(device)  # [H, W]
        # grid = torch.stack((j_range, i_range))  # [1, H, W, 2]
        # print(ref_coord[0, 100, 100, :])
        # print('match', (match[0, 100, 100, 0]+1)*(320-1)/2)
        # print('match', (match[0, 100, 100, 1]+1)*(256-1)/2)
        # print('before',p_match[:,:,:,0])


        # print(p_match[:,:,:,0])
        # print(p_match[:,:,:,1])
        # stop

        # mask_tgt_boundary = (ref_coord[:, :, :, 0] > 10) * (ref_coord[:, :, :, 0] < 310)\
        #                     * (ref_coord[:, :, :, 1] > 10)*(ref_coord[:, :, :, 1] < 246)

        # print(mask_tgt_boundary.shape)
        # mask_tgt_boundary_non_plane = (1-plane_boundary)*mask_tgt_boundary
        # mask_tgt_boundary = plane_boundary*mask_tgt_boundary
        # mask_tgt_boundary = valid_mask

        # tgt_valid_points = mask_tgt_boundary*grid
        # print(mask_tgt_boundary.shape)
        # print(torch.nonzero(mask_tgt_boundary.reshape(b, -1))[:,1])
        # print(torch.sum(mask_tgt_boundary))




        # tgt_non_zero_idx_non_plane = torch.nonzero(mask_tgt_boundary_non_plane.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0)
        # ref_non_zero_idx_non_plane = torch.nonzero(mask_tgt_boundary_non_plane.reshape(b, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
        # ref_coord_select_non_plane = torch.gather(p_match_copy.reshape(b, -1, 2).transpose(1, 2), index=ref_non_zero_idx_non_plane, dim=2)


        # ref_select_idx_non_plane = ref_coord_select_non_plane[:, 1, :].type(torch.long) * w + ref_coord_select_non_plane[:, 0, :].type(torch.long)

        # Q_tgt = pixel2cam(tgt_depth.squeeze(1), intrinsic.inverse())
        pc_img_grad_x = 0
        pc_img_grad_y = 0

        # 1022

        Q_tgt_copy = torch.zeros_like(Q_tgt)
        Q_tgt_copy[0, 0, :, :] = (Q_tgt[0, 0, :, :] - torch.min(Q_tgt[0, 0, :, :])) / (
                    torch.max(Q_tgt[0, 0, :, :]) - torch.min(Q_tgt[0, 0, :, :]))
        Q_tgt_copy[0, 1, :, :] = (Q_tgt[0, 1, :, :] - torch.min(Q_tgt[0, 1, :, :])) / (
                    torch.max(Q_tgt[0, 1, :, :]) - torch.min(Q_tgt[0, 1, :, :]))
        Q_tgt_copy[0, 2, :, :] = (Q_tgt[0, 2, :, :] - torch.min(Q_tgt[0, 2, :, :])) / (
                    torch.max(Q_tgt[0, 2, :, :]) - torch.min(Q_tgt[0, 2, :, :]))
        # print(Q_tgt_copy.shape) #1,3,h,w
        Q_tgt_copy_show = np.transpose(Q_tgt_copy[0].cpu().detach().numpy(), (1, 2, 0))
        # cv2.imshow('Q_tgt_copy_show', Q_tgt_copy_show)
        # cv2.waitKey(0)

        # geometric gradient
        grad_pc_x = torch.mean(torch.abs(Q_tgt_copy[:, :, :, :-1] - Q_tgt_copy[:, :, :, 1:]), 1, keepdim=True)
        grad_pc_y = torch.mean(torch.abs(Q_tgt_copy[:, :, :-1, :] - Q_tgt_copy[:, :, 1:, :]), 1, keepdim=True)
        # grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        # grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        # grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        # grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        # print(grad_pc_x.shape) #1,1,256,319
        # print(grad_pc_y.shape) #1,1,255,320

        # grad_pc_x = grad_img_x * torch.exp(-grad_pc_x)
        # grad_pc_x *= torch.exp(-grad_img_x)

        # grad_pc_x = grad_pc_x.repeat(1,3,1,1)
        # print('grad_shape',grad_pc_x.shape) # 1,1,256,319

        # grad_pc_y *= torch.exp(-grad_img_y)
        # grad_pc_y = grad_img_y * torch.exp(-grad_pc_y)

        # grad_pc_y = grad_pc_y.repeat(1,3,1,1)

        #
        # tgt_depth_normal = torch.zeros([320, 320]).to(device)
        # #
        # tgt_depth_normal[:256, :] = tgt_depth
        # tgt_depth_normal = tgt_depth_normal/torch.max(tgt_depth_normal)
        # depth_normal = depth2normal(tgt_depth_normal* 300) #3,318,318
        # # print(depth_normal[:,:254, :])
        #
        # normal_show = depth_normal.cpu().detach().numpy()*255
        #
        #
        # normal_image = cv2.cvtColor(np.transpose(normal_show, [1, 2, 0]), cv2.COLOR_BGR2RGB)
        # # _, seg_show, _ = seg_only(normal_image, 'image', 0)
        # # if online_iter >9 :
        # #     cv2.imshow('seg_show',seg_show)
        # #     cv2.waitKey(0)
        # # print('norm_map', normal_image)
        # normal_map = torch.zeros([1, 3, 256, 320]).to(device)
        # normal_map[0, :, 1:-1, 1:319] = torch.from_numpy(np.transpose(normal_image[:254, :, :], [2, 0, 1]))
        # normal_map = normal_map/torch.max(normal_map)


        # # normal_map = torch.max(normal_map, dim=1)
        # normal_map_grad_x = torch.mean(torch.abs(normal_map[:, :, :, :-1] - normal_map[:, :, :, 1:]), 1, keepdim=True)
        # normal_map_grad_y = torch.mean(torch.abs(normal_map[:, :, :-1, :] - normal_map[:, :, 1:, :]), 1, keepdim=True)
        # normal_map_grad_x *= torch.exp(-grad_img_x)
        # normal_map_grad_y *= torch.exp(-grad_img_y)
        # normal_map_np = normal_map[0].detach().cpu().numpy()
        # normal_map_np = np.transpose(normal_map_np, (1, 2, 0)).astype(np.uint8)
        # # print(normal_map_np[1:-1, 1:319,:])
        # normal_map_np = cv2.GaussianBlur(normal_map_np, (3, 3), 15)
        # cv2.imshow('normal_map', normal_map_np)
        # cv2.waitKey(0)

        # print(depth_normal.shape) #3,318,318
        # stop
        # print(grad_pc_x[:, :, 1:255, 1:319].shape)
        # print(depth_normal[:, 1:255, :].shape)
        # pc_img_grad = ((grad_pc_x[:, :, :-1, :] + grad_pc_y[:, :, :, :-1])*100*255).cpu().detach().numpy().astype(
        #     np.uint8)
        #+normal_map_grad_x[:, :, 1:255, 1:319]+normal_map_grad_y[:, :, 1:255, 1:319]
        # sobel_enhance = img_enhance(Sobelxy_tgt)
        # sobel_enhance = Sobelxy_tgt
        # cv2.imshow('sobel', sobel_enhance)
        # cv2.waitKey(0)
        # sobel_enhance = sobel_enhance>0
        Sobelxy_tgt_tensor = torch.from_numpy(Sobelxy_tgt).to(device).unsqueeze(0).unsqueeze(0)
        src_show_sobel_seg_tensor = torch.from_numpy(src_show_sobel_seg).to(device).unsqueeze(0).unsqueeze(0)

        # grad_pc_x_norm = grad_pc_x[:, :, 1:255, 1:319]/torch.max(grad_pc_x[:, :, 1:255, 1:319])
        # grad_pc_y_norm = grad_pc_y[:, :, 1:255, 1:319]/torch.max(grad_pc_y[:, :, 1:255, 1:319])
        # print('grad_pc_x_norm',grad_pc_x_norm)
        # print('grad_pc_y_norm',grad_pc_y_norm)
        # print('Sobelxy_tgt_tensor',Sobelxy_tgt_tensor)
        # pc_img_grad = (torch.exp(grad_pc_x[:, :, 1:255, 1:319] + grad_pc_y[:, :, 1:255, 1:319])*Sobelxy_tgt_tensor[:, :, 1:255, 1:319]).cpu().detach().numpy().astype(
        #     np.uint8)
        pc_second_grad_x = torch.mean(torch.abs(grad_pc_x[:, :, :, :-1] - grad_pc_x[:, :, :, 1:]), 1, keepdim=True)
        pc_second_grad_y = torch.mean(torch.abs(grad_pc_y[:, :, :-1, :] - grad_pc_y[:, :, 1:, :]), 1, keepdim=True)
        # print(pc_second_grad_x.shape) #1,1,256,318
        # print(pc_second_grad_y.shape) #1,1,254,320

        # pc_second_grad_x = pc_second_grad_x[0].cpu().detach().numpy()
        # pc_second_grad_y = pc_second_grad_y[0].cpu().detach().numpy()
        # pc_x_enhance = img_enhance(pc_second_grad_x)
        # pc_y_enhance = img_enhance(pc_second_grad_y)
        # print(pc_y_enhance.shape)
        # pc_second_grad_x_norm = (pc_second_grad_x[:, :, :254, :318]-torch.min(pc_second_grad_x[:, :, :254, :318]))/torch.max(pc_second_grad_x[:, :, :254, :318])
        # pc_second_grad_y_norm = (pc_second_grad_y[:, :, :254, :318]-torch.min(pc_second_grad_y[:, :, :254, :318]))/torch.max(pc_second_grad_y[:, :, :254, :318])
        # print(pc_second_grad_y_norm)
        # pc_img_grad = torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318])*Sobelxy_tgt_tensor[:, :, :254, :318].int()\
        # pc_img_grad = 255*torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318])*Sobelxy_tgt_tensor[:, :, :254, :318].int() \
        #         # *src_show_sobel_tensor[:, :, :254, :318]
        grad_geo = 100*100*(grad_pc_x[:, :, :254, :318]+grad_pc_y[:, :, :254, :318])
        # print(grad_geo.shape)
        grad_geo_show = np.transpose(grad_geo[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8)
        # grad_geo_show = img_enhance(grad_geo_show)
        # cv2.imshow('grad_geo_show', grad_geo_show)
        # cv2.waitKey(0)
        # stop
        # print('count', len(torch.nonzero(src_show_sobel_seg_tensor[:, :, :254, :318].reshape(-1))))
        # print('mean1',torch.sum(src_show_sobel_seg_tensor[:, :, :254, :318].float())/len(torch.nonzero(src_show_sobel_seg_tensor[:, :, :254, :318].reshape(-1))))
        tpt = torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318])
        mean1 = torch.sum(src_show_sobel_seg_tensor[:, :, :254, :318].float())/len(torch.nonzero(src_show_sobel_seg_tensor[:, :, :254, :318].reshape(-1)))
        mean2 = torch.sum(tpt)/len(torch.nonzero(tpt))
        # print('mean2', torch.sum(tpt)/len(torch.nonzero(tpt)))

        pc_img_grad = (torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318]))*Sobelxy_tgt_tensor[:, :, :254, :318].int() \
                    +2.5*mean2/mean1*src_show_sobel_seg_tensor[:, :, :254, :318]*Sobelxy_tgt_tensor[:, :, :254, :318].int()
        # print('pc_grad', pc_img_grad)
        # print('img', torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318]))
        # print(torch.max(Sobelxy_tgt_tensor[:, :, :254, :318]))
        # print('Sobelxy_tgt_tensor', Sobelxy_tgt_tensor[:, :, :254, :318]/torch.max(Sobelxy_tgt_tensor[:, :, :254, :318]))
        # print('src_show_sobel_tensor', src_show_sobel_tensor[:, :, :254, :318]/torch.max(src_show_sobel_tensor[:, :, :254, :318]))
        # print(torch.max(src_show_sobel_tensor[:, :, :254, :318]))
        # print(pc_img_grad)
        # stop

        # pc_img_grad = (normal_map[0,0,:254,:318]+normal_map[0,1,:254,:318]+normal_map[0,2,:254,:318]).unsqueeze(2)
        pc_img_grad_x = (torch.exp(grad_pc_x[:, :, :254, :318]) * Sobelxy_tgt_tensor[:, :,:254, :318].int() +2.5*mean2/mean1*src_show_sobel_seg_tensor[:, :, :254, :318])\
                        *src_show_sobel_seg_tensor[:, :, :254, :318]
        pc_img_grad_y = (torch.exp(grad_pc_y[:, :, :254, :318]) * Sobelxy_tgt_tensor[:, :,:254, :318].int() +2.5*mean2/mean1*src_show_sobel_seg_tensor[:, :, :254, :318]) \
                         *src_show_sobel_seg_tensor[:, :, :254, :318]
        # cv2.imshow('pc_img_ori', np.transpose(pc_img_grad[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8))
        # pc_img_grad_show = 255*((255-pc_img_grad)/torch.max(pc_img_grad)).repeat(1,3,1,1)*(normal_map[:,:,:254,:318] * 0.225 + 0.45)
        # pc_img_grad_numpy =np.transpose ((pc_second_grad_x[0,:, :254, :318] + pc_second_grad_y_norm[ 0,:, :254, :318]).cpu().detach().numpy(),(1,2,0))
        # # pc_img_grad_numpy = img_enhance(pc_img_grad_numpy*255)
        # # cv2.imshow('pc_img_grad_numpy',pc_img_grad_numpy)
        # pc_img_grad_numpy = pc_img_grad_numpy * np.expand_dims(sobel_enhance[:254, :318], axis=2)\
                            # *np.expand_dims(src_show_sobel[:254, :318], axis=2)

        # pc_img_grad = np.transpose(pc_img_grad[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8)

        pc_img_grad_enhance = img_enhance_final((np.transpose(pc_img_grad[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8)))
        # print('pc_img_grad_enhance', pc_img_grad_enhance)
        pc_img_grad = torch.from_numpy(pc_img_grad_enhance[:,:,0]).to(device)
    else:
        pc_img_grad = None
    # print(pc_img_grad.shape)
    # pc_img_grad = pc_img_grad[:,:,0]
    # pc_img_grad_show = np.transpose(pc_img_grad.cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8)
    # cv2.imshow('pc_img_grad', pc_img_grad_enhance)
    # cv2.waitKey(0)


    # pc_img_grad[pc_img_grad > 255] = 255
    # pc_img_grad[pc_img_grad < 100] = 0
    # pc_img_grad = cv2.bilateralFilter(pc_img_grad, 5, 200, 100)
    # pc_img_grad_show= np.transpose(pc_img_grad_show[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8)
    # _, seg_show, _ = seg_only(pc_img_grad_show, 'image', 0)
    #
    # cv2.imshow('pc_img_grad',pc_img_grad_show)
    # cv2.imshow('pc_seg', seg_show)
    # cv2.waitKey(0)


    # print(pc_img_grad/(100*255))

    # pc_img_grad =(pc_img_grad/255) * np.expand_dims(sobel_enhance[:254, :318], axis=2)
    # pc_img_grad = (normal_map[:, :, 1:-1, 1:319]).cpu().detach().numpy().astype(
    #     np.uint8)
    # print(pc_img_grad)
    # pc_img = (Q_tgt_copy * 255).cpu().detach().numpy().astype(np.uint8)

    # img_color = cv2.applyColorMap(cv2.convertScaleAbs(pc_img_grad, alpha=1), cv2.COLORMAP_RAINBOW)
    # print('3', img_color)

    return rsme_loss_global, Q_tgt.detach(), pc_img_grad

def calculate_pc_normal(cloud_tensor, a, b, i):
    # cloud = pcl.load('output/003784.pcd')
    # print('load output/003784.pcd')
    # print(cloud_tensor.transpose(1, 0).detach().cpu().numpy().shape)
    # print('cloud_tensor', cloud_tensor.shape)
    pc_array = np.array(cloud_tensor.transpose(1, 0).detach().cpu().numpy(), dtype=np.float32)

    cloud = pcl.PointCloud()
    cloud.from_array(pc_array)

    # filter_vox = cloud.make_voxel_grid_filter()
    # filter_vox.set_leaf_size(0.005, 0.005, 0.005)
    # cloud_filtered = filter_vox.filter()
    # print(cloud_filtered.size)

    seg = cloud.make_segmenter_normals(ksearch=50)  # 平面分割
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.1)
    seg.set_normal_distance_weight(0.3)
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    # print('indice',indices)
    # stop
    # print(-coefficients[0]/coefficients[2],-coefficients[1]/coefficients[2],-coefficients[2]/coefficients[2],-coefficients[3]/coefficients[2])
    q = cloud.extract(indices, negative=False)
    # print(q.size)
    # stop
    filter_stat = q.make_statistical_outlier_filter()
    filter_stat.set_mean_k(50)
    filter_stat.set_std_dev_mul_thresh(1.0)
    p = filter_stat.filter()

    # ne = cloud.make_NormalEstimation()
    ne = p.make_NormalEstimation()
    ne.set_KSearch(20)
    normals = ne.compute()
    # print('normals', normals[0])
    normals_arr = normals.to_array()[:, :3] #n,3
    normals_tensor = torch.from_numpy(normals_arr)
    # print(normals_tensor.shape)
    normals_tensor = normals_tensor/normals_tensor[:, 2].unsqueeze(1)

    # the normal is stored in the first 3 components (0-2), and the curvature is stored in component 3
    # if i == 10:
    #     viewer = pcl.pcl_visualization.PCLVisualizering()
    #     viewer.SetBackgroundColor(0.0, 0.0, 0.5)
    #     viewer.AddPointCloud(p)
    #     viewer.AddPointCloudNormals(p, normals, 10, 0.05, b'normals')
    #
    #     flag = True
    #     while flag:
    #         flag = not (viewer.WasStopped())
    #         viewer.SpinOnce()

    # stop
    return torch.tensor(indices), torch.median(normals_tensor, dim=0)[0], coefficients

    normals_tensor = torch.mean(normals_tensor, dim=0).unsqueeze(0)
    # print('average_norm', average_norm.shape)

    plane_norm = torch.tensor([a, b, -1]).unsqueeze(1)
    # print('plane_norm',plane_norm.shape)
    dot_multi = normals_tensor @ plane_norm
    # print(dot_multi.shape)
    division_norm = (normals_tensor[:, 0] **2 + normals_tensor[:, 1] **2 + normals_tensor[:, 2] **2)+\
                    (a.cpu()**2 + b.cpu()**2 +1)
    # print(division_norm.shape)

    diff_angle_cos = dot_multi/(torch.sqrt(division_norm).unsqueeze(1))
    print('cos', diff_angle_cos) #n,1

    # normals_mask = diff_angle_rad < 0.174 or diff_angle_rad > 2.96
    normals_mask = torch.abs(diff_angle_cos) > 0.90
    # print(normals_mask)
    # print(normals_mask.shape)
        # (diff_angle_cos < -0.98*torch.ones_like(diff_angle_cos))


    # the normal is stored in the first 3 components (0-2), and the curvature is stored in component 3
    # viewer = pcl.pcl_visualization.PCLVisualizering()
    # viewer.SetBackgroundColor(0.0, 0.0, 0.5)
    # viewer.AddPointCloud(p)
    # viewer.AddPointCloudNormals(p, normals, 10, 0.05, b'normals')
    #
    # flag = True
    # while flag:
    #     flag = not (viewer.WasStopped())
    #     viewer.SpinOnce()

    # stop



def fit_plane(use_poses, tgt_depth, pose_inv, intrinsic, mask_plane,iter_num, a, b ,d, i, flag):
    global x_train_proj, y_train_proj, z_train_proj
    global x_train_proj2, y_train_proj2, z_train_proj2
    global x_show, y_show, z_show

    if torch.sum(mask_plane) == 0:
        return 0
    # mask_plane = torch.ones_like(mask_plane)
    batch = tgt_depth.shape[0]
    h = tgt_depth.shape[1]
    w = tgt_depth.shape[2]

    Q_tgt = pixel2cam(tgt_depth.squeeze(1), intrinsic.inverse())
    # stop
    # mask_plane[:, :, -50:, :] = 1.
    tgt_non_zero_idx = torch.nonzero(mask_plane.reshape(batch, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
    # tgt_non_zero_idx = torch.nonzero(mask_plane_with_norm.reshape(batch, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)

    if use_poses:
        # print(pose_vec2mat(pose_inv).shape)
        # print(torch.tensor([[0., 0., 0., 1.]], device=device).shape)
        # print(torch.cat([pose_vec2mat(pose_inv), torch.tensor([[0., 0., 0., 1.]], device=device).unsqueeze(0)], dim=1))
        # stop
        # pose_inv_mat = torch.cat([pose_vec2mat(pose_inv), torch.tensor([[0., 0., 0., 1.]], device=device).unsqueeze(0)], dim=1).inverse()[:, :3, :]

        pose_inv_mat = pose_inv[:, :3, :]
        rot_inv, tr_inv = pose_inv_mat[:, :, :3], pose_inv_mat[:, :, -1:]
        # Q_tgt_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)
        Q_ref_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)
        Q_tgt= rot_inv @ Q_ref_flat + tr_inv

    # print('Q_tgt', Q_tgt.shape) # 1,3, 81920
    Q_tgt_grad_x = Q_tgt[:, 0, 1:]-Q_tgt[:, 0, :-1]
    Q_tgt_grad_y = Q_tgt[:, 1, 1:]-Q_tgt[:, 1, :-1]
    Q_tgt_grad_z = Q_tgt[:, 2, 1:]-Q_tgt[:, 2, :-1]
    # print(Q_tgt_grad_x.shape) # 1, 81920
    Q_tgt_grad_x_avrg = torch.mean(Q_tgt_grad_x[:,40000:50000],dim=1)
    Q_tgt_grad_y_avrg = torch.mean(Q_tgt_grad_y[:,40000:50000],dim=1)
    Q_tgt_grad_z_avrg = torch.mean(Q_tgt_grad_z[:,40000:50000],dim=1)
    loss_pc_gard_x = torch.mean(((Q_tgt_grad_x - Q_tgt_grad_x_avrg*torch.ones_like(Q_tgt_grad_x).to(device)))**2)
    loss_pc_gard_y = torch.mean(((Q_tgt_grad_y - Q_tgt_grad_y_avrg*torch.ones_like(Q_tgt_grad_x).to(device)))**2)
    loss_pc_gard_z = torch.mean(((Q_tgt_grad_z - Q_tgt_grad_z_avrg*torch.ones_like(Q_tgt_grad_x).to(device)))**2)
    # print(Q_tgt_grad_x - Q_tgt_grad_x_avrg*torch.ones_like(Q_tgt_grad_x))
    # print('Q_x_avrg', Q_tgt_grad_x_avrg)
    # print('loss_pc_grad', loss_pc_gard_x)
    # stop

    Q_tgt_select = torch.gather(Q_tgt.reshape(batch, 3, -1), index=tgt_non_zero_idx, dim=2)[0]
    # print('tgt', tgt_non_zero_idx.shape) #1,3,n
    # print(tgt_non_zero_idx[:, :, 887])
    point_center = torch.mean(Q_tgt_select.reshape(3, -1), dim=-1)
    # print('Q_tgt_select',Q_tgt_select.shape)
    normals_mask, normal_median, coeff = calculate_pc_normal(Q_tgt_select, a, b, i)
    if flag == 'plane_ini':
        return coeff
    # print('normal',normals_mask.shape) #n, 1
    # stop
    # print(tgt_non_zero_idx.shape) # 1,3, n
    # normals_indx_non0 = torch.nonzero(normals_mask)
    # print(normals_indx_non0.shape)#n,1
    # print(normals_indx_non0)
    # print('normal', normals_mask)
    final_indx_non0 = tgt_non_zero_idx[:, 0, normals_mask]
    # print(final_indx_non0) #1,n
    normal_mask_ori = torch.zeros_like(mask_plane.reshape(-1))
    normal_mask_ori[final_indx_non0] = 1.
    normal_mask_ori = normal_mask_ori.reshape(mask_plane.shape)

    tgt_non_zero_idx_norm = torch.nonzero(normal_mask_ori.reshape(batch, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
    Q_tgt_select_normal = torch.gather(Q_tgt.reshape(batch, 3, -1), index=tgt_non_zero_idx_norm, dim=2)[0]



    # mask_plane_with_norm = mask_plane * normals_mask.reshape(mask_plane.shape).to(device)
    # print(Q_tgt_select.shape)
    # x_train = Q_tgt_select[0][:5].detach()
    # y_train = Q_tgt_select[1][:5].detach()
    # z_train = Q_tgt_select[2][:5].detach()

    # x_train = Q_tgt_select_normal[0]
    # y_train = Q_tgt_select_normal[1]
    # z_train = Q_tgt_select_normal[2]

    x_train = Q_tgt_select[0]
    y_train = Q_tgt_select[1]
    z_train = Q_tgt_select[2]

    if i == 11:
        x_show = x_train
        y_show = y_train
        z_show = z_train

    division = a*a + b*b + 1

    # x_train = ((b * b + 1) * x_train_all - a * (b * y_train_all - z_train_all + d)) / division
    # y_train = ((a * a + 1) * y_train_all - b * (a * x_train_all - z_train_all + d)) / division
    # z_train = ((a * a + b * b) * z_train_all + (a * x_train_all + b * y_train_all + d)) / division

    if iter_num == 1 and flag == 'tgt':
        x_train_proj = ((b*b + 1)*x_train - a*(b*y_train - z_train + d))/division
        y_train_proj = ((a*a + 1)*y_train - b*(a*x_train -z_train + d))/division
        z_train_proj = ((a*a + b*b) * z_train + (a*x_train + b*y_train + d))/division

    # if i == 12 and iter_num == 10 and flag == 'next':
    #     x_train_proj2 = ((b * b + 1) * x_train - a * (b * y_train - z_train + d)) / division
    #     y_train_proj2 = ((a * a + 1) * y_train - b * (a * x_train - z_train + d)) / division
    #     z_train_proj2 = ((a * a + b * b) * z_train + (a * x_train + b * y_train + d)) / division
    if iter_num == 1 and flag == 'next':
        x_train_proj2 = ((b * b + 1) * x_train - a * (b * y_train - z_train + d)) / division
        y_train_proj2 = ((a * a + 1) * y_train - b * (a * x_train - z_train + d)) / division
        z_train_proj2 = ((a * a + b * b) * z_train + (a * x_train + b * y_train + d)) / division

        # print(x_train_proj,y_train_proj,z_train_proj)
        # stop
    # print(torch.max(x_train))
    # print(x_train)
    # print(y_train)
    # print(z_train)
    # stop

    # a = torch.tensor([1.1], requires_grad=True)
    # b = a * 2
    # print(b.requires_grad)
    # print(b)
    # b.backward()
    # print(a.grad)
    # stop
    # 0.957665205001831
    # 0.5875314474105835
    # 0.3561740219593048

    # a = torch.tensor([0.957665205001831], requires_grad=True, device=device)
    # b = torch.tensor([0.5875314474105835], requires_grad=True, device=device)
    # d = torch.tensor([0.3561740219593048], requires_grad=True, device=device)

    learing_rate = 0.1
    losss = []
    # print(a)
    # print(b)
    # print(d)
    loss_boundary = 0.
    # if i > 9 and flag == 'l':
    #     # print('boundary',boundary)
    #     # print('iter',i)
    #     # print('max_x',torch.max(x_train))
    #     loss_boundary = torch.sum(((x_train-1.1*torch.ones_like(x_train).to(device)).clamp(min=0))**2)
    #     # print('loss_boundary',loss_boundary)
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train) * y_train + d.expand_as(x_train)
    outlier_mask = torch.abs(predictions-z_train) > torch.median(torch.abs(predictions-z_train))
    # print('predition', a, b, d, predictions)
    # loss = torch.mean(outlier_mask*((predictions - z_train) ** 2))
    # x_curr = x_train_proj
    # y_curr = y_train_proj
    # z_curr = z_train_proj
    if flag == 'tgt':
        x_curr = x_train_proj.detach()
        y_curr = y_train_proj.detach()
        z_curr = z_train_proj.detach()
        # x_curr = x_train_proj
        # y_curr = y_train_proj
        # z_curr = z_train_proj
    elif flag == 'next':
        x_curr = x_train_proj2.detach()
        y_curr = y_train_proj2.detach()
        z_curr = z_train_proj2.detach()

    loss_proj = torch.abs(z_curr-z_train) + torch.abs(x_curr-x_train) + torch.abs(y_curr-y_train)
    # print(x_train_proj, y_train_proj, z_train_proj)
    # print('loss_proj', torch.mean(loss_proj))
    if torch.mean(loss_proj)<0.05:
        loss_proj_final = 0.
    else:
        loss_proj_final = torch.mean(loss_proj)
    loss = torch.mean(torch.abs(predictions - z_train))+2.5*loss_proj_final
           # + 50*(loss_pc_gard_x + loss_pc_gard_y + loss_pc_gard_z)\

           # 2.5*loss_proj_final
 # + 10*(loss_pc_gard_x + loss_pc_gard_y + loss_pc_gard_z)\
           #
           #

    # + 0.1 * loss_proj_final
    # loss = torch.mean(torch.abs(predictions - z_train)) + 2.5*torch.max(torch.cat([torch.mean(loss_proj).unsqueeze(0), torch.tensor([0.05], device=device)]))

    # return loss, mask_plane_with_norm
    # draw plane and pc


    return loss, normal_mask_ori, normal_median, coeff


def fit_plane_initial(use_poses, tgt_depth, pose_inv, intrinsic, mask_plane,i, iter_num, a, b ,d, line, a_l, b_l, d_l):
    global boundary
    # if i == 12:
    #     boundary = (0.5*(line[0][0] + line[0][1])).detach()
    # if i == 12:
    #     if line[0]<line[1]:
    #         min_x = line[0]
    #         min_y = line[2]
    #         max_x = line[1]
    #         max_y = line[3]
    #     else:
    #         min_x = line[1]
    #         min_y = line[3]
    #         max_x = line[0]
    #         max_y = line[2]
    #     mask_plane_cut = torch.zeros_like(mask_plane)
    #     mask_plane_cut[:,:,int(min_y),int(min_x)] = 1
    #     mask_plane = mask_plane_cut*mask_plane
        # d_predict =
    if torch.sum(mask_plane) == 0:
        return 0
    batch = tgt_depth.shape[0]
    h = tgt_depth.shape[1]
    w = tgt_depth.shape[2]
    tgt_non_zero_idx = torch.nonzero(mask_plane.reshape(batch, -1))[:, 1].unsqueeze(0).unsqueeze(0).repeat(1, 3, 1)
    Q_tgt = pixel2cam(tgt_depth.squeeze(1), intrinsic.inverse())

    if use_poses:
        # print(pose_vec2mat(pose_inv).shape)
        # print(torch.tensor([[0., 0., 0., 1.]], device=device).shape)
        # print(torch.cat([pose_vec2mat(pose_inv), torch.tensor([[0., 0., 0., 1.]], device=device).unsqueeze(0)], dim=1))
        # stop
        # pose_inv_mat = torch.cat([pose_vec2mat(pose_inv), torch.tensor([[0., 0., 0., 1.]], device=device).unsqueeze(0)], dim=1).inverse()[:, :3, :]

        pose_inv_mat = pose_inv[:, :3, :]
        rot_inv, tr_inv = pose_inv_mat[:, :, :3], pose_inv_mat[:, :, -1:]
        # Q_tgt_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)
        Q_ref_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)
        Q_tgt= rot_inv @ Q_ref_flat + tr_inv

    Q_tgt_select = torch.gather(Q_tgt.reshape(batch, 3, -1), index=tgt_non_zero_idx, dim=2)[0]
    # print(Q_tgt_select.shape)
    # x_train = Q_tgt_select[0][:5].detach()
    # y_train = Q_tgt_select[1][:5].detach()
    # z_train = Q_tgt_select[2][:5].detach()

    x_train = Q_tgt_select[0]
    y_train = Q_tgt_select[1]
    z_train = Q_tgt_select[2]
    # print('len_line', len(line))

    # if len(line) != 0:
    #     predictions = a.detach().expand_as(x_train) * x_train + b.detach().expand_as(x_train) * y_train + d.expand_as(x_train)
    # else:
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train) * y_train + d.expand_as(x_train)
    outlier_mask = torch.abs(predictions-z_train) > torch.median(torch.abs(predictions-z_train))

    # if i == 12 and iter_num % 1000 == 0:
    #     print('###############', d_predict, d, a,b)

    if i == 0 or i == 38 or i == 12:
        loss = torch.mean(torch.abs(predictions - z_train))
    else:
        d_min = torch.min((z_train - a.detach().expand_as(x_train) * x_train - b.detach().expand_as(x_train) * y_train))
        d_max = torch.max((z_train - a.detach().expand_as(x_train) * x_train - b.detach().expand_as(x_train) * y_train))
        # if i == 45:
        #     d_predict = 0.3 * (d_max + d_min)
        # else:
        # d_predict = 0.4 * (d_max + d_min)
        # loss = torch.abs(d - d_predict)
        loss = torch.mean(torch.abs(predictions - z_train))\
               # +torch.abs(d - d_predict)

    # if i == 12 and iter_num % 10000 == 0 :
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     plt.title("*plane", fontsize=30)
    #     # print(x_show.shape)
    #     # print(line[0].shape)
    #     # ax.scatter([line[0][0], line[0][1]], [line[1][0], line[1][1]], [line[2][0], line[2][1]])
    #     ax.scatter(x_show.cpu().data.numpy(), y_show.cpu().data.numpy(), z_show.cpu().data.numpy())
    #     # ax.scatter(x_train.cpu().data.numpy(), y_train.cpu().data.numpy(), z_train.cpu().data.numpy())
    #     # ax.scatter(x_test.data.numpy(), y_test.data.numpy(), z_test.data.numpy(), c='#DC143C')
    #     adata = float(a.data.cpu().numpy()[0])
    #     bdata = float(b.data.cpu().numpy()[0])
    #     ddata = float(d.data.cpu().numpy()[0])
    #
    #     a_l = float(a_l.data.cpu().numpy()[0])
    #     b_l = float(b_l.data.cpu().numpy()[0])
    #     d_l = float(d_l.data.cpu().numpy()[0])
    #     # print(adata)
    #     # print(bdata)
    #     # print(ddata)
    #     x = np.arange(-0.3, 0.3, 0.1)
    #     y = np.arange(-0.3, 0.3, 0.1)
    #
    #     x = np.arange(-0.5, 1, 0.1)
    #     # y = np.arange(-0.5, 1, 0.1)
    #     # 生成网格数据
    #     X, Y = np.meshgrid(x, y)
    #     Z = adata * X + bdata * Y + ddata
    #     ax.plot_surface(X, Y, Z,
    #                     color='g',
    #                     alpha=0.6
    #                     )
    #     # X_l, Y_l = np.meshgrid(x, y)
    #     # Z_l = a_l * X_l + b_l * Y_l + d_l
    #     # ax.plot_surface(X_l, Y_l, Z_l,
    #     #                 color='r',
    #     #                 alpha=0.6
    #     #                 )
    #     # ax.plot(line[0], line[1], line[2], color='r')
    #     # ax.plot([0, 0.1*adata], [0, 0.1*bdata], [0, -0.1], color='r')
    #     ax.set_xlabel('X Label')
    #     ax.set_ylabel('Y Label')
    #     ax.set_zlabel('Z Label')
    #     # print('draw',adata,bdata, ddata)
    #     # print('draw',a_l,b_l, d_l)
    #     # print(line[0], line[1], line[2])
    #     plt.show()

    return loss


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        # y1, y2 = int(0.1 * gt.size(1)), int(0.9 * gt.size(1))
        # x1, x2 = int(0.1 * gt.size(2)), int(0.9 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        # print('valid_gt', current_gt)
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]

