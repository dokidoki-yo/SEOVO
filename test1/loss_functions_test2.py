from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from test1.inverse_warp_test1 import inverse_warp2, pose_vec2mat, pixel2cam, set_id_grid
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import pcl
import pcl.pcl_visualization

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

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

def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, matches, matches_inv, global_pose, Q_last):
    photo_loss = 0
    geometry_loss = 0
    pc_loss = 0
    match_loss = 0
    photo_masks = []

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth, pose, pose_inv, match, match_inv in zip(ref_imgs, ref_depths, poses, poses_inv, matches, matches_inv):
        for s in range(num_scales):
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

            photo_loss1, geometry_loss1, match_loss1, photo_mask_tgt, pc_loss1, Q_curr, fused_grad= compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, match, global_pose,Q_last)
            photo_loss2, geometry_loss2, match_loss2, photo_mask_ref, _, _, _= compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode, None, match_inv,global_pose,Q_last)
            photo_loss2, geometry_loss2, trian_loss2, pc_loss2, match_loss2 = 0,0,0,0,0
            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)
            pc_loss += pc_loss1
            match_loss += (match_loss1 + match_loss2)

            photo_masks.append(photo_mask_tgt)
            photo_masks.append(photo_mask_ref)

    return photo_loss, geometry_loss, match_loss, photo_masks, pc_loss, Q_curr, fused_grad

def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode, src_seg_show_seg, match, global_pose, Q_last):

    ref_img_warped, ref_coord_warped_norm, _, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    diff_match = (ref_coord_warped_norm - match).abs().clamp(0, 1).reshape(1, -1, 2).transpose(1,2).reshape(1, 2, 256,320)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask
    #
    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    photometric_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
    match_loss = mean_on_mask(diff_match, valid_mask)
    global_pc_loss, Q_curr, fused_grad = compute_3d_loss(tgt_depth, tgt_img, ref_depth, pose, intrinsic, match, global_pose, Q_last, src_seg_show_seg)
    mean_value = diff_img[:, 0, :, :] + diff_img[:, 1, :, :] + diff_img[:, 2, :, :]
    pho_mask0 = mean_value < 1.5*photometric_loss
    return photometric_loss, geometry_consistency_loss, match_loss, pho_mask0, global_pc_loss, Q_curr, fused_grad

# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value

def compute_smooth_loss(tgt_depth, tgt_img, pc_grad):

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

        grad_img_x = grad[0]
        grad_img_y = grad[1]

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        return grad_disp_x.mean() + grad_disp_y.mean()

    pc_grad_tensor_x = torch.zeros([1,1,256,319]).to(device)
    pc_grad_tensor_y = torch.zeros([1,1,255,320]).to(device)
    pc_grad_tensor_x[:,:,:254,:318] = pc_grad[0].float()
    pc_grad_tensor_y[:,:,:254,:318] = pc_grad[1].float()
    loss_smooth = get_smooth_loss_pc(tgt_depth[0], [pc_grad_tensor_x,pc_grad_tensor_y])
    return loss_smooth

def Sobel_function(img):
    src_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
    img_grad = 255 * np.power(img_grad / 255, 0.5)
    img_grad = np.around(img_grad)
    img_grad[img_grad > 255] = 255
    img_grad[img_grad < 50] = 0
    sigma_img = img_grad.astype(np.uint8)
    return sigma_img

def img_enhance_final(img_grad):
    img_grad = 255 * np.power(img_grad / 255, 0.45)
    img_grad = np.around(img_grad)
    img_grad[img_grad > 255] = 255
    img_grad[img_grad< 50] = 0
    sigma_img = img_grad.astype(np.uint8)
    return sigma_img

def compute_3d_loss(tgt_depth, img, ref_depth, pose, intrinsic, p_match, global_pose, Q_last, src_seg_show_seg):

    b = 1
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
        Q_tgt = pixel2cam(ref_depth.squeeze(1), intrinsic.inverse()) #b, 3 ,h,w
        rsme_loss_global = 0
    else:
        pose_mat = torch.cat([pose.squeeze(0), torch.tensor([[0, 0, 0, 1]], device=device)])
        global_pose = (torch.from_numpy(global_pose).float().to(device) @ pose_mat).unsqueeze(0)

        rot_inv, tr_inv = global_pose[:, :3, :3], global_pose[:, :3, -1:]

        Q_tgt_flat = Q_tgt.reshape(tgt_depth.shape[0], 3, -1)

        Q_global_tgt = rot_inv@Q_tgt_flat+tr_inv
        Q_tgt_select_global = torch.gather(Q_global_tgt.reshape(b, 3, -1), index=tgt_non_zero_idx.repeat(1,3,1), dim=2)

        Q_ref_select_global = torch.gather(Q_last.reshape(b, 3, -1), index=ref_select_idx.unsqueeze(1).repeat(1, 3, 1), dim=2)

        rsme_loss_global = torch.mean(torch.abs(F.normalize(Q_ref_select_global.reshape(3, -1).detach()-Q_tgt_select_global.reshape(3, -1), dim=1)))

    if src_seg_show_seg is not None:
        src = ((img[0].detach().cpu().numpy()) * 0.225 + 0.45) * 255
        src = np.transpose(src, (1, 2, 0)).astype(np.uint8)
        sobel_seg, _, _ = Sobel_function(src_seg_show_seg)
        src_show_sobel_seg = img_enhance_seg(sobel_seg)
        Sobelxy_tgt, _, _ = Sobel_function(src)

        Q_tgt_copy = torch.zeros_like(Q_tgt)
        Q_tgt_copy[0, 0, :, :] = (Q_tgt[0, 0, :, :] - torch.min(Q_tgt[0, 0, :, :])) / (
                    torch.max(Q_tgt[0, 0, :, :]) - torch.min(Q_tgt[0, 0, :, :]))
        Q_tgt_copy[0, 1, :, :] = (Q_tgt[0, 1, :, :] - torch.min(Q_tgt[0, 1, :, :])) / (
                    torch.max(Q_tgt[0, 1, :, :]) - torch.min(Q_tgt[0, 1, :, :]))
        Q_tgt_copy[0, 2, :, :] = (Q_tgt[0, 2, :, :] - torch.min(Q_tgt[0, 2, :, :])) / (
                    torch.max(Q_tgt[0, 2, :, :]) - torch.min(Q_tgt[0, 2, :, :]))

        Q_tgt_copy_show = np.transpose(Q_tgt_copy[0].cpu().detach().numpy(), (1, 2, 0))
        # cv2.imshow('Q_tgt_copy_show', Q_tgt_copy_show)
        # cv2.waitKey(0)

        # geometric gradient
        grad_pc_x = torch.mean(torch.abs(Q_tgt_copy[:, :, :, :-1] - Q_tgt_copy[:, :, :, 1:]), 1, keepdim=True)
        grad_pc_y = torch.mean(torch.abs(Q_tgt_copy[:, :, :-1, :] - Q_tgt_copy[:, :, 1:, :]), 1, keepdim=True)
        Sobelxy_tgt_tensor = torch.from_numpy(Sobelxy_tgt).to(device).unsqueeze(0).unsqueeze(0)
        src_show_sobel_seg_tensor = torch.from_numpy(src_show_sobel_seg).to(device).unsqueeze(0).unsqueeze(0)
        pc_second_grad_x = torch.mean(torch.abs(grad_pc_x[:, :, :, :-1] - grad_pc_x[:, :, :, 1:]), 1, keepdim=True)
        pc_second_grad_y = torch.mean(torch.abs(grad_pc_y[:, :, :-1, :] - grad_pc_y[:, :, 1:, :]), 1, keepdim=True)
        grad_geo = 100*100*(grad_pc_x[:, :, :254, :318]+grad_pc_y[:, :, :254, :318])
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
        pc_img_grad = (torch.exp(pc_second_grad_x[:, :, :254, :318] + pc_second_grad_y[:, :, :254, :318]))*Sobelxy_tgt_tensor[:, :, :254, :318].int() \
                    +2.5*mean2/mean1*src_show_sobel_seg_tensor[:, :, :254, :318]*Sobelxy_tgt_tensor[:, :, :254, :318].int()

        pc_img_grad_enhance = img_enhance_final((np.transpose(pc_img_grad[0].cpu().detach().numpy(),(1,2,0)).astype(np.uint8)))
        pc_img_grad = torch.from_numpy(pc_img_grad_enhance[:,:,0]).to(device)
    else:
        pc_img_grad = None
    # cv2.imshow('pc_img_grad', pc_img_grad_enhance)
    # cv2.waitKey(0)

    return rsme_loss_global, Q_tgt.detach(), pc_img_grad

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

