import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .feature_pyramid import FeaturePyramid
from .pwc_tf import PWC_tf
# from .pose_opt import pose_opt
# from pytorch_ssim import SSIM
from .net_utils import warp_flow
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import cv2
from .ransac_test import reduced_ransac_test
# from difficp import ICP6DoF
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# def SSIM(x, y):
#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2
#
#     mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
#     mu_y = nn.AvgPool2d(3, 1, padding=1)(y)
#
#     sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
#     sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
#     sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y
#
#     SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
#     SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
#
#     SSIM = SSIM_n / SSIM_d
#     return SSIM
#
# def transformerFwd(U,
#                    flo,
#                    out_size,
#                    name='SpatialTransformerFwd'):
#     """Forward Warping Layer described in
#     'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'
#
#     Parameters
#     ----------
#     U : float
#         The output of a convolutional net should have the
#         shape [num_batch, height, width, num_channels].
#     flo: float
#         The optical flow used for forward warping
#         having the shape of [num_batch, height, width, 2].
#     backprop: boolean
#         Indicates whether to back-propagate through forward warping layer
#     out_size: tuple of two ints
#         The size of the output of the network (height, width)
#     """
#
#     def _repeat(x, n_repeats):
#         rep = torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1,0)
#         x = x.view([-1,1]).mm(rep)
#         return x.view([-1]).int()
#
#     def _interpolate(im, x, y, out_size):
#         # constants
#         num_batch, height, width, channels = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
#         out_height = out_size[0]
#         out_width = out_size[1]
#         max_y = int(height - 1)
#         max_x = int(width - 1)
#
#         # scale indices from [-1, 1] to [0, width/height]
#         x = (x + 1.0) * (width - 1.0) / 2.0
#         y = (y + 1.0) * (height - 1.0) / 2.0
#
#         # do sampling
#         x0 = (torch.floor(x)).int()
#         x1 = x0 + 1
#         y0 = (torch.floor(y)).int()
#         y1 = y0 + 1
#
#         x0_c = torch.clamp(x0, 0, max_x)
#         x1_c = torch.clamp(x1, 0, max_x)
#         y0_c = torch.clamp(y0, 0, max_y)
#         y1_c = torch.clamp(y1, 0, max_y)
#
#         dim2 = width
#         dim1 = width * height
#         # base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.get_device())
#         base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(device)
#
#         base_y0 = base + y0_c * dim2
#         base_y1 = base + y1_c * dim2
#         idx_a = base_y0 + x0_c
#         idx_b = base_y1 + x0_c
#         idx_c = base_y0 + x1_c
#         idx_d = base_y1 + x1_c
#
#         # use indices to lookup pixels in the flat image and restore
#         # channels dim
#         im_flat = im.view([-1, channels])
#         im_flat = im_flat.float()
#
#         # and finally calculate interpolated values
#         x0_f = x0.float()
#         x1_f = x1.float()
#         y0_f = y0.float()
#         y1_f = y1.float()
#         wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
#         wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
#         wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
#         wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)
#
#         zerof = torch.zeros_like(wa)
#         wa = torch.where(
#             (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof)
#         wb = torch.where(
#             (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof)
#         wc = torch.where(
#             (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof)
#         wd = torch.where(
#             (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof)
#
#         zeros = torch.zeros(
#             size=[
#                 int(num_batch) * int(height) *
#                 int(width), int(channels)
#             ],
#             dtype=torch.float)
#         # output = zeros.to(im.get_device())
#         output = zeros.to(device)
#         output = output.scatter_add(dim=0, index=idx_a.long().unsqueeze(1).repeat(1,channels), src=im_flat * wa)
#         output = output.scatter_add(dim=0, index=idx_b.long().unsqueeze(1).repeat(1,channels), src=im_flat * wb)
#         output = output.scatter_add(dim=0, index=idx_c.long().unsqueeze(1).repeat(1,channels), src=im_flat * wc)
#         output = output.scatter_add(dim=0, index=idx_d.long().unsqueeze(1).repeat(1,channels), src=im_flat * wd)
#
#         return output
#
#     def _meshgrid(height, width):
#         # This should be equivalent to:
#         x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
#                                  np.linspace(-1, 1, height))
#         #  ones = np.ones(np.prod(x_t.shape))
#         #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
#         return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()
#
#     def _transform(flo, input_dim, out_size):
#         num_batch, height, width, num_channels = input_dim.shape[0:4]
#
#         # grid of (x_t, y_t, 1), eq (1) in ref [1]
#         height_f = float(height)
#         width_f = float(width)
#         out_height = out_size[0]
#         out_width = out_size[1]
#         x_s, y_s = _meshgrid(out_height, out_width)
#         # x_s = x_s.to(flo.get_device()).unsqueeze(0)
#         x_s = x_s.to(device).unsqueeze(0)
#         x_s = x_s.repeat([num_batch, 1, 1])
#
#         # y_s = y_s.to(flo.get_device()).unsqueeze(0)
#         y_s = y_s.to(device).unsqueeze(0)
#         y_s =y_s.repeat([num_batch, 1, 1])
#
#         x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
#         y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)
#
#         x_t_flat = x_t.view([-1])
#         y_t_flat = y_t.view([-1])
#
#         input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
#                                             out_size)
#
#         output = input_transformed.view([num_batch, out_height, out_width, num_channels])
#         return output
#
#     #out_size = int(out_size)
#     output = _transform(flo, U, out_size)
#     return output


class Model_flow_test(nn.Module):
    def __init__(self):
        super(Model_flow_test, self).__init__()
        self.fpyramid = FeaturePyramid()
        self.pwc_model = PWC_tf()
        
        # hyperparameters
        # self.dataset = 'nyuv2'
        # self.num_scales = 1
        # self.flow_consist_alpha = 3.0
        # self.flow_consist_beta = 0.05
        # self.inlier_thres = 0.1
        # self.rigid_thres = 1.0
        # self.ransac_points = 6000
        # self.depth_sample_ratio = 0.20
        # self.depth_match_num = 6000
        # self.filter = reduced_ransac_test(check_num=self.ransac_points, thres=self.inlier_thres, dataset=self.dataset)
        # self.transform = transforms.Compose(
        #     [
        #         transforms.Grayscale(num_output_channels=1),
        #         transforms.ToTensor()
        #     ]
        # )rigid_thres

    # def get_occlusion_mask_from_flow(self, tensor_size, flow):
    #     mask = torch.ones(tensor_size).to(device)
    #     h, w = mask.shape[2], mask.shape[3]
    #     occ_mask = transformerFwd(mask.permute(0,2,3,1), flow.permute(0,2,3,1), out_size=[h,w]).permute(0,3,1,2)
    #
    #     with torch.no_grad():
    #         occ_mask = torch.clamp(occ_mask, 0.0, 1.0)
    #     return occ_mask
    #
    # def get_flow_norm(self, flow, p=2):
    #     '''
    #     Inputs:
    #     flow (bs, 2, H, W)
    #     '''
    #     flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
    #     return flow_norm
    #
    # def get_visible_masks(self, optical_flows, optical_flows_rev):
    #     # get occlusion masks
    #     batch_size, _, img_h, img_w = optical_flows[0].shape
    #     img2_visible_masks, img1_visible_masks = [], []
    #     for s, (optical_flow, optical_flow_rev) in enumerate(zip(optical_flows, optical_flows_rev)):
    #         shape = [batch_size, 1, int(img_h / (2**s)), int(img_w / (2**s))]
    #         img2_visible_masks.append(self.get_occlusion_mask_from_flow(shape, optical_flow))
    #         img1_visible_masks.append(self.get_occlusion_mask_from_flow(shape, optical_flow_rev))
    #     return img2_visible_masks, img1_visible_masks
    #
    # def get_consistent_masks(self, optical_flows, optical_flows_rev):
    #     # get consist masks
    #     batch_size, _, img_h, img_w = optical_flows[0].shape
    #     img2_consis_masks, img1_consis_masks, fwd_flow_diff_pyramid, bwd_flow_diff_pyramid = [], [], [], []
    #     for s, (optical_flow, optical_flow_rev) in enumerate(zip(optical_flows, optical_flows_rev)):
    #
    #         bwd2fwd_flow = warp_flow(optical_flow_rev, optical_flow)
    #         fwd2bwd_flow = warp_flow(optical_flow, optical_flow_rev)
    #
    #         fwd_flow_diff = torch.abs(bwd2fwd_flow + optical_flow)
    #         fwd_flow_diff_pyramid.append(fwd_flow_diff)
    #         bwd_flow_diff = torch.abs(fwd2bwd_flow + optical_flow_rev)
    #         bwd_flow_diff_pyramid.append(bwd_flow_diff)
    #
    #         # flow consistency condition
    #         # bwd_consist_bound = torch.max(self.flow_consist_beta * self.get_flow_norm(optical_flow_rev), torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(optical_flow_rev.get_device()))
    #         bwd_consist_bound = torch.max(self.flow_consist_beta * self.get_flow_norm(optical_flow_rev), torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(device))
    #         # fwd_consist_bound = torch.max(self.flow_consist_beta * self.get_flow_norm(optical_flow), torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(optical_flow.get_device()))
    #         fwd_consist_bound = torch.max(self.flow_consist_beta * self.get_flow_norm(optical_flow), torch.from_numpy(np.array([self.flow_consist_alpha])).float().to(device))
    #         with torch.no_grad():
    #             noc_masks_img2 = (self.get_flow_norm(bwd_flow_diff) < bwd_consist_bound).float()
    #             noc_masks_img1 = (self.get_flow_norm(fwd_flow_diff) < fwd_consist_bound).float()
    #             img2_consis_masks.append(noc_masks_img2)
    #             img1_consis_masks.append(noc_masks_img1)
    #     return img2_consis_masks, img1_consis_masks, fwd_flow_diff_pyramid, bwd_flow_diff_pyramid
    #
    # def generate_img_pyramid(self, img, num_pyramid):
    #     img_h, img_w = img.shape[2], img.shape[3]
    #     img_pyramid = []
    #     for s in range(num_pyramid):
    #         img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
    #         img_pyramid.append(img_new)
    #     return img_pyramid
    #
    # def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
    #     img_warped_pyramid = []
    #     for img, flow in zip(img_pyramid, flow_pyramid):
    #         img_warped_pyramid.append(warp_flow(img, flow))
    #     return img_warped_pyramid
    #
    # def compute_loss_pixel(self, img_pyramid, img_warped_pyramid, occ_mask_list):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
    #         divider = occ_mask.mean((1,2,3)) #8,1,h,w
    #         # print('points_num', torch.sum(occ_mask.reshape((8,-1)),dim=-1))
    #         img_diff = torch.abs((img - img_warped)) * occ_mask.repeat(1,3,1,1)
    #         # img_diff_grad = img_diff[:,:,:254,:318] * grad_mask
    #         loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
    #         loss_list.append(loss_pixel[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1) # (B)
    #     return loss
    #
    # def compute_loss_photometric(self, img_pyramid, img_warped_pyramid, occ_mask_list):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
    #         divider = occ_mask.mean((1,2,3))
    #         img_diff = torch.abs((img - img_warped)) * occ_mask.repeat(1,3,1,1)
    #         loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
    #         loss_list.append(loss_pixel[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1) # (B)
    #     return loss
    #
    # def compute_loss_patch(self, optical_flows):
    #     batch_size, _, img_h, img_w = optical_flows[0].shape #[n,8,2,256,320]
    #     window_size = 3
    #
    #
    #
    # def compute_loss_ssim(self, img_pyramid, img_warped_pyramid, occ_mask_list):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
    #         divider = occ_mask.mean((1,2,3))
    #         occ_mask_pad = occ_mask.repeat(1,3,1,1)
    #         ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
    #         loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1,2,3))
    #         loss_ssim = loss_ssim / (divider + 1e-12)
    #         loss_list.append(loss_ssim[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1)
    #     # print('in',loss)
    #     return loss
    #
    # def gradients(self, img):
    #     dy = img[:,:,1:,:] - img[:,:,:-1,:]
    #     dx = img[:,:,:,1:] - img[:,:,:,:-1]
    #     return dx, dy
    #
    # def cal_grad2_error(self, flow, img):
    #     img_grad_x, img_grad_y = self.gradients(img)
    #     w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
    #     w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))
    #
    #     dx, dy = self.gradients(flow)
    #     dx2, _ = self.gradients(dx)
    #     _, dy2 = self.gradients(dy)
    #     error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
    #     return error / 2.0
    #
    # def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         flow, img = optical_flows[scale], img_pyramid[scale]
    #         error = self.cal_grad2_error(flow/20.0, img)
    #         loss_list.append(error[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1)
    #     return loss
    #
    # def compute_loss_flow_consis(self, fwd_flow_diff_pyramid, occ_mask_list):
    #     loss_list = []
    #     for scale in range(self.num_scales):
    #         fwd_flow_diff, occ_mask = fwd_flow_diff_pyramid[scale], occ_mask_list[scale]
    #         divider = occ_mask.mean((1,2,3))
    #         loss_consis = (fwd_flow_diff * occ_mask).mean((1,2,3))
    #         loss_consis = loss_consis / (divider + 1e-12)
    #         loss_list.append(loss_consis[:,None])
    #     loss = torch.cat(loss_list, 1).sum(1)
    #     return loss
    #
    # def inference_flow(self, img1, img2):
    #     img_hw = [img1.shape[2], img1.shape[3]]
    #     feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)
    #     optical_flow = self.pwc_model(feature_list_1, feature_list_2, img_hw)[0]
    #     return optical_flow
    #
    # def inference_corres(self, img1, img2):
    #     batch_size, img_h, img_w = img1.shape[0], img1.shape[2], img1.shape[3]
    #
    #     # get the optical flows and reverse optical flows for each pair of adjacent images
    #     feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)
    #     optical_flows = self.pwc_model(feature_list_1, feature_list_2, [img_h, img_w])
    #     optical_flows_rev = self.pwc_model(feature_list_2, feature_list_1, [img_h, img_w])
    #
    #     # get occlusion masks
    #     img2_visible_masks, img1_visible_masks = self.get_visible_masks(optical_flows, optical_flows_rev)
    #     # get consistent masks
    #     img2_consis_masks, img1_consis_masks, fwd_flow_diff_pyramid, bwd_flow_diff_pyramid = self.get_consistent_masks(optical_flows, optical_flows_rev)
    #     # get final valid masks
    #     img2_valid_masks, img1_valid_masks = [], []
    #     for i, (img2_visible_mask, img1_visible_mask, img2_consis_mask, img1_consis_mask) in enumerate(zip(img2_visible_masks, img1_visible_masks, img2_consis_masks, img1_consis_masks)):
    #         img2_valid_masks.append(img2_visible_mask * img2_consis_mask)
    #         img1_valid_masks.append(img1_visible_mask * img1_consis_mask)
    #
    #     return optical_flows[0], optical_flows_rev[0], img1_valid_masks[0], img2_valid_masks[0], fwd_flow_diff_pyramid[0], bwd_flow_diff_pyramid[0]
    #
    # def meshgrid(self, h, w):
    #     xx, yy = np.meshgrid(np.arange(0,w), np.arange(0,h))
    #     meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
    #     meshgrid = torch.from_numpy(meshgrid)
    #     return meshgrid
    #
    # def top_ratio_sample(self, match, mask, ratio):
    #     # match: [b, 4, -1] mask: [b, 1, -1]
    #     b, total_num = match.shape[0], match.shape[-1]
    #     scores, indices = torch.topk(mask, int(ratio * total_num), dim=-1)  # [B, 1, ratio*tnum]
    #     select_match = torch.gather(match.transpose(1, 2), index=indices.squeeze(1).unsqueeze(-1).repeat(1, 1, 4),
    #                                 dim=1).transpose(1, 2)  # [b, 4, ratio*tnum]
    #     return select_match, scores

    # def robust_rand_sample(self, match, mask, num):
    #     # match: [b, 4, -1] mask: [b, 1, -1]
    #     b, n = match.shape[0], match.shape[2]
    #     nonzeros_num = torch.min(torch.sum(mask > 0, dim=-1))  # []
    #
    #     if nonzeros_num.detach().cpu().numpy() == n:
    #         rand_int = torch.randint(0, n, [num])
    #         select_match = match[:, :, rand_int]
    #     else:
    #         # If there is zero score in match, sample the non-zero matches.
    #         num = np.minimum(nonzeros_num.detach().cpu().numpy(), num)
    #         select_idxs = []
    #         for i in range(b):
    #             nonzero_idx = torch.nonzero(mask[i, 0, :])  # [nonzero_num,1]
    #             # print(int(num))
    #             rand_int = torch.randint(0, nonzero_idx.shape[0], [int(num)])
    #             select_idx = nonzero_idx[rand_int, :]  # [num, 1]
    #             select_idxs.append(select_idx)
    #         select_idxs = torch.stack(select_idxs, 0)  # [b,num,1]
    #         select_match = torch.gather(match.transpose(1, 2), index=select_idxs.repeat(1, 1, 4), dim=1).transpose(1,
    #                                                                                                 2)  # [b, 4, num]
    #
    #     return select_match, num
    # def rt_from_fundamental_mat_nyu(self, fmat, K, depth_match):
    #     # F: [b, 3, 3] K: [b, 3, 3] depth_match: [b ,4, n]
    #     # verify_match = self.rand_sample(depth_match, 5000) # [b,4,100]
    #     verify_match = depth_match.transpose(1, 2).cpu().detach().numpy()
    #     b = fmat.shape[0]
    #     fmat_ = K.transpose(1, 2).bmm(fmat)
    #     essential_mat = fmat_.bmm(K)  # E = K^T * F *K
    #
    #     iden = torch.cat([torch.eye(3), torch.zeros([3, 1])], -1).unsqueeze(0).repeat(b, 1, 1).to(
    #         device)  # [b,3,4]
    #     P1 = K.bmm(iden)  # P1 with identity rotation and zero translation
    #     flags = []
    #     number_inliers = []
    #     P2 = []
    #     for i in range(b):
    #         cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'),
    #                                         verify_match[i, :, :2].astype('float64'), \
    #                                         verify_match[i, :, 2:].astype('float64'),
    #                                         cameraMatrix=K[i, :, :].cpu().detach().numpy().astype('float64'))
    #         # cnum, R, t, _ = cv2.recoverPose(essential_mat[i].cpu().detach().numpy().astype('float64'),
    #         #                                 verify_match[i, :, 2:].astype('float64'),
    #         #                                 verify_match[i, :, :2].astype('float64'),
    #         #                                 cameraMatrix=K[i, :, :].cpu().detach().numpy().astype('float64'))
    #         p2 = torch.from_numpy(np.concatenate([R, t*0.01], axis=-1)).float().to(device)
    #         P2.append(p2)
    #         if cnum > depth_match.shape[-1] / 7.0:
    #             flags.append(1)
    #         else:
    #             flags.append(0)
    #         number_inliers.append(cnum)
    #
    #     P2 = K.bmm(torch.stack(P2, axis=0))
    #
    #     return P1, P2, flags

    # def ray_angle_filter(self, match, P1, P2, return_angle=False):
    #     # match: [b, 4, n] P: [B, 3, 4]
    #     b, n = match.shape[0], match.shape[2]
    #     K = P1[:, :, :3]  # P1 with identity rotation and zero translation
    #     K_inv = torch.inverse(K)
    #     RT1 = K_inv.bmm(P1)  # [b, 3, 4]
    #     RT2 = K_inv.bmm(P2)
    #     ones = torch.ones([b, 1, n]).to(match.get_device())
    #     pts1 = torch.cat([match[:, :2, :], ones], 1)
    #     pts2 = torch.cat([match[:, 2:, :], ones], 1)
    #
    #     ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
    #     ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
    #     ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(RT1[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    #     ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
    #     ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
    #     ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(RT2[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    #
    #     # We compute the angle betwwen vertical line from ray1 origin to ray2 and ray1.
    #     p1p2 = (ray1_origin - ray2_origin).repeat(1, 1, n)
    #     verline = ray2_origin.repeat(1, 1, n) + torch.sum(p1p2 * ray2_dir, dim=1,
    #                                                       keepdim=True) * ray2_dir - ray1_origin.repeat(1, 1,
    #                                                                                                     n)  # [b,3,n]
    #     cosvalue = torch.sum(ray1_dir * verline, dim=1, keepdim=True) / \
    #                ((torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12) * (
    #                            torch.norm(verline, dim=1, keepdim=True, p=2) + 1e-12))  # [b,1,n]
    #
    #     mask = (cosvalue > 0.001).float()  # we drop out angles less than 1' [b,1,n]
    #     flag = 0
    #
    #     num = torch.min(torch.sum(mask, -1)).int()
    #     if num.cpu().detach().numpy() == 0:
    #         flag = 1
    #         filt_match = match[:, :, :100]
    #         if return_angle:
    #             return filt_match, flag, torch.zeros_like(mask).to(filt_match.get_device())
    #         else:
    #             return filt_match, flag
    #     nonzero_idx = []
    #     for i in range(b):
    #         idx = torch.nonzero(mask[i, 0, :])[:num]  # [num,1]
    #         nonzero_idx.append(idx)
    #     nonzero_idx = torch.stack(nonzero_idx, 0)  # [b,num,1]
    #     filt_match = torch.gather(match.transpose(1, 2), index=nonzero_idx.repeat(1, 1, 4), dim=1).transpose(1,
    #                                                                                                          2)  # [b,4,num]
    #     if return_angle:
    #         return filt_match, flag, mask
    #     else:
    #         return filt_match, flag

    # def midpoint_triangulate(self, match, K_inv, P1, P2):
    #     # match: [b, 4, num] P1: [b, 3, 4]
    #     # Match is in the image coordinates. P1, P2 is camera parameters. [B, 3, 4] match: [B, M, 4]
    #     b, n = match.shape[0], match.shape[2]
    #     RT1 = K_inv.bmm(P1)  # [b, 3, 4]
    #     RT2 = K_inv.bmm(P2)
    #
    #     ones = torch.ones([b, 1, n]).to(match.get_device())
    #     pts1 = torch.cat([match[:, :2, :], ones], 1)
    #     pts2 = torch.cat([match[:, 2:, :], ones], 1)
    #
    #     ray1_dir = (RT1[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts1)  # [b,3,n]
    #     ray1_dir = ray1_dir / (torch.norm(ray1_dir, dim=1, keepdim=True, p=2) + 1e-12)
    #     ray1_origin = (-1) * RT1[:, :, :3].transpose(1, 2).bmm(RT1[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    #     ray2_dir = (RT2[:, :, :3].transpose(1, 2)).bmm(K_inv).bmm(pts2)  # [b,3,n]
    #     ray2_dir = ray2_dir / (torch.norm(ray2_dir, dim=1, keepdim=True, p=2) + 1e-12)
    #     ray2_origin = (-1) * RT2[:, :, :3].transpose(1, 2).bmm(RT2[:, :, 3].unsqueeze(-1))  # [b, 3, 1]
    #
    #     dir_cross = torch.cross(ray1_dir, ray2_dir, dim=1)  # [b,3,n]
    #     denom = 1.0 / (torch.sum(dir_cross * dir_cross, dim=1, keepdim=True) + 1e-12)  # [b,1,n]
    #     origin_vec = (ray2_origin - ray1_origin).repeat(1, 1, n)  # [b,3,n]
    #     a1 = origin_vec.cross(ray2_dir, dim=1)  # [b,3,n]
    #     a1 = torch.sum(a1 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
    #     a2 = origin_vec.cross(ray1_dir, dim=1)  # [b,3,n]
    #     a2 = torch.sum(a2 * dir_cross, dim=1, keepdim=True) * denom  # [b,1,n]
    #     p1 = ray1_origin + a1 * ray1_dir
    #     p2 = ray2_origin + a2 * ray2_dir
    #     point = (p1 + p2) / 2.0  # [b,3,n]
    #     # Convert to homo coord to get consistent with other functions.
    #     point_homo = torch.cat([point, ones], dim=1).transpose(1, 2)  # [b,n,4]
    #     return point_homo

    # def reproject(self, P, point3d):
    #     # P: [b,3,4] point3d: [b,n,4]
    #     point2d = P.bmm(point3d.transpose(1,2)) # [b,4,n]
    #     point2d_coord = (point2d[:,:2,:] / (point2d[:,2,:].unsqueeze(1) + 1e-12)).transpose(1,2) # [b,n,2]
    #     point2d_depth = point2d[:,2,:].unsqueeze(1).transpose(1,2) # [b,n,1]
    #     return point2d_coord, point2d_depth

    # def filt_negative_depth(self, point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord):
    #     # Filter out the negative projection depth.
    #     # point2d_1_depth: [b, n, 1]
    #     b, n = point2d_1_depth.shape[0], point2d_1_depth.shape[1]
    #     mask = (point2d_1_depth > 0.01).float() * (point2d_2_depth > 0.01).float()
    #
    #     select_idxs = []
    #     flag = 0
    #     for i in range(b):
    #         if torch.sum(mask[i,:,0]) == n:
    #             idx = torch.arange(n).to(mask.get_device())
    #             # print('all negative')
    #         else:
    #             nonzero_idx = torch.nonzero(mask[i,:,0]).squeeze(1) # [k]
    #             if nonzero_idx.shape[0] < 0.1*n:
    #                 idx = torch.arange(n).to(mask.get_device())
    #                 flag = 1
    #             else:
    #                 res = torch.randint(0, nonzero_idx.shape[0], size=[n-nonzero_idx.shape[0]]).to(mask.get_device()) # [n-nz]
    #                 idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
    #         select_idxs.append(idx)
    #     select_idxs = torch.stack(select_idxs, dim=0) # [b,n]
    #     point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
    #     point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1) # [b,n,1]
    #     point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
    #     point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1,1,2), dim=1) # [b,n,2]
    #     return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

    # def filt_invalid_coord(self, point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h, max_w):
    #     # Filter out the negative projection depth. maybe out of the bound depth?
    #     # point2d_1_depth: [b, n, 1]
    #     b, n = point2d_1_coord.shape[0], point2d_1_coord.shape[1]
    #     max_coord = torch.Tensor([max_w, max_h]).to(point2d_1_coord.get_device())
    #     mask = (point2d_1_coord > 0).all(dim=-1, keepdim=True).float() * (point2d_2_coord > 0).all(dim=-1,
    #                                                                                                keepdim=True).float() * \
    #            (point2d_1_coord < max_coord).all(dim=-1, keepdim=True).float() * (point2d_2_coord < max_coord).all(
    #         dim=-1, keepdim=True).float()
    #
    #     flag = 0
    #     if torch.sum(1.0 - mask) == 0:
    #         return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag
    #
    #     select_idxs = []
    #     for i in range(b):
    #         if torch.sum(mask[i, :, 0]) == n:
    #             idx = torch.arange(n).to(mask.get_device())
    #         else:
    #             nonzero_idx = torch.nonzero(mask[i, :, 0]).squeeze(1)  # [k]
    #             if nonzero_idx.shape[0] < 0.1 * n:
    #                 idx = torch.arange(n).to(mask.get_device())
    #                 flag = 1
    #             else:
    #                 res = torch.randint(0, nonzero_idx.shape[0], size=[n - nonzero_idx.shape[0]]).to(mask.get_device())
    #                 idx = torch.cat([nonzero_idx, nonzero_idx[res]], 0)
    #         select_idxs.append(idx)
    #     select_idxs = torch.stack(select_idxs, dim=0)  # [b,n]
    #     point2d_1_depth = torch.gather(point2d_1_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    #     point2d_2_depth = torch.gather(point2d_2_depth, index=select_idxs.unsqueeze(-1), dim=1)  # [b,n,1]
    #     point2d_1_coord = torch.gather(point2d_1_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2),
    #                                    dim=1)  # [b,n,2]
    #     point2d_2_coord = torch.gather(point2d_2_coord, index=select_idxs.unsqueeze(-1).repeat(1, 1, 2),
    #                                    dim=1)  # [b,n,2]
    #     return point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag

    # def register_depth(self, depth_pred, coord_tri, depth_tri):
    #     # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
    #     batch, _, h, w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3]
    #     n = depth_tri.shape[1]
    #     coord_tri_nor = torch.stack(
    #         [2.0 * coord_tri[:, :, 0] / (w - 1.0) - 1.0, 2.0 * coord_tri[:, :, 1] / (h - 1.0) - 1.0], -1)
    #     # get the pred_depth of the tri_coord
    #     depth_inter = F.grid_sample(depth_pred, coord_tri_nor.view([batch, n, 1, 2]),
    #                                 padding_mode='reflection').squeeze(-1).transpose(1, 2)  # [b,n,1]
    #
    #     # Normalize
    #     scale = torch.median(depth_inter, 1)[0] / (torch.median(depth_tri, 1)[0] + 1e-12)
    #     scale = scale.detach()  # [b,1]
    #
    #     # scale_depth_inter = depth_inter / (scale.unsqueeze(-1) + 1e-12) #rescale pred_depth of tri_coord to tri_depth
    #     # scale_depth_pred = depth_pred / (scale.unsqueeze(-1).unsqueeze(-1) + 1e-12) #rescale pred_depth of whole_img to tri
    #     scale_depth_inter = depth_inter  #rescale pred_depth of tri_coord to tri_depth
    #     scale_depth_pred = depth_pred  #rescale pred_depth of whole_img to tri_depth
    #
    #     # affine adapt
    #     a, b = self.affine_adapt(scale_depth_inter, depth_tri, use_translation=False)
    #     # affine_depth_inter = a.unsqueeze(1) * scale_depth_inter + b.unsqueeze(1)  # [b,n,1]
    #     # affine_depth_pred = a.unsqueeze(-1).unsqueeze(-1) * scale_depth_pred + b.unsqueeze(-1).unsqueeze(
    #     #     -1)  # [b,1,h,w]
    #     affine_depth_inter = scale_depth_inter  # [b,n,1]
    #     affine_depth_pred = scale_depth_pred # [b,1,h,w]
    #     return affine_depth_pred, affine_depth_inter

    # def affine_adapt(self, depth1, depth2, use_translation=True, eps=1e-12):
    #     a_scale = self.scale_adapt(depth1, depth2, eps=eps)
    #     if not use_translation: # only fit the scale parameter
    #         return a_scale, torch.zeros_like(a_scale)
    #     else:
    #         with torch.no_grad():
    #             A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
    #             B = torch.sum(depth1 / (depth2 ** 2 + eps), dim=1) # [b,1]
    #             C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
    #             D = torch.sum(1.0 / (depth2 ** 2 + eps), dim=1) # [b,1]
    #             E = torch.sum(1.0 / (depth2 + eps), dim=1) # [b,1]
    #             a = (B*E - D*C) / (B*B - A*D + 1e-12)
    #             b = (B*C - A*E) / (B*B - A*D + 1e-12)
    #
    #             # check ill condition
    #             cond = (B*B - A*D)
    #             valid = (torch.abs(cond) > 1e-4).float()
    #             a = a * valid + a_scale * (1 - valid)
    #             b = b * valid
    #         return a, b

    # def scale_adapt(self, depth1, depth2, eps=1e-12):
    #     with torch.no_grad():
    #         A = torch.sum((depth1 ** 2) / (depth2 ** 2 + eps), dim=1) # [b,1]
    #         C = torch.sum(depth1 / (depth2 + eps), dim=1) # [b,1]
    #         a = C / (A + eps)
    #     return a

    # def compute_epipolar_loss(self, fmat, match, mask):
    #     # fmat: [b, 3, 3] match: [b, 4, h*w] mask: [b,1,h*w]
    #     num_batch = match.shape[0]
    #     match_num = match.shape[-1]
    #
    #     points1 = match[:, :2, :]
    #     points2 = match[:, 2:, :]
    #     ones = torch.ones(num_batch, 1, match_num).to(points1.get_device())
    #     points1 = torch.cat([points1, ones], 1)  # [b,3,n]
    #     points2 = torch.cat([points2, ones], 1).transpose(1, 2)  # [b,n,3]
    #
    #     # compute fundamental matrix loss
    #     fmat = fmat.unsqueeze(1)
    #     fmat_tiles = fmat.view([-1, 3, 3])
    #     epi_lines = fmat_tiles.bmm(points1)  # [b,3,n]
    #     # print(epi_lines.shape)
    #     # print(epi_lines.permute([0, 2, 1]).size()) #[b,n,3]
    #     # print((epi_lines.permute([0, 2, 1]) * points2).shape)#[b,n,3]
    #     dist_p2l = torch.abs((epi_lines.permute([0, 2, 1]) * points2).sum(-1, keepdim=True))  # [b,n,1]
    #     a = epi_lines[:, 0, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
    #     b = epi_lines[:, 1, :].unsqueeze(1).transpose(1, 2)  # [b,n,1]
    #     dist_div = torch.sqrt(a * a + b * b) + 1e-6
    #     dist_map = dist_p2l / dist_div  # [B, n, 1]
    #     loss = (dist_map * mask.transpose(1, 2)).mean([1, 2]) / mask.mean([1, 2])
    #     return loss, dist_map

    # def register_depth(self, depth_pred, coord_tri, depth_tri):
    #     # depth_pred: [b, 1, h, w] coord_tri: [b,n,2] depth_tri: [b,n,1]
    #     batch, _, h, w = depth_pred.shape[0], depth_pred.shape[1], depth_pred.shape[2], depth_pred.shape[3]
    #     n = depth_tri.shape[1]
    #     coord_tri_nor = torch.stack(
    #         [2.0 * coord_tri[:, :, 0] / (w - 1.0) - 1.0, 2.0 * coord_tri[:, :, 1] / (h - 1.0) - 1.0], -1)
    #     depth_inter = F.grid_sample(depth_pred, coord_tri_nor.view([batch, n, 1, 2]),
    #                                 padding_mode='reflection').squeeze(-1).transpose(1, 2)  # [b,n,1]
    #     return depth_pred, depth_inter
    #
    # def get_rigid_mask(self, dist_map):
    #     rigid_mask = (dist_map < self.rigid_thres).float()
    #     inlier_mask = (dist_map < self.inlier_thres).float()
    #     rigid_score = rigid_mask * 1.0 / (1.0 + dist_map)
    #     return rigid_mask, inlier_mask, rigid_score

    # def get_trian_loss(self, tri_depth, pred_tri_depth):
    #     # depth: [b,n,1]
    #     # loss = torch.pow(1.0 - pred_tri_depth / (tri_depth + 1e-12), 2).mean((1,2))
    #     # print(tri_depth.shape)
    #     loss = torch.sum(torch.abs(torch.cross(tri_depth.repeat(1, 1, 3), pred_tri_depth.repeat(1, 1, 3))), dim=1)
    #     # print(loss.shape)
    #
    #     return loss

    def forward(self, inputs, is_training = True):
        # if is_training:
        # images, K, K_inv, tgt_depths, ref_depths = inputs
        # else:
        images, K, K_inv = inputs

        assert (images.shape[1] == 3)
        img_h, img_w = int(images.shape[2] / 2), images.shape[3]
        img1, img2 = images[:,:,:img_h,:], images[:,:,img_h:,:]

        b = img1.shape[0]

        # get the optical flows and reverse optical flows for each pair of adjacent images
        feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)


        optical_flows = self.pwc_model(feature_list_1, feature_list_2, [img_h, img_w])
        optical_flows_rev = self.pwc_model(feature_list_2, feature_list_1, [img_h, img_w])

        # if not is_training:
        return optical_flows[0], optical_flows_rev[0]

        # grid = self.meshgrid(img_h, img_w).float().to(device).unsqueeze(0).repeat(b, 1, 1, 1)  # [b,2,h,w]
        #
        # fwd_corres = torch.cat([(grid[:, 0, :, :] + optical_flows[0][:, 0, :, :]).unsqueeze(1),
        #                         (grid[:, 1, :, :] + optical_flows[0][:, 1, :, :]).unsqueeze(1)], 1)
        # bwd_corres = torch.cat([(grid[:, 0, :, :] + optical_flows_rev[0][:, 0, :, :]).unsqueeze(1),
        #                         (grid[:, 1, :, :] + optical_flows_rev[0][:, 1, :, :]).unsqueeze(1)], 1)
        # fwd_match = torch.cat([grid, fwd_corres], 1).to(device)  # [b,4,h,w]
        # bwd_match = torch.cat([grid, bwd_corres], 1).to(device)
        #
        # img2_visible_masks, img1_visible_masks = self.get_visible_masks(optical_flows, optical_flows_rev)
        # img2_consis_masks, img1_consis_masks, fwd_flow_diff_pyramid, bwd_flow_diff_pyramid = self.get_consistent_masks(
        #     optical_flows, optical_flows_rev)
        # img2_valid_masks, img1_valid_masks = [], []
        #
        # for i, (img2_visible_mask, img1_visible_mask, img2_consis_mask, img1_consis_mask) in enumerate(
        #         zip(img2_visible_masks, img1_visible_masks, img2_consis_masks, img1_consis_masks)):
        #     if self.dataset == 'nyuv2' or self.dataset == 'SUN3D':
        #         img2_valid_masks.append(img2_visible_mask)
        #         img1_valid_masks.append(img1_visible_mask)
        #     else:
        #         img2_valid_masks.append(img2_visible_mask * img2_consis_mask)
        #         img1_valid_masks.append(img1_visible_mask * img1_consis_mask)
        # img1_score_mask = img1_valid_masks[0] * 1.0 / (0.1 + fwd_flow_diff_pyramid[0].mean(1).unsqueeze(1))
        # img2_score_mask = img2_valid_masks[0] * 1.0 / (0.1 + bwd_flow_diff_pyramid[0].mean(1).unsqueeze(1))
        #
        # # boundary_mask = (fwd_corres[:, 0, :, :] > 10) * (fwd_corres[:, 0, :, :] < 310) * (
        # #         fwd_corres[:, 1, :, :] > 10) * (fwd_corres[:, 1, :, :] < 250)
        # # boundary_mask = torch.ones_like(boundary_mask) * boundary_mask.int()
        # #
        # # boundary_mask_inv = (bwd_corres[:, 0, :, :] > 10) * (bwd_corres[:, 0, :, :] < 310) * (
        # #         bwd_corres[:, 1, :, :] > 10) * (bwd_corres[:, 1, :, :] < 250)
        # # boundary_mask_inv = torch.ones_like(boundary_mask_inv) * boundary_mask_inv.int()
        #
        # F_final_1, F1_match = self.filter(fwd_match, img1_score_mask)
        # F_final_2, F2_match = self.filter(bwd_match, img2_score_mask)
        #
        # P1_K, P2_K, _ = self.rt_from_fundamental_mat_nyu(F_final_1.detach().to(device), K,
        #                                                          F1_match.to(device))
        # P1_K_inv, P2_K_inv, _ = self.rt_from_fundamental_mat_nyu(F_final_2.detach().to(device), K,
        #                                                                      F2_match.to(device))
        #
        #
        # # _______________________0505, add triangulation loss
        #
        # _, dist_map_1 = self.compute_epipolar_loss(F_final_1.to(device), fwd_match.view([b, 4, -1]),
        #                                            img1_valid_masks[0].view([b, 1, -1]))
        # dist_map_1 = dist_map_1.view([b, img_h, img_w, 1])
        #
        # # Compute geo loss for regularize correspondence.
        # rigid_mask_1, inlier_mask_1, rigid_score_1 = self.get_rigid_mask(dist_map_1)
        #
        # # We only use rigid mask to filter out the moving objects for computing geo loss.
        # geo_loss = (dist_map_1 * (rigid_mask_1 - inlier_mask_1)).mean((1, 2, 3)) / \
        #            (rigid_mask_1 - inlier_mask_1).mean((1, 2, 3))
        #
        # img1_rigid_score = rigid_score_1.permute(0, 3, 1, 2)
        # img1_depth_mask = img1_rigid_score * img1_score_mask
        # top_ratio_match, top_ratio_mask = self.top_ratio_sample(fwd_match.view([b, 4, -1]),
        #                                                         img1_depth_mask.view([b, 1, -1]),
        #                                                         ratio=self.depth_sample_ratio)  # [b, 4, ratio*h*w]
        # depth_match, depth_match_num = self.robust_rand_sample(top_ratio_match, top_ratio_mask,
        #                                                        num=self.depth_match_num)
        # # Get triangulated points
        # filt_depth_match, flag1 = self.ray_angle_filter(depth_match, P1_K, P2_K, return_angle=False)  # [b, 4, filt_num]
        # # print('filt_depth_match',filt_depth_match)
        # point3d_1 = self.midpoint_triangulate(filt_depth_match, K_inv, P1_K, P2_K)
        # # point3d_1 = self.midpoint_triangulate(depth_match, K_inv, P1, P2)
        # point2d_1_coord, point2d_1_depth = self.reproject(P1_K, point3d_1)  # [b,n,2], [b,n,1]
        # point2d_2_coord, point2d_2_depth = self.reproject(P2_K, point3d_1)
        #
        # # Filter out some invalid triangulation results to stablize training.
        # point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag2 = self.filt_negative_depth(
        #     point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord)
        #
        # point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, flag3 = self.filt_invalid_coord(
        #     point2d_1_depth, point2d_2_depth, point2d_1_coord, point2d_2_coord, max_h=img_h, max_w=img_w)
        #
        # rescaled_pred1, inter_pred1 = self.register_depth(torch.cat([tgt_depths[0], tgt_depths[0]], dim=0), point2d_1_coord, point2d_1_depth)
        # rescaled_pred2, inter_pred2 = self.register_depth(torch.cat([ref_depths[0][0], ref_depths[1][0]], dim=0), point2d_2_coord, point2d_2_depth)
        #
        # # print(rescaled_pred1.shape) # b,1, h,w
        # # print(inter_pred1.shape) #b,n,1
        #
        # pt_depth_loss = self.get_trian_loss(point2d_1_depth, inter_pred1) + self.get_trian_loss(point2d_2_depth,
        #                                                                                         inter_pred2)
        #
        # return pt_depth_loss