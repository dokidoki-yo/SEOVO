import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .feature_pyramid import FeaturePyramid
from .pwc_tf import PWC_tf
import torch
import torch.nn as nn

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

class Model_flow_test(nn.Module):
    def __init__(self):
        super(Model_flow_test, self).__init__()
        self.fpyramid = FeaturePyramid()
        self.pwc_model = PWC_tf()

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