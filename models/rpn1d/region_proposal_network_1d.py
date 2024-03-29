import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from datasets.chromatograms_dataset import ChromatogramsDataset
from models.modelzoo1d.depth_separable_conv_1d import DepthSeparableConv1d
from models.rpn1d.anchor_generation_1d import generate_anchors_1d
from models.rpn1d.anchor_target_layer_1d import anchor_target_layer_1d
from models.rpn1d.proposal_layer_1d import proposal_layer_1d

class RegionProposalNetwork1d(nn.Module):
    def __init__(
        self,
        model,
        load_backbone=False,
        backbone_path="",
        rpn_channels=16,
        rpn_kernel_size=3,
        pre_nms_topN=6000,
        nms_threshold=0.7,
        post_nms_topN=300,
        device='cpu',
        mode='train'):
        super(RegionProposalNetwork1d, self).__init__()
        self.device = device
        self.backbone = copy.deepcopy(model)

        if load_backbone:
            self.backbone.load_state_dict(
                torch.load(backbone_path, map_location=device).state_dict(),
                strict=False
            )

        self.anchors, self.num_anchors = generate_anchors_1d()
        self.anchors = torch.from_numpy(self.anchors).to(device)

        self.pre_nms_topN = pre_nms_topN
        self.nms_threshold = nms_threshold
        self.post_nms_topN = post_nms_topN

        self.rpn_net = nn.Sequential(
            DepthSeparableConv1d(
                out_channels[0],
                rpn_channels,
                kernel_size=rpn_kernel_size,
                padding=(rpn_kernel_size - 1) // 2,
                bias=True
            ),
            nn.ReLU(),
            nn.BatchNorm1d(rpn_channels)
        )

        self.rpn_cls_score_net = nn.Sequential(
            nn.Conv1d(
                rpn_channels,
                self.num_anchors,
                1
            )
        )

        self.rpn_bbox_pred_net = nn.Conv1d(
            rpn_channels,
            self.num_anchors * 2,
            1
        )

        self.mode = mode
    
    def proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, seq_len):
        return proposal_layer_1d(
            rpn_cls_prob,
            rpn_bbox_pred,
            seq_len,
            self.anchors,
            self.num_anchors,
            self.pre_nms_topN,
            self.nms_threshold,
            self.post_nms_topN
        )

    def anchor_target_layer(
        self,
        gt_boxes,
        seq_len):
        return anchor_target_layer_1d(
            gt_boxes,
            self.anchors,
            self.num_anchors,
            seq_len
        )

    def smooth_l1_loss(
        self,
        bbox_pred,
        bbox_targets,
        bbox_inside_weights,
        bbox_outside_weights,
        sigma=1.0,
        dim=[1]):
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1.0 / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box

        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)

        loss_box = loss_box.mean()

        return loss_box

    def rpn_loss(
        self,
        rpn_cls_prob,
        rpn_labels,
        rpn_bbox_pred,
        rpn_bbox_targets,
        rpn_bbox_inside_weights,
        rpn_bbox_outside_weights,
        sigma=1.0,
        dim=[1],
        loss_box_weight=1.0):
        # Class Loss
        rpn_cls_prob = rpn_cls_prob.view(-1)
        rpn_labels = rpn_labels.view(-1)
        rpn_select = (rpn_labels.data != -1).nonzero().view(-1)
        rpn_cls_prob = rpn_cls_prob.index_select(
            0, rpn_select.to(self.device))
        rpn_labels = rpn_labels.index_select(0, rpn_select).float().to(
            self.device)
        rpn_cross_entropy = F.binary_cross_entropy(rpn_cls_prob, rpn_labels)

        # Bounding Box Loss
        rpn_bbox_pred = rpn_bbox_pred.view(-1, 2).to(self.device)
        rpn_bbox_targets = rpn_bbox_targets.view(-1, 2).to(self.device)
        rpn_bbox_inside_weights = rpn_bbox_inside_weights.view(-1, 2).to(
            self.device)
        rpn_bbox_outside_weights = rpn_bbox_outside_weights.view(-1, 2).to(
            self.device)
        rpn_loss_box = self.smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma,
            dim=dim)

        rpn_loss = rpn_cross_entropy + loss_box_weight * rpn_loss_box

        return rpn_loss

    def forward(self, sequence, gt_boxes=None):
        if self.mode == 'train':
            assert sequence.size()[0] == 1, 'batch_size=1 train only'

        feature_map = self.backbone(sequence)

        rpn = self.rpn_net(feature_map)

        rpn_cls_score = self.rpn_cls_score_net(
            rpn
        )

        rpn_cls_prob = torch.sigmoid(rpn_cls_score)
        rpn_cls_prob = rpn_cls_prob.permute(0, 2, 1).contiguous()

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 1).contiguous()

        rpn_output = self.proposal_layer(
            rpn_cls_prob, rpn_bbox_pred, sequence.size(-1))

        output = rpn_output.data.cpu().numpy()

        if self.mode == 'test':
            return output
        elif self.mode == 'train':
            print('Top Outputs: {}'.format(output[:, 0, :]))

            (
                rpn_labels,
                rpn_bbox_targets,
                rpn_bbox_inside_weights,
                rpn_bbox_outside_weights
            ) = self.anchor_target_layer(gt_boxes, sequence.size(-1))

            rpn_labels = torch.from_numpy(rpn_labels).long()
            rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float()
            rpn_bbox_inside_weights = torch.from_numpy(
                rpn_bbox_inside_weights).float()
            rpn_bbox_outside_weights = torch.from_numpy(
                rpn_bbox_outside_weights).float()

            return self.rpn_loss(
                rpn_cls_prob,
                rpn_labels,
                rpn_bbox_pred,
                rpn_bbox_targets,
                rpn_bbox_inside_weights,
                rpn_bbox_outside_weights
            )
        else:
            raise NotImplementedError
