import numpy as np
import paddle
import paddle.nn as nn
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  #

    iw = paddle.min(paddle.unsqueeze(a[:, 2], axis=1), b[:, 2]) - paddle.max(paddle.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = paddle.min(paddle.unsqueeze(a[:, 3], axis=1), b[:, 3]) - paddle.max(paddle.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = paddle.Tensor.clip(iw, min=0)
    ih = paddle.Tensor.clip(ih, min=0)

    ua = paddle.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = paddle.Tensor.clip(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Layer):
    def forward(self, classifications, regressions, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                regression_losses.append(paddle.to_tensor(0).float().cuda())
                classification_losses.append(paddle.to_tensor(0).float().cuda())
                continue

            classification = paddle.Tensor.clip(classification, 1e-4, 1.0 - 1e-4)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            best_truth_overlap, best_truth_idx = paddle.max(IoU, axis=1)

            best_prior_overlap, best_prior_idx = paddle.max(IoU, axis=0)
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)

            for k in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[k]] = k

            # compute the label for classification
            targets = paddle.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[paddle.less_than(best_truth_overlap, 0.4), :] = 0

            positive_indices = paddle.greater_equal(best_truth_overlap, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[best_truth_idx, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[
                positive_indices, 4].long()] = 1

            # compute the loss for classification, 计算分类损失
            alpha_factor = paddle.ones(targets.shape).cuda() * alpha

            alpha_factor = paddle.where(paddle.equal(targets, 1.), alpha_factor,
                                        1. - alpha_factor)
            focal_weight = paddle.where(paddle.equal(targets, 1.), 1. - classification,
                                        classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

            bce = -(targets * paddle.log(classification) + (1.0 - targets) * paddle.log(
                1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = paddle.where(paddle.not_equal(targets, -1.0), cls_loss,
                                    paddle.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / paddle.Tensor.clip(num_positive_anchors.float(), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                assigned_ids = assigned_annotations[:, 5]
                assigned_annotations_next = paddle.zeros_like(assigned_annotations)
                reg_mask = paddle.ones(assigned_annotations.shape[0], 8).cuda()

                for m in range(assigned_annotations_next.shape[0]):
                    assigned_id = assigned_annotations[m, 5]
                    match_flag = False
                    for n in range(bbox_annotation_next.shape[0]):
                        if bbox_annotation_next[n, 5] == assigned_id:
                            match_flag = True
                            assigned_annotations_next[m, :] = bbox_annotation_next[n, :]
                            break

                    if match_flag is False:
                        reg_mask[m, 4:] = 0

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths_next = assigned_annotations_next[:, 2] - assigned_annotations_next[:, 0]
                gt_heights_next = assigned_annotations_next[:, 3] - assigned_annotations_next[:, 1]
                gt_ctr_x_next = assigned_annotations_next[:, 0] + 0.5 * gt_widths_next
                gt_ctr_y_next = assigned_annotations_next[:, 1] + 0.5 * gt_heights_next

                gt_widths = paddle.Tensor.clip(gt_widths, min=1)
                gt_heights = paddle.Tensor.clip(gt_heights, min=1)

                gt_widths_next = paddle.Tensor.clip(gt_widths_next, min=1)
                gt_heights_next = paddle.Tensor.clip(gt_heights_next, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = paddle.log(gt_widths / anchor_widths_pi)
                targets_dh = paddle.log(gt_heights / anchor_heights_pi)

                targets_dx_next = (gt_ctr_x_next - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy_next = (gt_ctr_y_next - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw_next = paddle.log(gt_widths_next / anchor_widths_pi)
                targets_dh_next = paddle.log(gt_heights_next / anchor_heights_pi)

                targets = paddle.stack((
                                       targets_dx, targets_dy, targets_dw, targets_dh, targets_dx_next, targets_dy_next,
                                       targets_dw_next, targets_dh_next))
                targets = targets.t()

                targets = targets / paddle.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]).cuda()
                # compute losses
                regression_diff = paddle.abs(
                    targets - regression[positive_indices, :]) * reg_mask

                regression_loss = paddle.where(
                    paddle.less_equal(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * paddle.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(paddle.to_tensor(0).float().cuda())

        return paddle.stack(classification_losses).mean(dim=0, keepdim=True), paddle.stack(regression_losses).mean(
            dim=0, keepdim=True)


class FocalLossReid(nn.Layer):
    def forward(self, classifications, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []

        for j in range(batch_size):

            classification = classifications[j, :, :]

            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                classification_losses.append(paddle.to_tensor(0).float().cuda())
                continue

            classification = paddle.Tensor.clip(classification, 1e-4, 1.0 - 1e-4)

            targets = paddle.ones(classification.shape) * -1
            targets = targets.cuda()

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            best_truth_overlap, best_truth_idx = paddle.max(IoU, axis=1)

            _, best_prior_idx = paddle.max(IoU, axis=0)
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)
            for k in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[k]] = k

            assigned_annotations = bbox_annotation[best_truth_idx, :]

            IoU_next = calc_iou(anchors[0, :, :], bbox_annotation_next[:, :4])
            best_truth_overlap_next, best_truth_idx_next = paddle.max(IoU_next, axis=1)

            _, best_prior_idx_next = paddle.max(IoU_next, axis=0)
            best_truth_overlap_next.index_fill_(0, best_prior_idx_next, 2)
            for k in range(best_prior_idx_next.size(0)):
                best_truth_idx_next[best_prior_idx_next[k]] = k

            assigned_annotations_next = bbox_annotation_next[best_truth_idx_next, :]

            reid_pos_thres = 0.5

            valid_samples = (paddle.greater_equal(best_truth_overlap, reid_pos_thres) & paddle.greater_equal(best_truth_overlap_next, 0.4)) | (paddle.greater_equal(best_truth_overlap_next, reid_pos_thres) & paddle.greater_equal(best_truth_overlap, 0.4))

            targets[valid_samples & paddle.equal(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 1

            targets[valid_samples & paddle.not_equal(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 0

            targets[paddle.less_than(best_truth_overlap, 0.4) | paddle.less_than(best_truth_overlap_next, 0.4), :] = 0

            targets[paddle.less_than(best_truth_overlap, reid_pos_thres) & paddle.greater_equal(best_truth_overlap, 0.4) & paddle.less_than(best_truth_overlap_next, reid_pos_thres) & paddle.greater_equal(best_truth_overlap_next, 0.4), :] = -1

            positive_indices = paddle.greater_equal(targets, 1)

            num_positive_anchors = positive_indices.sum()

            # compute losses
            alpha_factor = paddle.ones(targets.shape).cuda() * alpha

            alpha_factor = paddle.where(paddle.equal(targets, 1.), alpha_factor,
                                        1. - alpha_factor)
            focal_weight = paddle.where(paddle.equal(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * paddle.pow(focal_weight, gamma)

            bce = -(targets * paddle.log(classification) + (1.0 - targets) * paddle.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = paddle.where(paddle.not_equal(targets, -1.0), cls_loss, paddle.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum() / paddle.Tensor.clip(num_positive_anchors.float(), min=1.0))

        return paddle.stack(classification_losses).mean(dim=0, keepdim=True)
