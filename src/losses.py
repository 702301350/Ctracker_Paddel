import sys
import paddle_aux
import paddle
import numpy as np
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = paddle_aux.min(paddle.unsqueeze(x=a[:, 2], axis=1), b[:, 2]
        ) - paddle_aux.max(paddle.unsqueeze(x=a[:, 0], axis=1), b[:, 0])
    ih = paddle_aux.min(paddle.unsqueeze(x=a[:, 3], axis=1), b[:, 3]
        ) - paddle_aux.max(paddle.unsqueeze(x=a[:, 1], axis=1), b[:, 1])
    iw = paddle.clip(x=iw, min=0)
    ih = paddle.clip(x=ih, min=0)
    ua = paddle.unsqueeze(x=(a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1
        ) + area - iw * ih
    ua = paddle.clip(x=ua, min=1e-08)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


class FocalLoss(paddle.nn.Layer):

    def forward(self, classifications, regressions, anchors, annotations1,
        annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = tuple(classifications.shape)[0]
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
            bbox_annotation_next = bbox_annotation_next[
                bbox_annotation_next[:, 4] != -1]
            if tuple(bbox_annotation.shape)[0] == 0:
                regression_losses.append(paddle.to_tensor(data=0).astype(
                    dtype='float32').cuda(blocking=True))
                classification_losses.append(paddle.to_tensor(data=0).
                    astype(dtype='float32').cuda(blocking=True))
                continue
            classification = paddle.clip(x=classification, min=0.0001, max=
                1.0 - 0.0001)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            best_truth_overlap, best_truth_idx = paddle.max(x=IoU, axis=1
                ), paddle.argmax(x=IoU, axis=1)
            best_prior_overlap, best_prior_idx = paddle.max(x=IoU, axis=0
                ), paddle.argmax(x=IoU, axis=0)
            best_truth_overlap.index_fill_(axis=0, index=best_prior_idx,
                value=2)
            for j in range(best_prior_idx.shape[0]):
                best_truth_idx[best_prior_idx[j]] = j
            targets = paddle.ones(shape=tuple(classification.shape)) * -1
            targets = targets.cuda(blocking=True)
            targets[paddle.less_than(x=best_truth_overlap, y=paddle.
                to_tensor(0.4)), :] = 0
            positive_indices = paddle.greater_equal(x=best_truth_overlap, y
                =paddle.to_tensor(0.5))
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[best_truth_idx, :]
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices,
                4].astype(dtype='int64')] = 1
            alpha_factor = paddle.ones(shape=tuple(targets.shape)).cuda(
                blocking=True) * alpha
            alpha_factor = paddle.where(condition=paddle.equal(x=targets, y
                =1.0), x=alpha_factor, y=1.0 - alpha_factor)
            focal_weight = paddle.where(condition=paddle.equal(x=targets, y
                =1.0), x=1.0 - classification, y=classification)
            focal_weight = alpha_factor * paddle.pow(x=focal_weight, y=gamma)
            bce = -(targets * paddle.log(x=classification) + (1.0 - targets
                ) * paddle.log(x=1.0 - classification))
            cls_loss = focal_weight * bce
            cls_loss = paddle.where(condition=paddle.not_equal(x=targets, y
                =paddle.to_tensor(-1.0)), x=cls_loss, y=paddle.zeros(shape=
                tuple(cls_loss.shape)).cuda(blocking=True))
            classification_losses.append(cls_loss.sum() / paddle.clip(x=
                num_positive_anchors.astype(dtype='float32'), min=1.0))
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :
                    ]
                assigned_ids = assigned_annotations[:, 5]
                assigned_annotations_next = paddle.zeros_like(x=
                    assigned_annotations)
                reg_mask = paddle.ones(shape=[tuple(assigned_annotations.
                    shape)[0], 8]).cuda(blocking=True)
                for m in range(tuple(assigned_annotations_next.shape)[0]):
                    assigned_id = assigned_annotations[m, 5]
                    match_flag = False
                    for n in range(tuple(bbox_annotation_next.shape)[0]):
                        if bbox_annotation_next[n, 5] == assigned_id:
                            match_flag = True
                            assigned_annotations_next[m, :
                                ] = bbox_annotation_next[n, :]
                            break
                    if match_flag == False:
                        reg_mask[m, 4:] = 0
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[
                    :, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[
                    :, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths_next = assigned_annotations_next[:, 2
                    ] - assigned_annotations_next[:, 0]
                gt_heights_next = assigned_annotations_next[:, 3
                    ] - assigned_annotations_next[:, 1]
                gt_ctr_x_next = assigned_annotations_next[:, 0
                    ] + 0.5 * gt_widths_next
                gt_ctr_y_next = assigned_annotations_next[:, 1
                    ] + 0.5 * gt_heights_next
                gt_widths = paddle.clip(x=gt_widths, min=1)
                gt_heights = paddle.clip(x=gt_heights, min=1)
                gt_widths_next = paddle.clip(x=gt_widths_next, min=1)
                gt_heights_next = paddle.clip(x=gt_heights_next, min=1)
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = paddle.log(x=gt_widths / anchor_widths_pi)
                targets_dh = paddle.log(x=gt_heights / anchor_heights_pi)
                targets_dx_next = (gt_ctr_x_next - anchor_ctr_x_pi
                    ) / anchor_widths_pi
                targets_dy_next = (gt_ctr_y_next - anchor_ctr_y_pi
                    ) / anchor_heights_pi
                targets_dw_next = paddle.log(x=gt_widths_next /
                    anchor_widths_pi)
                targets_dh_next = paddle.log(x=gt_heights_next /
                    anchor_heights_pi)
                targets = paddle.stack(x=(targets_dx, targets_dy,
                    targets_dw, targets_dh, targets_dx_next,
                    targets_dy_next, targets_dw_next, targets_dh_next))
                targets = targets.t()
                targets = targets / paddle.to_tensor(data=[[0.1, 0.1, 0.2, 
                    0.2, 0.1, 0.1, 0.2, 0.2]], dtype='float32').cuda(blocking
                    =True)
                regression_diff = paddle.abs(x=targets - regression[
                    positive_indices, :]) * reg_mask
                regression_loss = paddle.where(condition=paddle.less_equal(
                    x=regression_diff, y=paddle.to_tensor(1.0 / 9.0)), x=
                    0.5 * 9.0 * paddle.pow(x=regression_diff, y=2), y=
                    regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(paddle.to_tensor(data=0).astype(
                    dtype='float32').cuda(blocking=True))
        return paddle.stack(x=classification_losses).mean(axis=0, keepdim=True
            ), paddle.stack(x=regression_losses).mean(axis=0, keepdim=True)


class FocalLossReid(paddle.nn.Layer):

    def forward(self, classifications, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = tuple(classifications.shape)[0]
        classification_losses = []
        for j in range(batch_size):
            classification = classifications[j, :, :]
            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[
                bbox_annotation_next[:, 4] != -1]
            if tuple(bbox_annotation.shape)[0] == 0:
                classification_losses.append(paddle.to_tensor(data=0).
                    astype(dtype='float32').cuda(blocking=True))
                continue
            classification = paddle.clip(x=classification, min=0.0001, max=
                1.0 - 0.0001)
            targets = paddle.ones(shape=tuple(classification.shape)) * -1
            targets = targets.cuda(blocking=True)
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            best_truth_overlap, best_truth_idx = paddle.max(x=IoU, axis=1
                ), paddle.argmax(x=IoU, axis=1)
            _, best_prior_idx = paddle.max(x=IoU, axis=0), paddle.argmax(x=
                IoU, axis=0)
            best_truth_overlap.index_fill_(axis=0, index=best_prior_idx,
                value=2)
            for j in range(best_prior_idx.shape[0]):
                best_truth_idx[best_prior_idx[j]] = j
            assigned_annotations = bbox_annotation[best_truth_idx, :]
            IoU_next = calc_iou(anchors[0, :, :], bbox_annotation_next[:, :4])
            best_truth_overlap_next, best_truth_idx_next = paddle.max(x=
                IoU_next, axis=1), paddle.argmax(x=IoU_next, axis=1)
            _, best_prior_idx_next = paddle.max(x=IoU_next, axis=0
                ), paddle.argmax(x=IoU_next, axis=0)
            best_truth_overlap_next.index_fill_(axis=0, index=
                best_prior_idx_next, value=2)
            for j in range(best_prior_idx_next.shape[0]):
                best_truth_idx_next[best_prior_idx_next[j]] = j
            assigned_annotations_next = bbox_annotation_next[
                best_truth_idx_next, :]
            reid_pos_thres = 0.5
            valid_samples = paddle.greater_equal(x=best_truth_overlap, y=
                paddle.to_tensor(reid_pos_thres)) & paddle.greater_equal(x=
                best_truth_overlap_next, y=paddle.to_tensor(0.4)
                ) | paddle.greater_equal(x=best_truth_overlap_next, y=
                paddle.to_tensor(reid_pos_thres)) & paddle.greater_equal(x=
                best_truth_overlap, y=paddle.to_tensor(0.4))
            targets[valid_samples & paddle.equal(x=assigned_annotations[:, 
                5], y=assigned_annotations_next[:, 5]), :] = 1
            targets[valid_samples & paddle.not_equal(x=assigned_annotations
                [:, 5], y=paddle.to_tensor(assigned_annotations_next[:, 5])), :
                ] = 0
            targets[paddle.less_than(x=best_truth_overlap, y=paddle.
                to_tensor(0.4)) | paddle.less_than(x=
                best_truth_overlap_next, y=paddle.to_tensor(0.4)), :] = 0
            targets[paddle.less_than(x=best_truth_overlap, y=paddle.
                to_tensor(reid_pos_thres)) & paddle.greater_equal(x=
                best_truth_overlap, y=paddle.to_tensor(0.4)) & paddle.
                less_than(x=best_truth_overlap_next, y=paddle.to_tensor(
                reid_pos_thres)) & paddle.greater_equal(x=
                best_truth_overlap_next, y=paddle.to_tensor(0.4)), :] = -1
            positive_indices = paddle.greater_equal(x=targets, y=paddle.
                to_tensor(1))
            num_positive_anchors = positive_indices.sum()
            alpha_factor = paddle.ones(shape=tuple(targets.shape)).cuda(
                blocking=True) * alpha
            alpha_factor = paddle.where(condition=paddle.equal(x=targets, y
                =1.0), x=alpha_factor, y=1.0 - alpha_factor)
            focal_weight = paddle.where(condition=paddle.equal(x=targets, y
                =1.0), x=1.0 - classification, y=classification)
            focal_weight = alpha_factor * paddle.pow(x=focal_weight, y=gamma)
            bce = -(targets * paddle.log(x=classification) + (1.0 - targets
                ) * paddle.log(x=1.0 - classification))
            cls_loss = focal_weight * bce
            cls_loss = paddle.where(condition=paddle.not_equal(x=targets, y
                =paddle.to_tensor(-1.0)), x=cls_loss, y=paddle.zeros(shape=
                tuple(cls_loss.shape)).cuda(blocking=True))
            classification_losses.append(cls_loss.sum() / paddle.clip(x=
                num_positive_anchors.astype(dtype='float32'), min=1.0))
        return paddle.stack(x=classification_losses).mean(axis=0, keepdim=True)
