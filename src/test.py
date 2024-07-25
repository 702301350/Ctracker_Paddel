import sys
import paddle_aux
import paddle
import os
import numpy as np
import time
import math
import copy
import pdb
import argparse
import sys
import cv2
import skimage.io
import skimage.transform
import skimage.color
import skimage
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, RGB_MEAN, RGB_STD
from scipy.optimize import linear_sum_assignment
from thop import profile
from thop import clever_format
print('CUDA available: {}'.format(paddle.device.cuda.device_count() >= 1))
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255,
    255), (255, 255, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0), (0, 
    255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 
    128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]


class detect_rect:
    def __init__(self):
        self.curr_frame = 0
        self.curr_rect = np.array([0, 0, 1, 1])
        self.next_rect = np.array([0, 0, 1, 1])
        self.conf = 0
        self.id = 0

    @property
    def position(self):
        x = (self.curr_rect[0] + self.curr_rect[2]) / 2
        y = (self.curr_rect[1] + self.curr_rect[3]) / 2
        return np.array([x, y])

    @property
    def size(self):
        w = self.curr_rect[2] - self.curr_rect[0]
        h = self.curr_rect[3] - self.curr_rect[1]
        return np.array([w, h])


class tracklet:

    def __init__(self, det_rect):
        self.id = det_rect.id
        self.rect_list = [det_rect]
        self.rect_num = 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame
        self.no_match_frame = 0
        self.last_observation = np.array([-1, -1, -1, -1])
        self.observations = []
        self.vel = np.array((0, 0))

    def add_rect(self, det_rect):
        self.rect_list.append(det_rect)
        self.rect_num = self.rect_num + 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame
        if self.last_observation.sum() > 0:
            previous_box = None
            for i in range(3):
                dt = 3 - i
                if self.last_frame - dt in range(len(self.observations)):
                    previous_box = self.observations[self.last_frame - dt]
                    break
            if previous_box is None:
                previous_box = self.last_observation
            self.vel = speed_direction(previous_box, det_rect.curr_rect)
        self.observations.append(det_rect.curr_rect)
        self.last_observation = det_rect.curr_rect

    @property
    def velocity(self):
        if self.rect_num < 2:
            return 0, 0
        elif self.rect_num < 6:
            return (self.rect_list[self.rect_num - 1].position - self.
                rect_list[self.rect_num - 2].position) / (self.rect_list[
                self.rect_num - 1].curr_frame - self.rect_list[self.
                rect_num - 2].curr_frame)
        else:
            v1 = (self.rect_list[self.rect_num - 1].position - self.
                rect_list[self.rect_num - 4].position) / (self.rect_list[
                self.rect_num - 1].curr_frame - self.rect_list[self.
                rect_num - 4].curr_frame)
            v2 = (self.rect_list[self.rect_num - 2].position - self.
                rect_list[self.rect_num - 5].position) / (self.rect_list[
                self.rect_num - 2].curr_frame - self.rect_list[self.
                rect_num - 5].curr_frame)
            v3 = (self.rect_list[self.rect_num - 3].position - self.
                rect_list[self.rect_num - 6].position) / (self.rect_list[
                self.rect_num - 3].curr_frame - self.rect_list[self.
                rect_num - 6].curr_frame)
            return (v1 + v2 + v3) / 3


def k_precive_observation(observations, curr_frame, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        aim_age = curr_frame - dt
        if aim_age >= 0 and aim_age < len(observations):
            return observations[aim_age]
    return observations[-1]


def speed_direction(track, det):
    CX1, CY1 = (det[0] + det[2]) / 2.0, (det[1] + det[3]) / 2.0
    CX2, CY2 = (track[0] + track[2]) / 2.0, (track[1] + track[3]) / 2.0
    dx = CX2 - CX1
    dy = CY2 - CY1
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-06
    dx = dx / norm
    dy = dy / norm
    return dx, dy


def DIoU(boxa, boxb):
    inter_x1 = max(boxa[0], boxb[0])
    inter_y1 = max(boxa[1], boxb[1])
    inter_x2 = min(boxa[2], boxb[2])
    inter_y2 = min(boxa[3], boxb[3])
    inter_h = max(0, inter_y2 - inter_y1)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_area = inter_w * inter_h
    union_area = (boxa[3] - boxa[1]) * (boxa[2] - boxa[0]) + (boxb[3] - boxb[1]
        ) * (boxb[2] - boxb[0]) - inter_area + 1e-08
    ac_x1 = min(boxa[0], boxb[0])
    ac_y1 = min(boxa[1], boxb[1])
    ac_x2 = max(boxa[2], boxb[2])
    ac_y2 = max(boxa[3], boxb[3])
    boxa_ctrx = boxa[0] + (boxa[2] - boxa[0]) / 2
    boxa_ctry = boxa[1] + (boxa[3] - boxa[1]) / 2
    boxb_ctrx = boxb[0] + (boxb[2] - boxb[0]) / 2
    boxb_ctry = boxb[1] + (boxb[3] - boxb[1]) / 2
    length_box_ctr = (boxb_ctrx - boxa_ctrx) * (boxb_ctrx - boxa_ctrx) + (
        boxb_ctry - boxa_ctry) * (boxb_ctry - boxa_ctry)
    length_ac = (ac_x2 - ac_x1) * (ac_x2 - ac_x1) + (ac_y2 - ac_y1) * (ac_y2 -
        ac_y1)
    diou = inter_area / (union_area + 1e-08) - length_box_ctr / length_ac
    return diou


def cal_diff_angle_cost(track, det_rect, iou_val):
    k_precive_obs = k_precive_observation(track.observations, det_rect.
        curr_frame, 3)
    if k_precive_obs[0] == -1:
        diff_angle_cost = 0
    else:
        dx, dy = speed_direction(k_precive_obs, det_rect.curr_rect)
        inertia_dx, inertia_dy = track.vel[0], track.vel[1]
        diff_angle_cos = inertia_dx * dx + inertia_dy * dy
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi / 2 - diff_angle) / np.pi
        diff_angle_cost = diff_angle * iou_val * 0.1
    return diff_angle_cost


def cal_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if i_w <= 0 or i_h <= 0:
        return 0
    i_s = i_w * i_h
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s)


def cal_simi_track_det(track, det_rect):
    if det_rect.curr_frame <= track.last_frame:
        print('cal_simi_track_det error')
        return 0
    elif det_rect.curr_frame - track.last_frame == 1:
        iou = cal_iou(track.last_rect.next_rect, det_rect.curr_rect)
        iou_val = DIoU(track.last_rect.next_rect, det_rect.curr_rect)
        diff_angle_cost = cal_diff_angle_cost(track, det_rect, iou_val)
        return iou + diff_angle_cost
    else:
        pred_rect = track.last_rect.curr_rect + np.append(track.velocity,
            track.velocity) * (det_rect.curr_frame - track.last_frame)
        iou = cal_iou(pred_rect, det_rect.curr_rect)
        iou_val = DIoU(pred_rect, det_rect.curr_rect)
        diff_angle_cost = cal_diff_angle_cost(track, det_rect, iou_val)
        return iou + diff_angle_cost


def track_det_match(tracklet_list, det_rect_list, min_iou=0.5):
    num1 = len(tracklet_list)
    num2 = len(det_rect_list)
    cost_mat = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i],
                det_rect_list[j])
    match_result = linear_sum_assignment(cost_mat)
    match_result = np.asarray(match_result)
    match_result = np.transpose(match_result)
    matches, unmatched1, unmatched2 = [], [], []
    for i in range(num1):
        if i not in match_result[:, 0]:
            unmatched1.append(i)
    for j in range(num2):
        if j not in match_result[:, 1]:
            unmatched2.append(j)
    for i, j in match_result:
        if cost_mat[i, j] > -min_iou:
            unmatched1.append(i)
            unmatched2.append(j)
        else:
            matches.append((i, j))
    return matches, unmatched1, unmatched2


def draw_caption(image, box, caption, color):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2,
        color, 2)


def run_each_dataset(model_dir, retinanet, dataset_path, subset, cur_dataset):
    print(cur_dataset)
    img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset,
        'img1'))
    img_list = [os.path.join(dataset_path, subset, cur_dataset, 'img1', _) for
        _ in img_list if 'jpg' in _ or 'png' in _]
    img_list = sorted(img_list)
    img_len = len(img_list)
    last_feat = None
    confidence_threshold = 0.4
    IOU_threshold = 0.5
    retention_threshold = 30
    det_list_all = []
    tracklet_all = []
    max_id = 0
    max_draw_len = 100
    draw_interval = 5
    img_width = 1920
    img_height = 1080
    fps = 30
    for i in range(img_len):
        det_list_all.append([])
    frame_times = []
    for idx in range(img_len + 1):
        i = idx - 1
        print('tracking: ', i)
        start_time = time.time()
        print('start_time: ', start_time)
        with paddle.no_grad():
            data_path1 = img_list[min(idx, img_len - 1)]
            img_origin1 = skimage.io.imread(data_path1)
            img_h, img_w, _ = tuple(img_origin1.shape)
            img_height, img_width = img_h, img_w
            resize_h, resize_w = math.ceil(img_h / 32) * 32, math.ceil(
                img_w / 32) * 32
            img1 = np.zeros((resize_h, resize_w, 3), dtype=img_origin1.dtype)
            img1[:img_h, :img_w, :] = img_origin1
            img1 = (img1.astype(np.float32) / 255.0 - np.array([[RGB_MEAN]])
                ) / np.array([[RGB_STD]])
            img1 = paddle.to_tensor(data=img1).transpose(perm=[2, 0, 1]).view(
                1, 3, resize_h, resize_w)
            scores, transformed_anchors, last_feat = retinanet(img1.cuda(
                blocking=True).astype(dtype='float32'), last_feat=last_feat)
            if idx > 0:
                idxs = np.where(scores > 0.1)
                for j in range(tuple(idxs[0].shape)[0]):
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    x3 = int(bbox[4])
                    y3 = int(bbox[5])
                    x4 = int(bbox[6])
                    y4 = int(bbox[7])
                    det_conf = float(scores[idxs[0][j]])
                    det_rect = detect_rect()
                    det_rect.curr_frame = idx
                    det_rect.curr_rect = np.array([x1, y1, x2, y2])
                    det_rect.next_rect = np.array([x3, y3, x4, y4])
                    det_rect.conf = det_conf
                    print(det_rect)
                    if det_rect.conf > confidence_threshold:
                        det_list_all[det_rect.curr_frame - 1].append(det_rect)
                if i == 0:
                    for j in range(len(det_list_all[i])):
                        det_list_all[i][j].id = j + 1
                        max_id = max(max_id, j + 1)
                        track = tracklet(det_list_all[i][j])
                        tracklet_all.append(track)
                    continue
                matches, unmatched1, unmatched2 = track_det_match(tracklet_all,
                    det_list_all[i], IOU_threshold)
                for j in range(len(matches)):
                    det_list_all[i][matches[j][1]].id = tracklet_all[matches
                        [j][0]].id
                    det_list_all[i][matches[j][1]].id = tracklet_all[matches
                        [j][0]].id
                    tracklet_all[matches[j][0]].add_rect(det_list_all[i][
                        matches[j][1]])
                delete_track_list = []
                for j in range(len(unmatched1)):
                    tracklet_all[unmatched1[j]].no_match_frame = tracklet_all[
                        unmatched1[j]].no_match_frame + 1
                    if tracklet_all[unmatched1[j]
                        ].no_match_frame >= retention_threshold:
                        delete_track_list.append(unmatched1[j])
                origin_index = set([k for k in range(len(tracklet_all))])
                delete_index = set(delete_track_list)
                left_index = list(origin_index - delete_index)
                tracklet_all = [tracklet_all[k] for k in left_index]
                for j in range(len(unmatched2)):
                    det_list_all[i][unmatched2[j]].id = max_id + 1
                    max_id = max_id + 1
                    track = tracklet(det_list_all[i][unmatched2[j]])
                    tracklet_all.append(track)
            end_time = time.time()
            print('end_time: ', end_time)
            frame_time = end_time - start_time
            frame_times.append(frame_time)
    Hz = 1 / (sum(frame_times) / len(frame_times))
    print('HZ: ', Hz)
    fout_tracking = open(os.path.join(model_dir, 'results', cur_dataset +
        '.txt'), 'w')
    save_img_dir = os.path.join(model_dir, 'results', cur_dataset)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    out_video = os.path.join(model_dir, 'results', cur_dataset + '.mp4')
    videoWriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('m',
        'p', '4', 'v'), fps, (img_width, img_height))
    id_dict = {}
    for i in range(img_len):
        print('saving: ', i)
        img = cv2.imread(img_list[i])
        for j in range(len(det_list_all[i])):
            x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)
            trace_id = det_list_all[i][j].id
            id_dict.setdefault(str(trace_id), []).append((int((x1 + x2) / 2
                ), y2))
            draw_trace_id = str(trace_id)
            draw_caption(img, (x1, y1, x2, y2), draw_trace_id, color=
                color_list[trace_id % len(color_list)])
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[
                trace_id % len(color_list)], thickness=2)
            trace_len = len(id_dict[str(trace_id)])
            trace_len_draw = min(max_draw_len, trace_len)
            for k in range(trace_len_draw - draw_interval):
                if k % draw_interval == 0:
                    draw_point1 = id_dict[str(trace_id)][trace_len - k - 1]
                    draw_point2 = id_dict[str(trace_id)][trace_len - k - 1 -
                        draw_interval]
                    cv2.line(img, draw_point1, draw_point2, color=
                        color_list[trace_id % len(color_list)], thickness=2)
            fout_tracking.write(str(i + 1) + ',' + str(trace_id) + ',' +
                str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(y2 -
                y1) + ',-1,-1,-1,-1\n')
        cv2.imwrite(os.path.join(save_img_dir, str(i + 1).zfill(6) + '.jpg'
            ), img)
        videoWriter.write(img)
        cv2.waitKey(0)
    fout_tracking.close()
    videoWriter.release()


def run_from_train(model_dir, root_path):
    if not os.path.exists(os.path.join(model_dir, 'results')):
        os.makedirs(os.path.join(model_dir, 'results'))
    retinanet = paddle.load(os.path.join(model_dir, 'model_final.pdparams'))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    # MOT17
    for seq_num in [1, 3, 6, 7, 8, 12, 14]:
        run_each_dataset(model_dir, retinanet, root_path, 'test', 'MOT17-{:02d}'.format(seq_num))
    for seq_num in [2, 4, 5, 9, 10, 11, 13]:
        run_each_dataset(model_dir, retinanet, root_path, 'train', 'MOT17-{:02d}'.format(seq_num))


def main(args=None):
    parser = argparse.ArgumentParser(description=
        'Simple script for testing a CTracker network.')
    parser.add_argument('--dataset_path', default=
        '', type=str,
        help='Dataset path, location of the images sequence.')
    parser.add_argument('--model_dir', default=
        './ctracker/', help='Path to model (.pdparams) file.')
    parser = parser.parse_args(args)
    if not os.path.exists(os.path.join(parser.model_dir, 'results')):
        os.makedirs(os.path.join(parser.model_dir, 'results'))
    retinanet = paddle.load(path=os.path.join(parser.model_dir,
        'model_final.pdparams'))
    use_gpu = True
    if use_gpu:
        retinanet = retinanet.cuda(blocking=True)
    retinanet.eval()
    for seq_num in [14]:
        run_each_dataset(parser.model_dir, retinanet, parser.dataset_path,
            'test', 'MOT17-{:02d}'.format(seq_num))


if __name__ == '__main__':
    main()
