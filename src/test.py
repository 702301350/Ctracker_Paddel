import argparse
import math
import os

import cv2
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import paddle
from scipy.optimize import linear_sum_assignment

from dataloader import RGB_MEAN, RGB_STD

print('CUDA available: {}'.format(paddle.device.cuda.device_count()))

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255),
              (0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255),
              (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]


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

    def add_rect(self, det_rect):
        self.rect_list.append(det_rect)  # 增加检测结果
        self.rect_num = self.rect_num + 1
        self.last_rect = det_rect
        self.last_frame = det_rect.curr_frame  # 重要，轨迹的最后一帧的帧序等于当前检测结果的帧序

    @property
    def velocity(self):
        if self.rect_num < 2:
            return 0, 0
        elif self.rect_num < 6:
            return (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 2].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 2].curr_frame)
        else:
            v1 = (self.rect_list[self.rect_num - 1].position - self.rect_list[self.rect_num - 4].position) / (
                        self.rect_list[self.rect_num - 1].curr_frame - self.rect_list[self.rect_num - 4].curr_frame)
            v2 = (self.rect_list[self.rect_num - 2].position - self.rect_list[self.rect_num - 5].position) / (
                        self.rect_list[self.rect_num - 2].curr_frame - self.rect_list[self.rect_num - 5].curr_frame)
            v3 = (self.rect_list[self.rect_num - 3].position - self.rect_list[self.rect_num - 6].position) / (
                        self.rect_list[self.rect_num - 3].curr_frame - self.rect_list[self.rect_num - 6].curr_frame)
            return (v1 + v2 + v3) / 3


def cal_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    i_w = min(x2, x4) - max(x1, x3)
    i_h = min(y2, y4) - max(y1, y3)
    if i_w <= 0 or i_h <= 0:
        return 0
    i_s = i_w * i_h  # 交集
    s_1 = (x2 - x1) * (y2 - y1)
    s_2 = (x4 - x3) * (y4 - y3)
    return float(i_s) / (s_1 + s_2 - i_s)


def cal_simi(det_rect1, det_rect2):
    return cal_iou(det_rect1.next_rect, det_rect2.curr_rect)


def cal_simi_track_det(track, det_rect):
    if det_rect.curr_frame <= track.last_frame:  # 要求当前检测的帧序大于前一个跟踪轨迹的帧序
        print("cal_simi_track_det error")
        return 0
    elif det_rect.curr_frame - track.last_frame == 1:  # 当前检测结果的帧序和轨迹最后一个检测的帧序相差1，则计算两者的IOU
        return cal_iou(track.last_rect.next_rect, det_rect.curr_rect)  # 调用cal_iou(),计算IOU
    else:
        pred_rect = track.last_rect.curr_rect + np.append(track.velocity, track.velocity) * (
                    det_rect.curr_frame - track.last_frame)
        return cal_iou(pred_rect, det_rect.curr_rect)


def track_det_match(tracklet_list, det_rect_list, min_iou=0.5):  #
    num1 = len(tracklet_list)  # 24
    num2 = len(det_rect_list)  # 23
    cost_mat = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            cost_mat[i, j] = -cal_simi_track_det(tracklet_list[i], det_rect_list[j])  # 调用cal_simi_track_det()函数，为啥带个负号

    match_result = linear_sum_assignment(cost_mat)  # 匈牙利算法进行匹配
    match_result = np.asarray(match_result)  # 轨迹池中的第16个跟踪目标未被匹配(轨迹池中的目标数大于检测结果目标数). np.asarray:将结构数据转换为数组类型
    match_result = np.transpose(
        match_result)  # 跟踪-检测对 [[0，0],[1，1],[2，2],[3，3],[4，7],[5，5],[6，6],[7，11],[8，9],[9，4],[10，10],[11，8]，[12，13],[13，14],[14，15],[15，12],[17，18],[18，17],[19，19],[20，16],[21,21],[22，20],[23，22]]

    matches, unmatched1, unmatched2 = [], [], []
    for i in range(num1):
        if i not in match_result[:, 0]:
            unmatched1.append(i)  # 未匹配的跟踪
    for j in range(num2):
        if j not in match_result[:, 1]:
            unmatched2.append(j)  # 未匹配的检测
    for i, j in match_result:
        if cost_mat[i, j] > -min_iou:  # min_iou:0.5
            unmatched1.append(i)
            unmatched2.append(j)
        else:
            matches.append((i, j))
    return matches, unmatched1, unmatched2


def draw_caption(image, box, caption, color):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)  # (b[0], b[1] - 8)：字体位置


def run_each_dataset(model_dir, retinanet, dataset_path, subset, cur_dataset):
    print(cur_dataset)  # current dataset sequence

    img_list = os.listdir(os.path.join(dataset_path, subset, cur_dataset, 'img1'))
    img_list = [os.path.join(dataset_path, subset, cur_dataset, 'img1', _) for _ in img_list if
                ('jpg' in _) or ('png' in _)]
    img_list = sorted(img_list)  # 对加载的图片进行排序

    img_len = len(img_list)  # 视频序列长度
    last_feat = None  # 在model.py出现了if last_feat is None，应该是用于测试

    confidence_threshold = 0.4  # 检测置信值的阈值
    IOU_threshold = 0.5
    retention_threshold = 10  # 保留阈值，大于该阈值，则删除该轨迹

    det_list_all = []  # 最重要变量1，加载所有视频帧的pred_bbox(经过soft-max后)
    tracklet_all = []  # 最重要变量2，加载所有视频帧的跟踪轨迹
    max_id = 0
    max_draw_len = 100  # 最大的轨迹长度
    draw_interval = 5  # 间隔
    img_width = 1920  # MOT20-01和MOT20-02：img_width和img_height分别为1920和1080，MOT20-03：[1173，880], MOT20-05：[1654，1080]
    img_height = 1080
    fps = 30

    for i in range(img_len):  # det_list_all:存储所有帧的detections
        det_list_all.append([])

    for idx in range(img_len + 1):
        i = idx - 1
        print('tracking: ', i)
        with paddle.no_grad():
            data_path1 = img_list[min(idx, img_len - 1)]
            img_origin1 = skimage.io.imread(data_path1)  # 读取图片
            img_h, img_w, _ = img_origin1.shape
            img_height, img_width = img_h, img_w
            resize_h, resize_w = math.ceil(img_h / 32) * 32, math.ceil(
                img_w / 32) * 32  # 使图像的长、宽是32的倍数. math.ceil:向上取整，即小数部分直接舍去，并向整数部分进1
            img1 = np.zeros((resize_h, resize_w, 3), dtype=img_origin1.dtype)  # [1088，1920，3]
            img1[:img_h, :img_w, :] = img_origin1
            img1 = (img1.astype(np.float32) / 255.0 - np.array([[RGB_MEAN]])) / np.array(
                [[RGB_STD]])  # astype:对数据类型进行转换，除以255就是归一化
            img1 = paddle.from_numpy(img1).permute(2, 0, 1).view(1, 3, resize_h, resize_w)  # [1，3，1088,1920]
            # scores:前景的置信度得分,transformed_anchors:经过soft-nms后的pred_boxes， last_feat：5个特征级上的特征
            scores, transformed_anchors, last_feat = retinanet(img1.cuda().float(), last_feat=last_feat)  # 执行检测过程(重要)
            if idx > 0:
                idxs = np.where(scores > 0.1)  # 检测置信值(前景的置信度)大于0.1的pred_bbox， idxs：0~45

                for j in range(idxs[0].shape[0]):  # 46
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])  # 上一帧检测置信值大于0.1的predict bbox
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    # 当前帧检测置信值大于0.1的的predict bbox
                    x3 = int(bbox[4])
                    y3 = int(bbox[5])
                    x4 = int(bbox[6])
                    y4 = int(bbox[7])

                    det_conf = float(scores[idxs[0][j]])  # 检测置信值(前景的置信度得分)
                    # det_rect：边界框对(相邻帧)
                    det_rect = detect_rect()
                    det_rect.curr_frame = idx
                    det_rect.curr_rect = np.array([x1, y1, x2, y2])
                    det_rect.next_rect = np.array([x3, y3, x4, y4])
                    det_rect.conf = det_conf

                    if det_rect.conf > confidence_threshold:  # confidence_threshold：0.4
                        det_list_all[det_rect.curr_frame - 1].append(det_rect)  # 加载所有视频帧置信值大于0.4的成对pred_bbox

                if i == 0:  # 视频第1帧
                    for j in range(len(det_list_all[i])):  # j:0~23
                        det_list_all[i][j].id = j + 1  # 第i帧第j个检测的id
                        max_id = max(max_id, j + 1)  # 最大id
                        track = tracklet(det_list_all[i][j])  # 调用tracklet()函数，添加轨迹
                        tracklet_all.append(track)
                    continue

                matches, unmatched1, unmatched2 = track_det_match(tracklet_all, det_list_all[i],
                                                                  IOU_threshold)  # 跟踪-检测匹配(IOU匹配). matches:匹配的跟踪-检测对，unmatched1：未匹配的跟踪，unmatched2：未匹配的检测

                for j in range(len(matches)):
                    det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][
                        0]].id  # 将匹配的跟踪轨迹id赋值给检测结果，det_list_all[i][n]:表示第i帧第n个成对pred_bbox. matches[j][1]:第j个跟踪-检测对中的检测
                    det_list_all[i][matches[j][1]].id = tracklet_all[matches[j][0]].id
                    tracklet_all[matches[j][0]].add_rect(det_list_all[i][matches[j][
                        1]])  # 跟踪轨迹池中添加完成新匹配的检测结果. matches[j][0]:第j个跟踪-检测对中的跟踪. 调用add_rect()函数

                delete_track_list = []  # 删除的轨迹
                for j in range(len(unmatched1)):  # unmatched1：未匹配的跟踪，例如第16号
                    tracklet_all[unmatched1[j]].no_match_frame = tracklet_all[unmatched1[
                        j]].no_match_frame + 1  # no_match_frame：未匹配帧的数量
                    if tracklet_all[unmatched1[j]].no_match_frame >= retention_threshold:  # retention_threshold：10
                        delete_track_list.append(unmatched1[j])

                origin_index = set([k for k in range(len(tracklet_all))])  # 创建一个集合
                delete_index = set(delete_track_list)
                left_index = list(origin_index - delete_index)  # 剩余的跟踪轨迹索引
                tracklet_all = [tracklet_all[k] for k in left_index]

                for j in range(len(unmatched2)):  # unmatched2:未匹配的检测
                    det_list_all[i][unmatched2[j]].id = max_id + 1  # i:帧序-1
                    max_id = max_id + 1
                    track = tracklet(det_list_all[i][unmatched2[j]])  # 调用tracklet()函数，将未匹配的检测添加到轨迹池中
                    tracklet_all.append(track)  #

    # **************visualize tracking result and save evaluate file(可视化跟踪结果并保存评估结果文件)****************

    fout_tracking = open(os.path.join(model_dir, 'results', cur_dataset + '.txt'), 'w')

    save_img_dir = os.path.join(model_dir, 'results', cur_dataset)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    out_video = os.path.join(model_dir, 'results', cur_dataset + '.mp4')  # 跟踪视频序列./trained_model/results/MOT20-01.mp4
    videoWriter = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (img_width,
                                                                                               img_height))  # cv2.VideoWriter:写视频. cv2.VideoWriter_fourcc('m', 'p', '4', 'v')：相对较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.m4v

    id_dict = {}  # 空字典

    for i in range(img_len):  # 第i+1帧
        print('saving: ', i)
        img = cv2.imread(img_list[i])  # BGR

        for j in range(len(det_list_all[i])):  # 检测结果列表中第i+1帧第j+1个成对pred_bbox

            x1, y1, x2, y2 = det_list_all[i][j].curr_rect.astype(int)  # 检测结果列表中第i+1帧第j+1个成对pred_bbox
            trace_id = det_list_all[i][j].id

            id_dict.setdefault(str(trace_id), []).append((int((x1 + x2) / 2),
                                                          y2))  # 字典设置默认值，例如：{'1':[(718,899),(718,902),(),...],'2':[799,891][],'3':[][]}，主要用于画线
            draw_trace_id = str(trace_id)
            draw_caption(img, (x1, y1, x2, y2), draw_trace_id,
                         color=color_list[trace_id % len(color_list)])  # 调用draw_caption()函数
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)  # 画跟踪框

            trace_len = len(id_dict[str(trace_id)])  # 轨迹长度和视频帧数相等
            trace_len_draw = min(max_draw_len, trace_len)  # max_draw_len:100，轨迹长度不能超过100

            for k in range(trace_len_draw - draw_interval):  # draw_interval：5
                if k % draw_interval == 0:  # 画轨迹(每间隔5帧画一直线)
                    draw_point1 = id_dict[str(trace_id)][
                        trace_len - k - 1]  # 轨迹线的第1个点的位置: 第一个目标(id=1)在第6帧时的位置(715, 907)
                    draw_point2 = id_dict[str(trace_id)][
                        trace_len - k - 1 - draw_interval]  # 轨迹线的第二个点的位置: 第一个目标(id=1)在第1帧时的位置(718, 899)
                    cv2.line(img, draw_point1, draw_point2, color=color_list[trace_id % len(color_list)],
                             thickness=2)  # 画线，将两个时刻的点连接起来，形成轨迹

            fout_tracking.write(
                str(i + 1) + ',' + str(trace_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2 - x1) + ',' + str(
                    y2 - y1) + ',-1,-1,-1,-1\n')  # 输出跟踪结果

        cv2.imwrite(os.path.join(save_img_dir, str(i + 1).zfill(6) + '.jpg'), img)  # 保存图像
        videoWriter.write(img)  # 将图片生成视频
        cv2.waitKey(0)  # 因为delay=0, 只会显示第一帧视频

    fout_tracking.close()  # 关闭写入结果的txt文件
    videoWriter.release()  # 关闭摄像头


def run_from_train(model_dir, root_path):
    if not os.path.exists(os.path.join(model_dir, 'results')):
        os.makedirs(os.path.join(model_dir, 'results'))
    retinanet = paddle.load(os.path.join(model_dir, 'model_final.pt'))

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
    parser = argparse.ArgumentParser(description='Simple script for testing a CTracker network.')

    parser.add_argument('--dataset_path', default=r'', type=str,
                        help='Dataset path, location of the images sequence.')
    parser.add_argument('--model_dir', default='./ctracker/', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if not os.path.exists(os.path.join(parser.model_dir, 'results')):  # 创建results/文件夹路径
        os.makedirs(os.path.join(parser.model_dir, 'results'))

    retinanet = (paddle.load(os.path.join(parser.model_dir, 'model_final.pt')))

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    # MOT17
    for seq_num in [1, 3, 6, 7, 8, 12, 14]:
        run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'test', 'MOT17-{:02d}'.format(seq_num))
    for seq_num in [2, 4, 5, 9, 10, 11, 13]:
        run_each_dataset(parser.model_dir, retinanet, parser.dataset_path, 'train', 'MOT17-{:02d}'.format(seq_num))


if __name__ == '__main__':
    main()
