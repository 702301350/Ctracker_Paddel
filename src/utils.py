import paddle
import numpy as np
import DCT_extraction

from paddle import nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)


# Backbone of ResNet And ResNext
class BA_module_resnet(nn.Layer):
    def __init__(self, pre_channels, cur_channel, reduction=16):
        super(BA_module_resnet, self).__init__()
        self.pre_fusions = nn.LayerList(
            [nn.Sequential(
                nn.Linear(pre_channel, cur_channel // reduction, bias_attr=False),
                nn.BatchNorm1D(cur_channel // reduction)
            )
                for pre_channel in pre_channels]
        )
        self.cur_fusion = nn.Sequential(
            nn.Linear(cur_channel, cur_channel // reduction, bias_attr=False),
            nn.BatchNorm1D(cur_channel // reduction)
        )
        self.generation = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cur_channel // reduction, cur_channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, pre_layers, cur_layer):
        b, cur_c, _, _ = cur_layer.size()

        pre_fusions = [self.pre_fusions[i](pre_layers[i].view(b, -1)) for i in range(len(pre_layers))]
        cur_fusion = self.cur_fusion(cur_layer.view(b, -1))
        fusion = cur_fusion + sum(pre_fusions)

        att_weights = self.generation(fusion).view(b, cur_c, 1, 1)

        return att_weights


# ResNet18 And 34
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Block
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4, reduction=16):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2D(inplanes, width, kernel_size=1, bias_attr=False)

        self.bn1 = nn.BatchNorm2D(width)
        self.conv2 = nn.Conv2D(width, width, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False, groups=groups)
        self.bn2 = nn.BatchNorm2D(width)
        self.conv3 = nn.Conv2D(width, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        self.ba = BA_module_resnet([width, width], 4 * planes, reduction)
        self.feature_extraction = nn.AdaptiveAvgPool2D(1)

        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.feature_extraction1 = DCT_extraction.MultiSpectralAttentionLayer(width, c2wh[planes], c2wh[planes],
                                                                              reduction=reduction,
                                                                              freq_sel_method='top16')
        self.feature_extraction2 = DCT_extraction.MultiSpectralAttentionLayer(width, c2wh[planes], c2wh[planes],
                                                                              reduction=reduction,
                                                                              freq_sel_method='top16')
        self.feature_extraction3 = DCT_extraction.MultiSpectralAttentionLayer(4 * planes, c2wh[planes], c2wh[planes],
                                                                              reduction=reduction,
                                                                              freq_sel_method='top16')


class BBoxTransform(nn.Layer):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = paddle.to_tensor(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = paddle.to_tensor(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2::4] - boxes[:, :, 0::4]
        heights = boxes[:, :, 3::4] - boxes[:, :, 1::4]
        ctr_x = boxes[:, :, 0::4] + 0.5 * widths
        ctr_y = boxes[:, :, 1::4] + 0.5 * heights

        dx = deltas[:, :, 0::4] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1::4] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2::4] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3::4] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = paddle.exp(dw) * widths
        pred_h = paddle.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes_x1 = pred_boxes_x1[:, :, :, np.newaxis]
        pred_boxes_y1 = pred_boxes_y1[:, :, :, np.newaxis]
        pred_boxes_x2 = pred_boxes_x2[:, :, :, np.newaxis]
        pred_boxes_y2 = pred_boxes_y2[:, :, :, np.newaxis]

        pred_boxes = paddle.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2]).reshape(boxes.shape)

        return pred_boxes


class ClipBoxes(nn.Layer):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0::4] = paddle.Tensor.clip(boxes[:, :, 0::4], min=0)
        boxes[:, :, 1::4] = paddle.Tensor.clip(boxes[:, :, 1::4], min=0)

        boxes[:, :, 2::4] = paddle.Tensor.clip(boxes[:, :, 2::4], max=width)
        boxes[:, :, 3::4] = paddle.Tensor.clip(boxes[:, :, 3::4], max=height)

        return boxes
