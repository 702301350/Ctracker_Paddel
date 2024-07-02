import math

import paddle
import numpy as np
import utils
import anchors
import losses

from paddle import nn
from lib.nms import cython_soft_nms_wrapper

pth_model_url = './path/resnext50_32x4d-7cdf4587.pth'  # model path


# 金字塔特征
class PyramidFeatures(nn.Layer):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2D(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_unsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2D(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_unsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2D(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2D(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)  # M5
        P5_upsampled_x = self.P5_upsampled(P5_x)  # 2x
        P5_x = self.P5_2(P5_x)  # P5

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x  # M4
        P4_upsampled_x = self.P4_upsampled(P4_x)  # 2X
        P4_x = self.P4_2(P4_x)  # P4

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x  # M3
        P3_x = self.P3_2(P3_x)  # P3

        P6_x = self.P6(C5)  # P6

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)  # P7

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


# 成对边界框回归(Paired Boxes Regression)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，为每个目标返回一个边界框对
class RegressionModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=1, features_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2D(features_size, num_anchors * 8, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 8)


# 目标分类分支(Object Classification branch)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，最后使用一个3*3的卷积加sigmoid激活函数预测置信度
class ClassificationModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=1, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2D(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out


# ID确认分支(ID Verification branch)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，最后使用一个3*3的卷积加sigmoid激活函数预测置信度
class ReidModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=1, num_classes=80, prior=0.01, feature_size=256):
        super(ReidModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2D(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        return out


class BAResNext(nn.Layer):
    def __init__(self, num_classes, block, layers, zero_init_residual=False,
                 groups=32, width_per_group=4, replace_stride_with_dilation=None, norm_layer=None,
                 reduction=16):

        self.inplanes = 64
        super(BAResNext, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
            self._norm_layer = norm_layer

        self.dilation = 1
        self.reduction = reduction
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(64)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], groups)
        self.layer2 = self._make_layer(block, 128, layers[1], groups, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], groups, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], groups, stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if block == utils.BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == utils.Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.num_classes = num_classes

        self.regressionModel = RegressionModel(512)
        self.classificationModel = ClassificationModel(512,
                                                       num_classes=num_classes)
        self.reidModel = ReidModel(512, num_classes=num_classes)

        self.anchors = anchors.Anchors()
        self.regressBoxes = utils.BBoxTransform()
        self.clipBoxes = utils.ClipBoxes()

        self.focalLoss = losses.FocalLoss()
        self.reidfocalLoss = losses.FocalLossReid()

        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal(fan_in=None)(m.weight)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                constant_init = paddle.nn.initializer.Constant(value=0.0)
                constant_init(m.weight)
                constant_init(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, utils.Bottleneck):
                    constant_init = paddle.nn.initializer.Constant(value=0.0)
                    constant_init(m.bn3.weight, 0)

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.reidModel.output.weight.data.fill_(0)
        self.reidModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, groups, stride=1, dilate=False):
        norm_layer = self._norm_layer  # 其实就是BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2D):
                layer.eval()

    def forward(self, inputs, last_feat=None):
        if self.training:
            img_batch_1, annotations_1, img_batch_2, annotations_2 = inputs
            img_batch = paddle.concat([img_batch_1, img_batch_2], 0)
            annotations = paddle.concat([annotations_1, annotations_2], 0)
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        anchors = self.anchors(img_batch.shape[2:])

        if self.training:
            track_features = []
            for ind, featmap in enumerate(features):
                featmap_t, featmap_t1 = paddle.chunk(featmap, chunks=2, axis=0)
                track_features.append(paddle.concat((featmap_t, featmap_t1), axis=1))

            reg_features = []
            cls_features = []
            reid_features = []
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)

                reid_feat = reid_mask.permute(0, 2, 3, 1)
                batch_size, width, height, _ = reid_feat.shape
                reid_feat = reid_feat.contiguous().view(batch_size, -1, self.num_classes)

                cls_mask = self.classificationModel(feature)

                cls_feat = cls_mask.permute(0, 2, 3, 1)
                cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)

                reg_in = feature * reid_mask * cls_mask

                reg_feat = self.regressionModel(reg_in)

                reg_features.append(reg_feat)
                cls_features.append(cls_feat)
                reid_features.append(reid_feat)
            regression = paddle.concat(reg_features, axis=1)

            classification = paddle.concat(cls_features, axis=1)

            reid = paddle.concat(reid_features, axis=1)

            return self.focalLoss(classification, regression, anchors, annotations_1, annotations_2), self.reidfocalLoss(reid, anchors, annotations_1, annotations_2)

        else:
            if last_feat is None:
                return paddle.zeros(0), paddle.zeros(0, 4), features
            track_features = []
            for ind, featmap in enumerate(features):
                track_features.append(paddle.concat((last_feat[ind], featmap), axis=1))

            reg_features = []
            cls_features = []
            reid_features = []
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)

                reid_feat = reid_mask.permute(0, 2, 3, 1)
                batch_size, width, height, _ = reid_feat.shape
                reid_feat = reid_feat.contiguous().view(batch_size, -1, self.num_classes)

                cls_mask = self.classificationModel(feature)

                cls_feat = cls_mask.permute(0, 2, 3, 1)
                cls_feat = cls_feat.contiguous().view(batch_size, -1, self.num_classes)

                reg_in = feature * reid_mask * cls_mask

                reg_feat = self.regressionModel(reg_in)

                reg_features.append(reg_feat)
                cls_features.append(cls_feat)
                reid_features.append(reid_feat)
            regression = paddle.concat(reg_features, axis=1)

            classification = paddle.concat(cls_features, axis=1)

            reid_score = paddle.concat(reid_features, axis=1)

            anchors = paddle.concat((anchors, anchors), axis=2)

            transformed_anchors = self.regressBoxes(anchors, regression)

            scores = paddle.max(classification, axis=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return paddle.zeros(0), paddle.zeros(0, 4), features

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            reid_score = reid_score[:, scores_over_thresh, :]

            final_bboxes = cython_soft_nms_wrapper(0.7, method='gaussian')(
                paddle.concat([transformed_anchors[:, :, :].contiguous(), scores, reid_score], axis=2)[0, :, :].cpu().numpy())

            return final_bboxes[:, -2], final_bboxes, features


def resnext50_32x4d(num_classes, pretrained=False, **kwargs):
    model = BAResNext(num_classes, utils.Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        checkpoint = paddle.load('../pth/resnext50_32x4d-7cdf4587.pth')
        model.load_state_dict(checkpoint, strict=False)
        print(model)

    return model

