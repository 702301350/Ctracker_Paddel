import paddle
import math
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
from lib.nms import cython_soft_nms_wrapper

pth_model_url = './path/resnext50_32x4d-7cdf4587.pth'  # model path

class PyramidFeatures(paddle.nn.Layer):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        self.P5_1 = paddle.nn.Conv2D(in_channels=C5_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = paddle.nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = paddle.nn.Conv2D(in_channels=feature_size, out_channels
            =feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_1 = paddle.nn.Conv2D(in_channels=C4_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = paddle.nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = paddle.nn.Conv2D(in_channels=feature_size, out_channels
            =feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_1 = paddle.nn.Conv2D(in_channels=C3_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = paddle.nn.Conv2D(in_channels=feature_size, out_channels
            =feature_size, kernel_size=3, stride=1, padding=1)
        self.P6 = paddle.nn.Conv2D(in_channels=C5_size, out_channels=
            feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_1 = paddle.nn.ReLU()
        self.P7_2 = paddle.nn.Conv2D(in_channels=feature_size, out_channels
            =feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        P6_x = self.P6(C5)
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(paddle.nn.Layer):

    def __init__(self, num_features_in, num_anchors=1, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=num_features_in,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act3 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act4 = paddle.nn.ReLU()
        self.output = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=num_anchors * 8, kernel_size=3, padding=1)

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
        out = out.transpose(perm=[0, 2, 3, 1])
        return out.view(tuple(out.shape)[0], -1, 8)


class ClassificationModel(paddle.nn.Layer):

    def __init__(self, num_features_in, num_anchors=1, num_classes=80,
        prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = paddle.nn.Conv2D(in_channels=num_features_in,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act3 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act4 = paddle.nn.ReLU()
        self.output = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = paddle.nn.Sigmoid()

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


class ReidModel(paddle.nn.Layer):

    def __init__(self, num_features_in, num_anchors=1, num_classes=80,
        prior=0.01, feature_size=256):
        super(ReidModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = paddle.nn.Conv2D(in_channels=num_features_in,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act3 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=feature_size, kernel_size=3, padding=1)
        self.act4 = paddle.nn.ReLU()
        self.output = paddle.nn.Conv2D(in_channels=feature_size,
            out_channels=num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = paddle.nn.Sigmoid()

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


class BAResNeXt(paddle.nn.Layer):

    def __init__(self, num_classes, block, layers, zero_init_residual=False,
        groups=32, width_per_group=4, replace_stride_with_dilation=None,
        norm_layer=None, reduction=16):
        self.inplanes = 64
        super(BAResNeXt, self).__init__()
        if norm_layer is None:
            norm_layer = paddle.nn.BatchNorm2D
            self._norm_layer = norm_layer
        self.dilation = 1
        self.reduction = reduction
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = norm_layer(64)
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], groups)
        self.layer2 = self._make_layer(block, 128, layers[1], groups,
            stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], groups,
            stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], groups,
            stride=2, dilate=replace_stride_with_dilation[2])
        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels,
                self.layer3[layers[2] - 1].conv2.out_channels, self.layer4[
                layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                self.layer3[layers[2] - 1].conv3.out_channels, self.layer4[
                layers[3] - 1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.num_classes = num_classes
        self.regressionModel = RegressionModel(512)
        self.classificationModel = ClassificationModel(512, num_classes=
            num_classes)
        self.reidModel = ReidModel(512, num_classes=num_classes)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()
        self.reidfocalLoss = losses.FocalLossReid()

        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                paddle.nn.initializer.KaimingNormal(fan_in=None)(m.weight)
            elif isinstance(m, (paddle.nn.BatchNorm2D, paddle.nn.GroupNorm)):
                init_Constant = paddle.nn.initializer.Constant(value=1)
                init_Constant(m.weight)
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bn3.weight)
        prior = 0.01

        self.classificationModel.output.weight.data.fill_(value=0)
        self.classificationModel.output.bias.data.fill_(value=-math.log((
            1.0 - prior) / prior))
        self.reidModel.output.weight.data.fill_(value=0)
        self.reidModel.output.bias.data.fill_(value=-math.log((1.0 - prior) /
            prior))
        self.regressionModel.output.weight.data.fill_(value=0)
        self.regressionModel.output.bias.data.fill_(value=0)
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, groups, stride=1, dilate=False
        ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=
                self.inplanes, out_channels=planes * block.expansion,
                kernel_size=1, stride=stride, bias_attr=False), norm_layer(
                planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            groups=self.groups, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width))
        return paddle.nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.sublayers():
            if isinstance(layer, paddle.nn.BatchNorm2D):
                layer.eval()

    def forward(self, inputs, last_feat=None):
        if self.training:
            img_batch_1, annotations_1, img_batch_2, annotations_2 = inputs
            img_batch = paddle.concat(x=[img_batch_1, img_batch_2], axis=0)
            annotations = paddle.concat(x=[annotations_1, annotations_2],
                axis=0)
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
        anchors = self.anchors(tuple(img_batch.shape)[2:])
        if self.training:
            track_features = []
            for ind, featmap in enumerate(features):
                featmap_t, featmap_t1 = paddle.chunk(x=featmap, chunks=2,
                    axis=0)
                track_features.append(paddle.concat(x=(featmap_t,
                    featmap_t1), axis=1))
            reg_features = []
            cls_features = []
            reid_features = []
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)
                reid_feat = reid_mask.transpose(perm=[0, 2, 3, 1])
                batch_size, width, height, _ = tuple(reid_feat.shape)
                reid_feat = reid_feat.view(batch_size, -1, self.num_classes)
                cls_mask = self.classificationModel(feature)
                cls_feat = cls_mask.transpose(perm=[0, 2, 3, 1])
                cls_feat = cls_feat.view(batch_size, -1, self.num_classes)
                reg_in = feature * reid_mask * cls_mask
                reg_feat = self.regressionModel(reg_in)
                reg_features.append(reg_feat)
                cls_features.append(cls_feat)
                reid_features.append(reid_feat)
            regression = paddle.concat(x=reg_features, axis=1)
            classification = paddle.concat(x=cls_features, axis=1)
            reid = paddle.concat(x=reid_features, axis=1)
            return self.focalLoss(classification, regression, anchors,
                annotations_1, annotations_2), self.reidfocalLoss(reid,
                anchors, annotations_1, annotations_2)
        else:
            if last_feat is None:
                return paddle.zeros(shape=[0]), paddle.zeros(shape=[0, 4]
                    ), features
            track_features = []
            for ind, featmap in enumerate(features):
                track_features.append(paddle.concat(x=(last_feat[ind],
                    featmap), axis=1))
            reg_features = []
            cls_features = []
            reid_features = []
            for ind, feature in enumerate(track_features):
                reid_mask = self.reidModel(feature)
                reid_feat = reid_mask.transpose(perm=[0, 2, 3, 1])
                batch_size, width, height, _ = tuple(reid_feat.shape)
                reid_feat = reid_feat.view(batch_size, -1, self.num_classes)
                cls_mask = self.classificationModel(feature)
                cls_feat = cls_mask.transpose(perm=[0, 2, 3, 1])
                cls_feat = cls_feat.view(batch_size, -1, self.num_classes)
                reg_in = feature * reid_mask * cls_mask
                reg_feat = self.regressionModel(reg_in)
                reg_features.append(reg_feat)
                cls_features.append(cls_feat)
                reid_features.append(reid_feat)
            regression = paddle.concat(x=reg_features, axis=1)
            classification = paddle.concat(x=cls_features, axis=1)
            reid_score = paddle.concat(x=reid_features, axis=1)
            anchors = paddle.concat(x=(anchors, anchors), axis=2)
            transformed_anchors = self.regressBoxes(anchors, regression)
            scores = (paddle.max(x=classification, axis=2, keepdim=True),
                paddle.argmax(x=classification, axis=2, keepdim=True))[0]
            scores_over_thresh = (scores > 0.05)[0, :, 0]
            if scores_over_thresh.sum() == 0:
                return paddle.zeros(shape=[0]), paddle.zeros(shape=[0, 4]
                    ), features
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            reid_score = reid_score[:, scores_over_thresh, :]
            final_bboxes = cython_soft_nms_wrapper(0.7, method='gaussian')(
                paddle.concat(x=[transformed_anchors[:, :, :], scores,
                reid_score], axis=2)[0, :, :].cpu().numpy())
            return final_bboxes[:, -2], final_bboxes, features


def resnext50_32x4d(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BAResNeXt(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        checkpoint = paddle.load(path=pth_model_url)
        model.set_state_dict(state_dict=checkpoint, use_structured_name=False)
        print(model)
    return model
