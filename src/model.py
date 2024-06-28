import paddle
import numpy as np

from paddle import nn

pth_model_url = './path/resnext50_32x4d-7cdf4587.pth'  # model path

# 金字塔特征
class PyramidFeatures(nn.Layer):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1              = nn.Conv2D(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_unsampled      = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2              = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1              = nn.Conv2D(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_unsampled      = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2              = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1              = nn.Conv2D(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2              = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6                = nn.Conv2D(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1              = nn.ReLU()
        self.P7_2              = nn.Conv2D(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

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

        self.conv1              = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act1               = nn.ReLU()

        self.conv2              = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act2               = nn.ReLU()

        self.conv3              = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act3               = nn.ReLU()

        self.conv4              = nn.Conv2D(num_features_in, features_size, kernel_size=3, padding=1)
        self.act4               = nn.ReLU()
        self.output             = nn.Conv2D(features_size, num_anchors * 8, kernel_size=3, padding=1)

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

        # out is B x C x W x H, with C = 4*num_anchors，我觉得应该是8*num_anchors(成对目标),成对边界框的(x1,y1,x2,y2)
        out = out.permute(0, 2, 3, 1)  #permute:将tensor的维度换位, out: (B,W,H,C=8)

        return out.contiguous().view(out.shape[0], -1, 8)  #contiguous: 即深拷贝，对out使用了.contiguous()后，改变后者的值，对out没有任何影响. view(B, W x H, C=8)

# 目标分类分支(Object Classification branch)：使用4个连续的3*3卷积和relu激活层交错进行特征学习，最后使用一个3*3的卷积加sigmoid激活函数预测置信度
class ClassificationModel(nn.Layer):
    def __init__(self, num_features_in, num_anchors=1, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1              = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1               = nn.ReLU()

        self.conv2              = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2               = nn.ReLU()

        self.conv3              = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3               = nn.ReLU()

        self.conv4              = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4               = nn.ReLU()

        self.output             = nn.Conv2D(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act         = nn.Sigmoid()

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

        self.conv1               = nn.Conv2D(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1                = nn.ReLU()

        self.conv2               = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2                = nn.ReLU()

        self.conv3               = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3                = nn.ReLU()

        self.conv4               = nn.Conv2D(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4                = nn.ReLU()

        self.output              = nn.Conv2D(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act          = nn.Sigmoid()

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
    def __initOO(self, num_classes, block, layers, zero_init_residual=False,
                 groups=32, ):
