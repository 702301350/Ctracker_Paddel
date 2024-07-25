import paddle
import warnings
from wxw_GLFIM import GLFIM

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [(d * (x - 1) + 1) for
            x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [(x // 2) for x in k]
    return p


class CBS(paddle.nn.Layer):
    default_act = paddle.nn.Silu()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels=c1, out_channels=c2,
            kernel_size=k, stride=s, padding=autopad(k, p, d), groups=g,
            dilation=d, bias_attr=False)
        self.bn = paddle.nn.BatchNorm2D(num_features=c2)
        self.act = self.default_act if act is True else act if isinstance(act,
            paddle.nn.Layer) else paddle.nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(paddle.nn.Layer):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 8
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c_ * 4, c2, 1, 1)
        self.m = paddle.nn.MaxPool2D(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(paddle.concat(x=(x, y1, y2, self.m(y2)), axis=1))


class ResConnect(paddle.nn.Layer):
    expansion = 1
    def __init__(self, filter_in, filter_out):
        super(ResConnect, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=filter_in, out_channels=
            filter_out, kernel_size=3, padding=1)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=filter_out, momentum=
            1 - 0.1)
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=filter_out, out_channels=
            filter_out, kernel_size=3, padding=1)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=filter_out, momentum=
            1 - 0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class RFA(paddle.nn.Layer):
    def __init__(self, in_channel, out_channel, pool_ratios=[0.1, 0.2, 0.3]):
        super().__init__()
        self.adaptive_pool_output_ratio = pool_ratios
        self.high_lateral_conv_attention = paddle.nn.Sequential(paddle.nn.
            Conv2D(in_channels=out_channel * len(self.
            adaptive_pool_output_ratio), out_channels=out_channel,
            kernel_size=1), paddle.nn.ReLU(), paddle.nn.Conv2D(in_channels=
            out_channel, out_channels=len(self.adaptive_pool_output_ratio),
            kernel_size=3, padding=1))
        self.high_lateral_conv = paddle.nn.LayerList()
        self.high_lateral_conv.extend([paddle.nn.Conv2D(in_channels=
            in_channel, out_channels=out_channel, kernel_size=1) for k in
            range(len(self.adaptive_pool_output_ratio))])

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        AdapPool_Features = [paddle.nn.functional.upsample(self.
            high_lateral_conv[j](paddle.nn.functional.adaptive_avg_pool2d(x
            =x, output_size=(max(1, int(h * self.adaptive_pool_output_ratio
            [j])), max(1, int(w * self.adaptive_pool_output_ratio[j]))))),
            size=(h, w), mode='bilinear', align_corners=True) for j in
            range(len(self.adaptive_pool_output_ratio))]
        Concat_AdapPool_Features = paddle.concat(x=AdapPool_Features, axis=1)
        fusion_weights = self.high_lateral_conv_attention(
            Concat_AdapPool_Features)
        fusion_weights = paddle.nn.functional.sigmoid(x=fusion_weights)
        adap_pool_fusion = 0
        for i in range(len(self.adaptive_pool_output_ratio)):
            adap_pool_fusion += paddle.unsqueeze(x=fusion_weights[:, i, :,
                :], axis=1) * AdapPool_Features[i]
        return adap_pool_fusion


class Body(paddle.nn.Layer):
    def __init__(self, channel):
        super(Body, self).__init__()
        self.R5 = SPPF(channel, channel)
        self.ResBlock5 = paddle.nn.Sequential(ResConnect(channel, channel),
            ResConnect(channel, channel), ResConnect(channel, channel))
        self.conv5_1 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.upsampled5 = paddle.nn.Upsample(scale_factor=2, mode='nearest')
        self.ResBlock4 = paddle.nn.Sequential(ResConnect(channel, channel),
            ResConnect(channel, channel), ResConnect(channel, channel))
        self.conv4_1 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.upsampled4 = paddle.nn.Upsample(scale_factor=2, mode='nearest')
        self.ResBlock3 = paddle.nn.Sequential(ResConnect(channel, channel),
            ResConnect(channel, channel), ResConnect(channel, channel))
        self.conv3_1 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.dow3 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=3, stride=2, padding=1)
        self.P3 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.ResBlock4_2 = paddle.nn.Sequential(ResConnect(channel, channel
            ), ResConnect(channel, channel), ResConnect(channel, channel))
        self.conv4_2 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.dow4 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=3, stride=2, padding=1)
        self.P4 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)
        self.ResBlock5_2 = paddle.nn.Sequential(ResConnect(channel, channel
            ), ResConnect(channel, channel), ResConnect(channel, channel))
        self.P5 = paddle.nn.Conv2D(in_channels=channel, out_channels=
            channel, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        M3, M4, M5 = inputs
        M5_1 = self.R5(M5)
        M5_2 = self.ResBlock5(M5_1)
        M5_3 = self.conv5_1(M5_2)
        M5_u = self.upsampled5(M5_3)
        M4_C5 = M5_u + M4
        M4_1 = self.ResBlock4(M4_C5)
        M4_2 = self.conv4_1(M4_1)
        M4_u = self.upsampled4(M4_2)
        M3_C4 = M4_u + M3
        M3_1 = self.ResBlock3(M3_C4)
        P3 = self.P3(M3_1)
        M3_2 = self.conv3_1(M3_1)
        M3_d = self.dow3(M3_2)
        M4_C3 = M3_d + M4_2
        M4_22 = self.ResBlock4_2(M4_C3)
        P4 = self.P4(M4_22)
        M4_33 = self.conv4_2(M4_22)
        M4_d = self.dow4(M4_33)
        M5_C4 = M4_d + M5_3
        M5_22 = self.ResBlock5_2(M5_C4)
        P5 = self.P5(M5_22)
        return P3, P4, P5


class SRFPN(paddle.nn.Layer):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(SRFPN, self).__init__()
        self.M3 = paddle.nn.Conv2D(in_channels=C3_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.M4 = paddle.nn.Conv2D(in_channels=C4_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.M5 = paddle.nn.Conv2D(in_channels=C5_size, out_channels=
            feature_size, kernel_size=1, stride=1, padding=0)
        self.body = paddle.nn.Sequential(Body(feature_size))
        self.P6 = paddle.nn.Conv2D(in_channels=feature_size, out_channels=
            feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_1 = paddle.nn.ReLU()
        self.P7_2 = paddle.nn.Conv2D(in_channels=feature_size, out_channels
            =feature_size, kernel_size=3, stride=2, padding=1)
        self.P3_att = GLFIM(feature_size, feature_size)
        self.P5_att = GLFIM(feature_size, feature_size)
        self.P7_att = GLFIM(feature_size, feature_size)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        M3 = self.M3(C3)
        M4 = self.M4(C4)
        M5 = self.M5(C5)
        P3_f, P4_f, P5_f = self.body([M3, M4, M5])
        P6_f = self.P6(P5_f)
        P7 = self.P7_1(P6_f)
        P7_f = self.P7_2(P7)
        P3_f = self.P3_att(P3_f)
        P5_f = self.P5_att(P5_f)
        P7_f = self.P7_att(P7_f)
        return [P3_f, P4_f, P5_f, P6_f, P7_f]
