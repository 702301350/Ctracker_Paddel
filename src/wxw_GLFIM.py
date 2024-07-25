import paddle_aux
import paddle
import math


class h_sigmoid(paddle.nn.Layer):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = paddle.nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(paddle.nn.Layer):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(paddle.nn.Layer):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = paddle.nn.AdaptiveAvgPool2D(output_size=(None, 1))
        self.pool_w = paddle.nn.AdaptiveAvgPool2D(output_size=(1, None))
        mip = max(8, inp // reduction)
        self.conv1 = paddle.nn.Conv2D(in_channels=inp, out_channels=mip,
            kernel_size=1, stride=1, padding=0)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=mip)
        self.act = h_swish()
        self.conv_h = paddle.nn.Conv2D(in_channels=mip, out_channels=oup,
            kernel_size=1, stride=1, padding=0)
        self.conv_w = paddle.nn.Conv2D(in_channels=mip, out_channels=oup,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = tuple(x.shape)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(perm=[0, 1, 3, 2])
        y = paddle.concat(x=[x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = paddle_aux.split(x=y, num_or_sections=[h, w], axis=2)
        x_w = x_w.transpose(perm=[0, 1, 3, 2])
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = paddle.multiply(x=a_h, y=paddle.to_tensor(a_w))
        return out


class DwConv(paddle.nn.Layer):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, groups=None):
        super(DwConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = paddle.nn.Conv2D(in_channels=in_channels,
            out_channels=out_channels, kernel_size=kernel_size, stride=
            stride, padding=padding, groups=groups, bias_attr=True)

    def forward(self, x):
        result = self.depthwise(x)
        return result


class GLFIM(paddle.nn.Layer):
    def __init__(self, in_dim, ou_dim, reduction=32):
        super(GLFIM, self).__init__()
        self.mip = max(8, in_dim // reduction)

        self.qh = paddle.nn.Conv2D(in_channels=in_dim, out_channels=in_dim,kernel_size=1, stride=1, padding=0)
        self.kh = paddle.nn.Conv2D(in_channels=in_dim, out_channels=in_dim,kernel_size=1, stride=1, padding=0)
        self.pool_qh = paddle.nn.AdaptiveAvgPool2D(output_size=(None, 1))
        self.pool_kh = paddle.nn.AdaptiveAvgPool2D(output_size=(None, 1))
        self.fc_qh = paddle.nn.Linear(in_features=in_dim, out_features=self.mip)
        self.fc_kh = paddle.nn.Linear(in_features=in_dim, out_features=self.mip)

        self.qw = paddle.nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        self.kw = paddle.nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1, padding=0)
        self.pool_qw = paddle.nn.AdaptiveAvgPool2D(output_size=(1, None))
        self.pool_kw = paddle.nn.AdaptiveAvgPool2D(output_size=(1, None))
        self.fc_qw = paddle.nn.Linear(in_features=in_dim, out_features=self.mip)
        self.fc_kw = paddle.nn.Linear(in_features=in_dim, out_features=self.mip)

        self.fc_v = paddle.nn.Linear(in_features=in_dim, out_features=self.mip)
        self.fc_out = paddle.nn.Linear(in_features=self.mip, out_features=in_dim)

        self.softmax = paddle.nn.Softmax(axis=-1)
        self.local_branch = CoordAtt(in_dim, ou_dim, reduction=reduction)

        out_0 = paddle.create_parameter(shape=paddle.zeros(shape=[1]).shape,
            dtype=paddle.zeros(shape=[1]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(
            shape=[1])))
        out_0.stop_gradient = not True

        self.gamma = out_0
        self.fusion = paddle.nn.Conv2D(in_channels=2 * in_dim, out_channels=ou_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = tuple(x.shape)
        qh = self.qh(x)
        kh = self.kh(x)
        qw = self.qw(x)
        kw = self.kw(x)
        res = x * self.local_branch(x)
        v = res.view(n, c, h * w).transpose(perm=[0, 2, 1])
        v = self.fc_v(v)
        qh = self.pool_qh(qh).squeeze(axis=-1).transpose(perm=[0, 2, 1])
        qh = self.fc_qh(qh)
        kh = self.pool_kh(kh).squeeze(axis=-1).transpose(perm=[0, 2, 1])
        kh = self.fc_kh(kh).transpose(perm=[0, 2, 1])
        att_h = paddle.bmm(x=qh, y=kh)
        att_h = self.softmax(att_h)
        qw = self.pool_qw(qw).squeeze(axis=-2).transpose(perm=[0, 2, 1])
        qw = self.fc_qw(qw)
        kw = self.pool_kw(kw).squeeze(axis=-2).transpose(perm=[0, 2, 1])
        kw = self.fc_kw(kw).transpose(perm=[0, 2, 1])
        att_w = paddle.bmm(x=qw, y=kw)
        att_w = self.softmax(att_w)
        v = v.view(n, h, w, self.mip)
        v = v.view(n, h, w * self.mip)
        out = paddle.matmul(x=att_h, y=v)
        out = out.view(n, h, w, self.mip).transpose(perm=[0, 1, 3, 2])
        out = out.view(n, h * self.mip, w)
        out = paddle.matmul(x=out, y=att_w)
        out = out.view(n, self.mip, h, w)
        out = out.view(n, self.mip, h * w).transpose(perm=[0, 2, 1])
        out = self.fc_out(out).transpose(perm=[0, 2, 1])
        out = out.view(n, c, h, w)
        out = self.gamma * out
        fin_out = paddle.concat(x=(res, out), axis=1)
        fin_out = self.fusion(fin_out)
        return fin_out
