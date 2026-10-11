# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = (
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "DySnakeConv",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
)


# ============================================================
# Dynamic Snake Convolution
# ============================================================


class DSConv(nn.Module):
    """Dynamic Snake Convolution.

    morph:
        0 -> snake sampling mainly along x direction
        1 -> snake sampling mainly along y direction
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        morph=0,
        kernel_size=3,
        extend_scope=1.0,
        if_offset=True,
    ):
        super().__init__()

        assert morph in [0, 1]
        assert kernel_size % 2 == 1, "kernel_size should be odd."

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.morph = morph
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset

        # ----------------------------------------------------
        # predict offsets
        #
        # first K channels  : y offsets
        # second K channels : x offsets
        # ----------------------------------------------------

        self.offset_conv = nn.Conv2d(
            in_ch,
            2 * kernel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.offset_bn = nn.BatchNorm2d(2 * kernel_size)

        # ----------------------------------------------------
        # convolution after snake sampling
        # ----------------------------------------------------

        if morph == 0:
            # sampled feature:
            # B C H (W*K)

            self.dsc_conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(1, kernel_size),
                stride=(1, kernel_size),
                padding=0,
                bias=False,
            )

        else:
            # sampled feature:
            # B C (H*K) W

            self.dsc_conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(kernel_size, 1),
                stride=(kernel_size, 1),
                padding=0,
                bias=False,
            )

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    # ========================================================
    # cumulative snake offset
    # ========================================================

    def _accumulate_offset(self, offset):
        """offset: B K H W.

        Instead of allowing each sampling point to move
        independently, offsets accumulate from the center.

        This is the core "snake-like continuity" constraint.
        """
        _b, k, _h, _w = offset.shape

        center = k // 2

        values = [None] * k

        # center point fixed
        values[center] = torch.zeros_like(offset[:, center])

        # positive direction
        for i in range(1, center + 1):
            values[center + i] = values[center + i - 1] + offset[:, center + i]

        # negative direction
        for i in range(1, center + 1):
            values[center - i] = values[center - i + 1] + offset[:, center - i]

        return torch.stack(values, dim=1)

    # ========================================================
    # build sampling grid
    # ========================================================

    def _deform_feature(self, x, offset):

        b, _c, h, w = x.shape

        k = self.kernel_size
        center = k // 2

        # ----------------------------------------------------
        # split offsets
        # ----------------------------------------------------

        offset_y = offset[:, :k]
        offset_x = offset[:, k:]

        # ----------------------------------------------------
        # snake continuity
        # ----------------------------------------------------

        if self.if_offset:
            if self.morph == 0:
                offset_y = self._accumulate_offset(offset_y)

                offset_y = offset_y * self.extend_scope

            else:
                offset_x = self._accumulate_offset(offset_x)

                offset_x = offset_x * self.extend_scope

        else:
            offset_y = torch.zeros_like(offset_y)

            offset_x = torch.zeros_like(offset_x)

        # ----------------------------------------------------
        # original coordinates
        # ----------------------------------------------------

        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )

        yy = yy.unsqueeze(0).expand(b, -1, -1)

        xx = xx.unsqueeze(0).expand(b, -1, -1)

        grids = []

        # ====================================================
        # Horizontal snake
        # ====================================================

        if self.morph == 0:
            for i in range(k):
                # fixed horizontal positions
                x_pos = xx + (i - center)

                # dynamically deform vertical position
                y_pos = yy + offset_y[:, i]

                # normalize coordinates to [-1, 1]

                if w > 1:
                    x_norm = 2.0 * x_pos / (w - 1) - 1.0
                else:
                    x_norm = torch.zeros_like(x_pos)

                if h > 1:
                    y_norm = 2.0 * y_pos / (h - 1) - 1.0
                else:
                    y_norm = torch.zeros_like(y_pos)

                grid = torch.stack([x_norm, y_norm], dim=-1)

                grids.append(grid)

            # B H (W*K) 2

            grid = torch.cat(grids, dim=2)

        # ====================================================
        # Vertical snake
        # ====================================================

        else:
            for i in range(k):
                # fixed vertical positions
                y_pos = yy + (i - center)

                # dynamically deform horizontal position
                x_pos = xx + offset_x[:, i]

                if w > 1:
                    x_norm = 2.0 * x_pos / (w - 1) - 1.0
                else:
                    x_norm = torch.zeros_like(x_pos)

                if h > 1:
                    y_norm = 2.0 * y_pos / (h - 1) - 1.0
                else:
                    y_norm = torch.zeros_like(y_pos)

                grid = torch.stack([x_norm, y_norm], dim=-1)

                grids.append(grid)

            # B (H*K) W 2

            grid = torch.cat(grids, dim=1)

        # ----------------------------------------------------
        # deformable sampling
        # ----------------------------------------------------

        x_deformed = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        return x_deformed

    # ========================================================
    # forward
    # ========================================================

    def forward(self, x):

        # predict offsets

        offset = self.offset_conv(x)

        offset = self.offset_bn(offset)

        # restrict offset range

        offset = torch.tanh(offset)

        # deform sampling

        x = self._deform_feature(x, offset)

        # snake convolution

        x = self.dsc_conv(x)

        x = self.bn(x)

        x = self.act(x)

        return x


# ============================================================
# Dynamic Snake Convolution fusion block
# ============================================================


class DySnakeConv(nn.Module):
    """┌─ Normal 3x3 Conv ─┐ Input ───────┼─ DSConv-X ────────┼─ Concat └─ DSConv-Y ────────┘ | 1x1 | Output.
    """

    def __init__(
        self,
        c1,
        c2,
        k=3,
    ):
        super().__init__()

        # ----------------------------------------------------
        # normal convolution branch
        # ----------------------------------------------------

        self.conv_normal = Conv(c1, c2, k=3, s=1)

        # ----------------------------------------------------
        # horizontal snake branch
        # ----------------------------------------------------

        self.conv_x = DSConv(
            in_ch=c1,
            out_ch=c2,
            morph=0,
            kernel_size=k,
        )

        # ----------------------------------------------------
        # vertical snake branch
        # ----------------------------------------------------

        self.conv_y = DSConv(
            in_ch=c1,
            out_ch=c2,
            morph=1,
            kernel_size=k,
        )

        # ----------------------------------------------------
        # fusion
        # ----------------------------------------------------

        self.fuse = Conv(c2 * 3, c2, k=1, s=1)

    def forward(self, x):

        x0 = self.conv_normal(x)

        x1 = self.conv_x(x)

        x2 = self.conv_y(x)

        x = torch.cat([x0, x1, x2], dim=1)

        x = self.fuse(x)

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class LTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, eps):
        """Y = a * log(clamp(tanh(exp(b*x)), min=eps)).

        Forward uses in-place operations to reduce memory usage.
        Only x, a, b are saved for backward.
        """
        # Only save necessary tensors
        ctx.save_for_backward(x, a, b)
        ctx.eps = eps

        # -------------------------------------------------
        # Original:
        # y = exp(b*x)
        # y = tanh(y)
        # y = clamp(y, min=eps)
        # y = a * log(y)
        #
        # Use the same tensor buffer repeatedly.
        # -------------------------------------------------

        y = x * b

        y.exp_()
        y.tanh_()
        y.clamp_min_(eps)
        y.log_()
        y.mul_(a)

        return y

    @staticmethod
    def backward(ctx, grad_output):

        x, a, b = ctx.saved_tensors
        eps = ctx.eps

        # =================================================
        # Recompute forward quantities
        # =================================================

        z = b * x
        u = torch.exp(z)
        t = torch.tanh(u)

        # clamp:
        #
        # f(z) = log(clamp(tanh(exp(z)), eps))
        #
        # If tanh(exp(z)) < eps,
        # clamp makes the function constant,
        # therefore derivative = 0.
        active = t >= eps

        # =================================================
        # derivative
        #
        # f(z) = log(tanh(exp(z)))
        #
        # u = exp(z)
        #
        # df/dz =
        #
        # u * (1 - tanh(u)^2) / tanh(u)
        #
        # =================================================

        t_safe = torch.clamp(t, min=eps)

        dfdz = u * (1.0 - t * t) / t_safe

        # exp(z) may overflow for a very large positive z.
        # In this region tanh(exp(z)) == 1 and derivative == 0.
        dfdz = torch.where(torch.isfinite(dfdz), dfdz, torch.zeros_like(dfdz))

        dfdz = torch.where(active, dfdz, torch.zeros_like(dfdz))

        # =================================================
        # gradient w.r.t. x
        #
        # y = a * f(bx)
        #
        # dy/dx = a * f'(bx) * b
        # =================================================

        grad_x = grad_output * a * dfdz * b

        # =================================================
        # gradient w.r.t. a
        #
        # dy/da = f(bx)
        # =================================================

        f = torch.log(t_safe)

        grad_a = (grad_output * f).sum()

        # =================================================
        # gradient w.r.t. b
        #
        # dy/db = a * f'(bx) * x
        # =================================================

        grad_b = (grad_output * a * dfdz * x).sum()

        return grad_x, grad_a, grad_b, None


class LTE(nn.Module):
    """Low-memory implementation of:.

        LTE(x) = a * log(tanh(exp(b*x)))

    with numerical clamp:

        LTE(x) =
            a * log(clamp(tanh(exp(b*x)), min=eps))

    The mathematical function is unchanged.
    """

    def __init__(self, a=1.0, b=1.0, learnable_a=True, learnable_b=True, eps=1e-6):
        super().__init__()

        self.eps = eps

        if learnable_a:
            self.a = nn.Parameter(torch.tensor(float(a)))
        else:
            self.register_buffer("a", torch.tensor(float(a)))

        if learnable_b:
            self.b = nn.Parameter(torch.tensor(float(b)))
        else:
            self.register_buffer("b", torch.tensor(float(b)))

    def forward(self, x):

        return LTEFunction.apply(x, self.a, self.b, self.eps)


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    # default_act = nn.SiLU()  # default activation
    default_act = LTE(a=0.5, b=0.9, learnable_a=False, learnable_b=False)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=None):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
