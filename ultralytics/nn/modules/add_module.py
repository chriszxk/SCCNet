import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from typing import Sequence, Tuple, Union, List
from einops import rearrange

import math
 
 
class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(AKConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
 
    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
 
    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
 
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
 
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
 
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
 
        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
 
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
 
        return out
 
    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number),indexing='xy')
 
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
 
    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),indexing='xy')
 
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
        return p_0
 
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
 
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
 
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
        # 根据实际情况调整
        index = index.clamp(min=0, max=x.shape[-1] - 1)
 
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
        return x_offset
 
    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)
 
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

def _as_wavelet(wavelet):
    """
    Convert a wavelet name or Wavelet object to a pywt.Wavelet object.

    Args:
        wavelet: PyWavelets Wavelet object or valid wavelet name.

    Returns:
        pywt.Wavelet
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)

    return wavelet


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Torch implementation of numpy.outer for 1D vectors.
    """
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])

    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)

    return a_mul * b_mul


def construct_2d_filt(
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> torch.Tensor:
    """
    Construct 2D wavelet filters.

    Filter order:
        LL, LH, HL, HH

    Returns:
        Tensor with shape [4, H, W].
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)

    filt = torch.stack(
        [ll, lh, hl, hh],
        dim=0,
    )

    return filt


def get_filter_tensors(
    wavelet,
    flip: bool,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert PyWavelets filters to PyTorch tensors.

    Args:
        wavelet:
            PyWavelets Wavelet object or wavelet name.
        flip:
            Whether to reverse the filter coefficients.
        device:
            Target device.
        dtype:
            Tensor data type.

    Returns:
        dec_lo, dec_hi, rec_lo, rec_hi
    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(
        filt: Sequence[float],
    ) -> torch.Tensor:

        if isinstance(filt, torch.Tensor):

            if flip:
                filt = filt.flip(-1)

            return filt.unsqueeze(0).to(
                device=device,
                dtype=dtype,
            )

        if flip:
            filt = filt[::-1]

        return torch.tensor(
            filt,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)

    return (
        dec_lo_tensor,
        dec_hi_tensor,
        rec_lo_tensor,
        rec_hi_tensor,
    )


def _get_pad(
    data_len: int,
    filt_len: int,
) -> Tuple[int, int]:
    """
    Compute wavelet padding size.

    Args:
        data_len:
            Length of input feature.
        filt_len:
            Length of wavelet filter.

    Returns:
        right_pad, left_pad
    """

    pad_right = (2 * filt_len - 3) // 2
    pad_left = (2 * filt_len - 3) // 2

    # Make odd input size compatible with stride-2 decomposition.
    if data_len % 2 != 0:
        pad_right += 1

    return pad_right, pad_left


def fwt_pad2(
    data: torch.Tensor,
    wavelet,
    mode: str = "replicate",
) -> torch.Tensor:
    """
    Padding for 2D discrete wavelet transform.

    Args:
        data:
            Input tensor [B, C, H, W].
        wavelet:
            Wavelet name or pywt.Wavelet.
        mode:
            Padding mode.

    Returns:
        Padded tensor.
    """
    wavelet = _as_wavelet(wavelet)

    pad_bottom, pad_top = _get_pad(
        data.shape[-2],
        len(wavelet.dec_lo),
    )

    pad_right, pad_left = _get_pad(
        data.shape[-1],
        len(wavelet.dec_lo),
    )

    data = F.pad(
        data,
        [
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        ],
        mode=mode,
    )

    return data


class DWT(nn.Module):
    """
    2D Discrete Wavelet Transform.
    """

    def __init__(
        self,
        dec_lo,
        dec_hi,
        wavelet="haar",
        level=1,
        mode="replicate",
    ):
        super().__init__()

        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.mode = mode

        # Register as buffers so they automatically move with model.cuda()
        self.register_buffer(
            "dec_lo",
            dec_lo.clone().detach(),
        )

        self.register_buffer(
            "dec_hi",
            dec_hi.clone().detach(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        b, c, h, w = x.shape

        level = self.level

        if level is None:
            level = pywt.dwtn_max_level(
                [h, w],
                self.wavelet,
            )

        wavelet_component: List[
            Union[
                torch.Tensor,
                Tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
            ]
        ] = []

        l_component = x

        dec_lo = self.dec_lo.to(
            device=x.device,
            dtype=x.dtype,
        )

        dec_hi = self.dec_hi.to(
            device=x.device,
            dtype=x.dtype,
        )

        dwt_kernel = construct_2d_filt(
            lo=dec_lo,
            hi=dec_hi,
        )

        # [4, K, K]
        # ->
        # [4*C, 1, K, K]
        dwt_kernel = dwt_kernel.repeat(
            c,
            1,
            1,
        )

        dwt_kernel = dwt_kernel.unsqueeze(
            dim=1
        )

        for _ in range(level):

            l_component = fwt_pad2(
                l_component,
                self.wavelet,
                mode=self.mode,
            )

            h_component = F.conv2d(
                l_component,
                dwt_kernel,
                stride=2,
                groups=c,
            )

            res = rearrange(
                h_component,
                "b (c f) h w -> b c f h w",
                f=4,
            )

            (
                l_component,
                lh_component,
                hl_component,
                hh_component,
            ) = res.split(1, dim=2)

            l_component = l_component.squeeze(2)

            wavelet_component.append(
                (
                    lh_component.squeeze(2),
                    hl_component.squeeze(2),
                    hh_component.squeeze(2),
                )
            )

        wavelet_component.append(
            l_component
        )

        return wavelet_component[::-1]


class IDWT(nn.Module):
    """
    2D Inverse Discrete Wavelet Transform.
    """

    def __init__(
        self,
        rec_lo,
        rec_hi,
        wavelet="haar",
        level=1,
        mode="constant",
    ):
        super().__init__()

        self.wavelet = _as_wavelet(wavelet)
        self.level = level
        self.mode = mode

        self.register_buffer(
            "rec_lo",
            rec_lo.clone().detach(),
        )

        self.register_buffer(
            "rec_hi",
            rec_hi.clone().detach(),
        )

    def forward(
        self,
        x,
        weight=None,
    ):

        l_component = x[0]

        _, c, _, _ = l_component.shape

        if weight is None:

            rec_lo = self.rec_lo.to(
                device=l_component.device,
                dtype=l_component.dtype,
            )

            rec_hi = self.rec_hi.to(
                device=l_component.device,
                dtype=l_component.dtype,
            )

            idwt_kernel = construct_2d_filt(
                lo=rec_lo,
                hi=rec_hi,
            )

            idwt_kernel = idwt_kernel.repeat(
                c,
                1,
                1,
            )

            idwt_kernel = idwt_kernel.unsqueeze(
                dim=1
            )

        else:

            idwt_kernel = torch.flip(
                weight,
                dims=[-1, -2],
            )

        filt_len = idwt_kernel.shape[-1]

        for c_pos, components in enumerate(
            x[1:]
        ):

            lh_component = components[0]
            hl_component = components[1]
            hh_component = components[2]

            l_component = torch.cat(
                [
                    l_component.unsqueeze(2),
                    lh_component.unsqueeze(2),
                    hl_component.unsqueeze(2),
                    hh_component.unsqueeze(2),
                ],
                dim=2,
            )

            l_component = rearrange(
                l_component,
                "b c f h w -> b (c f) h w",
            )

            l_component = F.conv_transpose2d(
                l_component,
                idwt_kernel,
                stride=2,
                groups=c,
            )

            pad_left = (2 * filt_len - 3) // 2
            pad_right = (2 * filt_len - 3) // 2
            pad_top = (2 * filt_len - 3) // 2
            pad_bottom = (2 * filt_len - 3) // 2

            if c_pos < len(x) - 2:

                pred_w = (
                    l_component.shape[-1]
                    - pad_left
                    - pad_right
                )

                next_w = x[c_pos + 2][0].shape[-1]

                pred_h = (
                    l_component.shape[-2]
                    - pad_top
                    - pad_bottom
                )

                next_h = x[c_pos + 2][0].shape[-2]

                if next_w != pred_w:
                    pad_right += 1

                    pred_w = (
                        l_component.shape[-1]
                        - pad_left
                        - pad_right
                    )

                    assert next_w == pred_w, (
                        "Wavelet padding error in width."
                    )

                if next_h != pred_h:
                    pad_bottom += 1

                    pred_h = (
                        l_component.shape[-2]
                        - pad_top
                        - pad_bottom
                    )

                    assert next_h == pred_h, (
                        "Wavelet padding error in height."
                    )

            if pad_top > 0:
                l_component = l_component[
                    ..., pad_top:, :
                ]

            if pad_bottom > 0:
                l_component = l_component[
                    ..., :-pad_bottom, :
                ]

            if pad_left > 0:
                l_component = l_component[
                    ..., pad_left:
                ]

            if pad_right > 0:
                l_component = l_component[
                    ..., :-pad_right
                ]

        return l_component


class WaveDown(nn.Module):
    """
    Wavelet-based downsampling for Ultralytics.

    Input:
        [B, c1, H, W]

    Output:
        [B, c2, H/2, W/2]

    Args:
        c1: input channels, automatically provided by Ultralytics
        c2: output channels, specified in YAML
        wavelet: wavelet type, default "haar"
    """

    def __init__(
        self,
        c1,
        c2,
        wavelet="haar",
    ):
        super().__init__()

        if c2 % 4 != 0:
            raise ValueError(
                f"WaveDown requires c2 divisible by 4, got c2={c2}"
            )

        # --------------------------------------------------
        # Wavelet filters
        # --------------------------------------------------
        dec_lo, dec_hi, _, _ = get_filter_tensors(
            wavelet=wavelet,
            flip=True,
            device="cpu",
            dtype=torch.float32,
        )

        self.dwt = DWT(
            dec_lo=dec_lo,
            dec_hi=dec_hi,
            wavelet=wavelet,
            level=1,
        )

        # --------------------------------------------------
        # Pixel-Unshuffle branch
        #
        # c1 -> c2/4
        # PixelUnshuffle(2):
        # c2/4 -> c2
        #
        # H,W -> H/2,W/2
        # --------------------------------------------------
        self.pixel_down = nn.Sequential(
            nn.Conv2d(
                c1,
                c2 // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

        # --------------------------------------------------
        # DWT:
        #
        # c1 -> 4*c1
        #
        # LL, LH, HL, HH
        # --------------------------------------------------
        self.subband_attn = nn.Sequential(
            nn.Conv2d(
                c1 * 4,
                c1 * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=c1 * 4,
                bias=True,
            ),

            nn.GroupNorm(
                c1 * 4,
                c1 * 4,
            ),

            nn.GELU(),

            nn.Conv2d(
                c1 * 4,
                c2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),

            nn.Sigmoid(),
        )

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        self.dwconv = nn.Conv2d(
            c2,
            c2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=c2,
            bias=True,
        )

        self.pwconv = nn.Conv2d(
            c2,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):

        # ------------------------------
        # Pixel-Unshuffle downsampling
        # ------------------------------
        x_down = self.pixel_down(x)

        # ------------------------------
        # Wavelet decomposition
        # ------------------------------
        ya, (yh, yv, yd) = self.dwt(x)

        # LL + LH + HL + HH
        wavelet_features = torch.cat(
            [ya, yh, yv, yd],
            dim=1,
        )

        # ------------------------------
        # Subband attention
        # ------------------------------
        attn = self.subband_attn(
            wavelet_features
        )

        # ------------------------------
        # Feature enhancement
        # ------------------------------
        x = x_down * attn + x_down

        # ------------------------------
        # Feature fusion
        # ------------------------------
        x = self.dwconv(x)
        x = self.pwconv(x)

        return x

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    x = torch.randn(
        2,
        32,
        128,
        128,
        device=device,
    )

    model = WaveDown(
        dim=32,
        wavelet="haar",
    ).to(device)

    with torch.no_grad():
        y = model(x)

    print("Device :", device)
    print("Input  :", x.shape)
    print("Output :", y.shape)