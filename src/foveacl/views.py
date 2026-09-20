"""GPU view generation for FoveaCL (docs/propose/propose04.md section 4).

All views are produced by one resampling routine. Image coordinates are normalised to [-1, 1] on both axes. An output
pixel with coordinates (u, v) in [-1, 1]^2 samples the image at

    p = c + s * g(u, v)                     c (B, 2) fixation / crop centre, s (B, 2) half-extent per axis (sign = flip)

with g(u, v) = (u, v) for uniform sampling and, for cortical magnification M(r) = 1 / (r + a),

    g(u, v) = a_rel * (exp(rho * w) - 1) * (u, v) / rho,   rho = |(u, v)|,  w = ln(1 + 1 / a_rel)

where a_rel = a / R is the foveation constant in units of the field radius, so that rho = 1 maps to the field radius.
Points outside the image are mid-grey. Aliasing is limited by sampling from an average-pooled pyramid whose level
follows the local sampling stride.
"""

import math

import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def base_grid(n, a_rel, device):
    """Returns g (n, n, 2) and the radial stride d|g|/drho (n, n), both relative to uniform sampling."""
    t = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n * 2 - 1
    v, u = torch.meshgrid(t, t, indexing="ij")
    uv = torch.stack([u, v], -1)
    if math.isinf(a_rel):
        return uv, torch.ones(n, n, device=device)
    rho = uv.norm(dim=-1).clamp_min(1e-6)
    w = math.log1p(1 / a_rel)
    r = a_rel * torch.expm1(rho * w)
    return uv * (r / rho)[..., None], a_rel * w * torch.exp(rho * w)


def resample(img, centre, scale, n, a_rel=math.inf, levels=4):
    """img (B, 3, S, S) float in [0, 1]; centre, scale (B, 2) in normalised image units. Returns (B, 3, n, n)."""
    B, _, S, _ = img.shape
    g, stride = base_grid(n, a_rel, img.device)
    grid = centre[:, None, None, :] + scale[:, None, None, :] * g[None]
    # source pixels per output pixel, (B, n, n)
    lod = torch.log2((stride[None] * scale.abs().amax(1)[:, None, None] * S / n).clamp_min(1))
    lod = lod.clamp_max(levels - 1)
    x = img - 0.5                                   # zero padding of grid_sample becomes mid-grey
    out = 0
    for l in range(levels):
        wgt = (1 - (lod - l).abs()).clamp_min(0)
        if wgt.any():
            out = out + wgt[:, None] * F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if l < levels - 1:
            x = F.avg_pool2d(x, 2)
    return out + 0.5


def fixation_params(B, device, sigma_c=0.3, f_range=(0.5, 1.0), flip=True):
    """Fixation centre ~ N(0, sigma_c) per axis, field radius f ~ U(f_range), both in image half-side units."""
    f = torch.empty(B, 1, device=device).uniform_(*f_range)
    c = (torch.randn(B, 2, device=device) * sigma_c)
    lim = 1 - 0.5 * f
    c = torch.maximum(torch.minimum(c, lim), -lim)
    s = f.expand(B, 2).clone()
    if flip:
        s[:, 0] *= torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
    return c, s


def crop_params(B, device, area=(0.08, 1.0), ratio=(3 / 4, 4 / 3), flip=True):
    """RandomResizedCrop expressed as (centre, half-extent)."""
    s = torch.empty(B, device=device).uniform_(*area)
    r = torch.exp(torch.empty(B, device=device).uniform_(math.log(ratio[0]), math.log(ratio[1])))
    w, h = (s * r).sqrt().clamp_max(1), (s / r).sqrt().clamp_max(1)
    half = torch.stack([w, h], 1)
    c = (torch.rand(B, 2, device=device) * 2 - 1) * (1 - half)
    if flip:
        half[:, 0] *= torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
    return c, half


def centre_params(B, device):
    return torch.zeros(B, 2, device=device), torch.ones(B, 2, device=device)


def gaussian_blur(x, sigma, kmax):
    """Per-sample separable Gaussian blur. x (B, C, H, W), sigma (B,) in pixels, kernel size 2 * kmax + 1."""
    B, C, H, W = x.shape
    t = torch.arange(-kmax, kmax + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t[None] / sigma.clamp_min(1e-3)[:, None]) ** 2)
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)             # (B*C, K)
    x = x.reshape(1, B * C, H, W)
    x = F.conv2d(F.pad(x, (kmax, kmax, 0, 0), mode="reflect"), k[:, None, None, :], groups=B * C)
    x = F.conv2d(F.pad(x, (0, 0, kmax, kmax), mode="reflect"), k[:, None, :, None], groups=B * C)
    return x.reshape(B, C, H, W)


def color_jitter(x, strength=0.8, hue=0.2, p=0.8, p_gray=0.2):
    """SimCLR colour distortion, fixed order brightness, contrast, saturation, hue, then random grayscale."""
    B, dev = x.shape[0], x.device
    apply = (torch.rand(B, 1, 1, 1, device=dev) < p).float()
    fac = lambda: 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * strength
    gray = lambda z: (0.299 * z[:, 0:1] + 0.587 * z[:, 1:2] + 0.114 * z[:, 2:3])
    y = (x * fac()).clamp(0, 1)
    m = gray(y).mean((2, 3), keepdim=True)
    y = ((y - m) * fac() + m).clamp(0, 1)
    gy = gray(y)
    y = ((y - gy) * fac() + gy).clamp(0, 1)
    # hue: rotation of the chroma plane in YIQ
    th = (torch.rand(B, device=dev) * 2 - 1) * hue * 2 * math.pi
    cos, sin = th.cos(), th.sin()
    to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = cos, -sin, sin, cos
    M = torch.linalg.inv(to_yiq)[None] @ rot @ to_yiq[None]
    y = torch.einsum("bij,bjhw->bihw", M, y).clamp(0, 1)
    x = apply * y + (1 - apply) * x
    g = (torch.rand(B, 1, 1, 1, device=dev) < p_gray).float()
    return g * gray(x).expand(-1, 3, -1, -1) + (1 - g) * x


def normalize(x):
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor(STD, device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x - mean) / std


class ViewMaker:
    """mode 'crop' = RandomResizedCrop with uniform sampling; mode 'fix' = fixation views with foveation a_rel."""

    def __init__(self, n=112, mode="fix", a_rel=math.inf, sigma_c=0.3, f_min=0.5, p_blur=0.5):
        self.n, self.mode, self.a_rel, self.sigma_c, self.f_min, self.p_blur = n, mode, a_rel, sigma_c, f_min, p_blur

    @torch.no_grad()
    def train_view(self, img, acuity_sigma=0.0):
        """img (B, S, S, 3) uint8 on GPU -> normalised (B, 3, n, n)."""
        x = img.permute(0, 3, 1, 2).float() / 255
        B = x.shape[0]
        if acuity_sigma > 0:
            x = gaussian_blur(x, torch.full((B,), acuity_sigma, device=x.device), math.ceil(3 * acuity_sigma))
        if self.mode == "crop":
            c, s = crop_params(B, x.device)
            x = resample(x, c, s, self.n)
        else:
            c, s = fixation_params(B, x.device, self.sigma_c, (self.f_min, 1.0))
            x = resample(x, c, s, self.n, self.a_rel)
        x = color_jitter(x)
        smax = 2.0 * self.n / 224
        sig = torch.empty(B, device=x.device).uniform_(0.1, smax)
        sig = torch.where(torch.rand(B, device=x.device) < self.p_blur, sig, torch.full_like(sig, 1e-3))
        x = gaussian_blur(x, sig, math.ceil(3 * smax))
        return normalize(x).contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def test_view(self, img):
        """Single central fixation covering the whole image (the NSD viewing condition)."""
        x = img.permute(0, 3, 1, 2).float() / 255
        c, s = centre_params(x.shape[0], x.device)
        x = resample(x, c, s, self.n, self.a_rel if self.mode == "fix" else math.inf)
        return normalize(x).contiguous(memory_format=torch.channels_last)
