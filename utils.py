# utils.py  (UPDATED with key remap)
import os
import glob
import torch
import torch.nn as nn
from typing import Tuple
from diffusers import AutoencoderKL
from torchvision import transforms
from contextlib import nullcontext

# ----------------------------
# 预处理：给原始PIL图像/张量统一到 [-1,1]，分辨率到 img_size
# ----------------------------
def get_sd_input_transform(img_size: int = 256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),                     # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],      # -> [-1,1]
                             [0.5, 0.5, 0.5]),
    ])

# 反归一化（可视化用）
def denorm_to_uint8(x: torch.Tensor) -> torch.Tensor:
    # x: [-1,1] -> [0,1]
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    return (x * 255.0).clamp(0, 255).byte()

# ----------------------------
# Stable Diffusion VAE
# ----------------------------
def load_sd_vae(device: str = None) -> AutoencoderKL:
    """
    加载 SD v1-5 的 VAE。仅做特征提取，不参与训练。
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    vae.requires_grad_(False).eval().to(device)

    # ✅ 更省内存/更稳：切片 + 平铺
    try:
        vae.enable_slicing()
        vae.enable_tiling()
    except Exception:
        pass
    return vae

@torch.inference_mode()
def vae_latents(vae: AutoencoderKL, imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B,3,H,W] ，数值需在 [-1,1]
    返回：mean-pooled 的 VAE latent 向量 [B, 4]
    备注：为提速和稳健性，这里将输入统一降到 128x128 再编码。
    """
    # ✅ 防止CPU/MPS卡死：先降分辨率到 128x128
    imgs = torch.nn.functional.interpolate(imgs, size=(128, 128), mode="bilinear", align_corners=False)
    imgs = imgs.to(next(vae.parameters()).device)

    # ✅ 混合精度（CUDA/MPS用），CPU 自动忽略
    dev = imgs.device.type
    amp_ctx = torch.autocast(dev) if dev in ("cuda", "mps") else nullcontext()
    with amp_ctx:
        dist = vae.encode(imgs).latent_dist   # Normal(mu, sigma)
        z = dist.sample() * 0.18215           # SD 约定缩放
        z_vec = z.mean(dim=(2, 3))            # [B,4]
        return z_vec

# ----------------------------
# DCGAN 判别器：提中间特征
# 与你 dcgan.py 中的结构对齐（去掉最后线性层）
# ----------------------------
class DCGANDiscriminator(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        def block(i, o, bn=True):
            layers = [
                nn.Conv2d(i, o, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                layers.append(nn.BatchNorm2d(o, 0.8))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(channels, 16, bn=False),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )

    def forward(self, x):
        # x: [-1,1]  [B,3,H,W]
        return self.features(x)  # [B,128,H',W']

def _find_d_weight(default_dir: str) -> str:
    """
    在默认目录下自动寻找判别器权重文件。
    支持常见命名：D.pth, discriminator.pth, *D*.pth
    """
    if not os.path.isdir(default_dir):
        return ""
    candidates = []
    for pat in ["D.pth", "discriminator.pth", "*D*.pth", "*.pt", "*.pth"]:
        candidates += glob.glob(os.path.join(default_dir, pat))
    # 按修改时间排序，取最近的一个
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] if candidates else ""

def _remap_dcgan_keys(sd: dict) -> dict:
    """
    将 PyTorch-GAN 保存的 keys: model.* 映射到本实现的 features.* 结构。
    若原本就是 features.* 则原样返回。
    """
    # 已是 features.* 直接返回
    if any(k.startswith("features.") for k in sd.keys()):
        return sd

    # 没有 model.* 前缀也直接返回（让后续过滤决定）
    if not any(k.startswith("model.") for k in sd.keys()):
        return sd

    remapped = {}

    # Conv 层映射：block0/1/2/3
    conv_map = [
        (0,  "features.0.0"),  # 第一块 Conv
        (3,  "features.1.0"),
        (7,  "features.2.0"),
        (11, "features.3.0"),
    ]
    for idx, tgt in conv_map:
        w, b = f"model.{idx}.weight", f"model.{idx}.bias"
        if w in sd: remapped[f"{tgt}.weight"] = sd[w]
        if b in sd: remapped[f"{tgt}.bias"]   = sd[b]

    # BN 映射（首块无 BN）
    bn_map = [
        (6,  "features.1.3"),
        (10, "features.2.3"),
        (14, "features.3.3"),
    ]
    for idx, tgt in bn_map:
        for suf in ["weight","bias","running_mean","running_var","num_batches_tracked"]:
            k = f"model.{idx}.{suf}"
            if k in sd:
                remapped[f"{tgt}.{suf}"] = sd[k]

    return remapped

def load_dcgan_discriminator(
    weight_path: str = "",
    device: str = None,
    channels: int = 3,
) -> DCGANDiscriminator:
    """
    加载你训练好的 DCGAN 判别器，并切到 eval 模式（仅提特征）。
    weight_path 为空时，会自动在
      ../PyTorch-GAN/implementations/dcgan/saved_models/
    下搜索最近的 .pth / .pt
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    D = DCGANDiscriminator(channels=channels).to(device)
    D.eval().requires_grad_(False)

    if not weight_path:
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         "..", "PyTorch-GAN", "implementations", "dcgan", "saved_models")
        )
        weight_path = _find_d_weight(default_dir)

    if weight_path and os.path.isfile(weight_path):
        # ✅ 安全加载，仅权重；并做形状匹配过滤 + 键名重映射
        sd = torch.load(weight_path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        sd = _remap_dcgan_keys(sd)  # ← 关键：把 model.* 映射到 features.*

        own = D.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and hasattr(v, "size") and v.size() == own[k].size()}
        missing, unexpected = D.load_state_dict(sd, strict=False)
        print(f"[INFO] loaded D weights: matched={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[WARN] 未找到判别器权重，将仅使用未加载权重的特征提取（可运行但效果弱）。")

    return D

@torch.inference_mode()
def dcgan_mid_features(D: DCGANDiscriminator, imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: [B,3,H,W] in [-1,1]
    返回：判别器中间层的全局平均池化特征 [B,128]
    """
    h = D(imgs)                     # [B,128,h',w']
    z = h.mean(dim=(2, 3))          # [B,128]
    return z

# ----------------------------
# 融合：把 VAE 与 DCGAN 向量拼接
# ----------------------------
@torch.inference_mode()
def fused_visual_vector(
    vae: AutoencoderKL,
    D: DCGANDiscriminator,
    imgs: torch.Tensor
) -> torch.Tensor:
    """
    返回： [B, 4 + 128] 的视觉向量
    """
    z_vae = vae_latents(vae, imgs)          # [B,4]
    z_d   = dcgan_mid_features(D, imgs)     # [B,128]
    return torch.cat([z_vae, z_d], dim=1)   # [B,132]
