import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

from utils import load_sd_vae, load_dcgan_discriminator, vae_latents, dcgan_mid_features, get_sd_input_transform
from vae_gan_fusion import FusionCaptioner, Vocab

# --------------------------
# 生成文本序列
# --------------------------
def greedy_decode(model, vis_vec, vocab, max_len=15):
    model.eval()
    bos = 1
    eos = 2

    y = torch.tensor([[bos]]).to(vis_vec.device)  # [1,1]

    for _ in range(max_len):
        logits = model(vis_vec, y)  # [1,T,V]
        next_token = logits[0, -1].argmax().item()
        y = torch.cat([y, torch.tensor([[next_token]]).to(y.device)], dim=1)

        if next_token == eos:
            break

    tokens = y[0].tolist()[1:-1]  # 去掉 bos/eos
    words = [vocab.itos[t] for t in tokens]
    return " ".join(words)

# --------------------------
# 主函数
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    # --------------------------
    # 1) 加载模型 checkpoint
    # --------------------------
    ckpt = torch.load(args.model, map_location=device)
    vocab_list = ckpt["vocab"]
    vocab = Vocab(["dummy"])  # 构造空 vocab 再覆盖
    vocab.itos = vocab_list
    vocab.stoi = {w: i for i, w in enumerate(vocab_list)}

    # captioner
    vis_dim = 132 if ckpt.get("use_gan", False) else 4
    model = FusionCaptioner(len(vocab.itos), vis_dim=vis_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --------------------------
    # 2) 加载特征模型
    # --------------------------
    vae = load_sd_vae(device)
    D = load_dcgan_discriminator(device=device) if args.use_gan else None

    # --------------------------
    # 3) 读取图片
    # --------------------------
    tf = get_sd_input_transform(args.img_size)
    img = Image.open(args.img).convert("RGB")
    img_tensor = tf(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # --------------------------
    # 4) 提取视觉特征
    # --------------------------
    with torch.no_grad():
        z_vae = vae_latents(vae, img_tensor)
        if args.use_gan and D is not None:
            z_d = dcgan_mid_features(D, img_tensor)
            vis_vec = torch.cat([z_vae, z_d], dim=1)
        else:
            vis_vec = z_vae

    # --------------------------
    # 5) 解码 Caption
    # --------------------------
    caption = greedy_decode(model, vis_vec, vocab)
    print("\n========================")
    print("Caption:")
    print(caption)
    print("========================\n")

if __name__ == "__main__":
    main()

