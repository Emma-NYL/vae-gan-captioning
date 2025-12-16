# eval_test.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt

from utils import (
    load_sd_vae,
    load_dcgan_discriminator,
    vae_latents,
    dcgan_mid_features,
    get_sd_input_transform,
)
from vae_gan_fusion import (
    FusionCaptioner,
    Vocab,
    CIFAR100WeakCaption,
    collate,
    set_seed,
)

# ------- greedy decoding，用来在 test set 上生成 caption -------
def greedy_decode(model, vis_vec, vocab, max_len=15):
    model.eval()
    bos, eos = 1, 2
    y = torch.tensor([[bos]], device=vis_vec.device)  # [1,1]
    for _ in range(max_len):
        logits = model(vis_vec, y)           # [1,T,V]
        next_token = logits[0, -1].argmax().item()
        y = torch.cat(
            [y, torch.tensor([[next_token]], device=y.device)], dim=1
        )
        if next_token == eos:
            break
    tokens = y[0].tolist()[1:-1]            # 去掉 bos/eos
    words = [vocab.itos[t] for t in tokens]
    return " ".join(words)

# ------- 把 [-1,1] 的 CIFAR 图像还原成 [0,1]，方便可视化 -------
def denorm_for_vis(x: torch.Tensor) -> torch.Tensor:
    # x: [3,H,W] in [-1,1]
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    return x

def show_samples(sample_imgs, sample_caps, num_show=5):
    num = min(num_show, len(sample_caps), sample_imgs.size(0))
    if num == 0:
        return
    plt.figure(figsize=(3 * num, 3))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        img = denorm_for_vis(sample_imgs[i]).permute(1, 2, 0).cpu()  # [H,W,3]
        plt.imshow(img)
        plt.title(sample_caps[i], fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../PyTorch-GAN/data")
    ap.add_argument("--img_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="ckpt/fusion_captioner_epochX.pt",
    )
    ap.add_argument("--use_gan", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="debug 用，>0 时只跑前 N 个 batch",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    # ---------- 1. 加载 checkpoint ----------
    ckpt = torch.load(args.model, map_location=device)
    vocab_list = ckpt["vocab"]
    vocab = Vocab(["dummy"])             # 占位，再覆盖
    vocab.itos = vocab_list
    vocab.stoi = {w: i for i, w in enumerate(vocab_list)}

    use_gan_in_ckpt = ckpt.get("use_gan", False)
    if args.use_gan and not use_gan_in_ckpt:
        print(
            "[WARN] ckpt 是不用 GAN 训练的，但你传了 --use_gan；"
            "我还是会创建带 GAN 的特征。"
        )
    elif (not args.use_gan) and use_gan_in_ckpt:
        print(
            "[WARN] ckpt 是用 GAN 训练的，但你没加 --use_gan；"
            "结果会不匹配。建议两边保持一致。"
        )

    vis_dim = 132 if use_gan_in_ckpt else 4
    model = FusionCaptioner(
        vocab_size=len(vocab.itos),
        vis_dim=vis_dim,
        hidden=512,
        num_vis_tokens=4,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---------- 2. test set DataLoader ----------
    full_test = CIFAR100WeakCaption(
        args.data_root, train=False, img_size=args.img_size
    )
    # 不需要再划分，整个 CIFAR100 test 就是我们的 test set
    def collate_fn(batch):
        return collate(batch, vocab)

    test_loader = DataLoader(
        full_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ---------- 3. 加载 VAE + （可选）GAN 判别器 ----------
    vae = load_sd_vae(device)
    D = load_dcgan_discriminator(device=device) if args.use_gan else None
    if args.use_gan and D is None:
        print(
            "[WARN] --use_gan 但没加载到判别器权重，将退化为 VAE-only。"
        )
    if (D is None) and use_gan_in_ckpt:
        print(
            "[WARN] ckpt 是用 GAN 训练的，但现在评估时没用 GAN 特征，"
            "loss 可能会偏。"
        )

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # ---------- 4. 在 test set 上计算平均 loss，并存几张示例 ----------
    total_loss, steps = 0.0, 0
    example_captions = []        # 文本示例
    example_imgs = None          # 对应的图像（只取第一个 batch）

    print("[INFO] start evaluating on CIFAR-100 test set ...")
    for step, (imgs, y_inp, y_tgt) in enumerate(test_loader, start=1):
        imgs = imgs.to(device)
        y_inp = y_inp.to(device)
        y_tgt = y_tgt.to(device)

        with torch.no_grad():
            z_vae = vae_latents(vae, imgs)   # [B,4]
            if args.use_gan and D is not None:
                z_d = dcgan_mid_features(D, imgs)
                vis_vec = torch.cat([z_vae, z_d], dim=1)
            else:
                vis_vec = z_vae

            logits = model(vis_vec, y_inp)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y_tgt.reshape(-1),
            )
            total_loss += loss.item()
            steps += 1

            # 第一个 batch：保存几张图片和对应 caption，用来展示
            if example_imgs is None:
                # 保存原始 tensor（还在 [-1,1]，之后再反归一化）
                example_imgs = imgs.detach().cpu()
                num_show = min(5, imgs.size(0))
                for j in range(num_show):
                    cap = greedy_decode(
                        model, vis_vec[j : j + 1], vocab
                    )
                    example_captions.append(cap)

        if args.max_steps > 0 and steps >= args.max_steps:
            print(f"[DEBUG] reached max_steps={args.max_steps}, stop early.")
            break

    avg_loss = total_loss / max(1, steps)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n[Test] average loss: {avg_loss:.4f}, perplexity: {ppl:.2f}\n")

    print("Sample test captions:")
    for i, c in enumerate(example_captions, 1):
        print(f"  #{i}: {c}")

    # ---------- 5. 画出示例图片 + caption ----------
    if example_imgs is not None and len(example_captions) > 0:
        show_samples(example_imgs, example_captions, num_show=5)

if __name__ == "__main__":
    main()

