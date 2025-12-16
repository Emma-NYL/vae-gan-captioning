# vae_gan_fusion.py
import os, math, random, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from collections import Counter
from typing import List, Tuple

from utils import (
    load_sd_vae, load_dcgan_discriminator,
    vae_latents, dcgan_mid_features,
    get_sd_input_transform, fused_visual_vector
)

# -----------------------------
# 一些超参
# -----------------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="../PyTorch-GAN/data",
                    help="root 下应包含 cifar-100-python/")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)   # macOS: 0
    ap.add_argument("--save_dir", type=str, default="ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gan", action="store_true", help="若有判别器权重，建议打开")

    # ✔ 默认不限制训练（0 表示完整训练）
    ap.add_argument("--max_steps", type=int, default=0,
                    help="最多训练多少个 batch（0 = 不限制，仅调试用）")

    return ap.parse_args()

# -----------------------------
# “弱监督”文本：用类名→模板句
# -----------------------------
TEMPLATES = [
    "a photo of a {}",
    "this is a {}",
    "a small {}",
    "a {} on the scene",
    "the picture of a {}",
]

class CIFAR100WeakCaption(torch.utils.data.Dataset):
    def __init__(self, root: str, train: bool, img_size: int):
        self.base = datasets.CIFAR100(root=root, train=train, download=False)
        self.names = self.base.classes
        self.tf = get_sd_input_transform(img_size)  # [-1,1]
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, y = self.base[idx]
        name = self.names[y].replace("_", " ")
        cap = random.choice(TEMPLATES).format(name)
        return self.tf(img), cap

# 简易词表/编码器
class Vocab:
    def __init__(self, texts: List[str], min_freq: int = 1):
        cnt = Counter()
        for t in texts:
            cnt.update(t.lower().split())
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + [w for w, c in cnt.items() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(w, 3) for w in s.lower().split()]

def build_vocab(loader, max_samples=6000):
    texts = []
    for _, caps in loader:
        texts += list(caps)
        if len(texts) >= max_samples: break
    return Vocab(texts)

def collate(batch, vocab: Vocab):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    seqs = []
    for c in caps:
        ids = [1] + vocab.encode(c) + [2]
        seqs.append(torch.tensor(ids))
    maxlen = max(len(s) for s in seqs)
    y = torch.full((len(seqs), maxlen), 0, dtype=torch.long)
    for i, s in enumerate(seqs):
        y[i, :len(s)] = s
    return imgs, y[:, :-1], y[:, 1:]

# -----------------------------
# 融合 + LSTM Captioner
# -----------------------------
class FusionCaptioner(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 512, vis_dim: int = 132, num_vis_tokens: int = 4):
        super().__init__()
        self.num_vis_tokens = num_vis_tokens
        self.proj = nn.Linear(vis_dim, hidden)
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, vis_vec, y_inp):
        vis_token = self.proj(vis_vec).unsqueeze(1)
        vis_tokens = vis_token.repeat(1, self.num_vis_tokens, 1)
        txt = self.embed(y_inp)
        x = torch.cat([vis_tokens, txt], dim=1)
        h, _ = self.lstm(x)
        logits = self.out(h)
        return logits[:, self.num_vis_tokens:, :]

# -----------------------------
# 训练/验证
# -----------------------------
def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    args = get_args()
    set_seed(args.seed)

    # 自动选择设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    # 数据
    full_train = CIFAR100WeakCaption(args.data_root, train=True, img_size=args.img_size)
    n = len(full_train)
    n_train = int(0.6*n); n_val = int(0.2*n)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    train_idx, val_idx = idx[:n_train], idx[n_train:n_train+n_val]

    tmp_loader = DataLoader(full_train, batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(train_idx))
    vocab = build_vocab(tmp_loader)
    print("vocab size:", len(vocab.itos))

    collate_fn = lambda batch: collate(batch, vocab)
    train_loader = DataLoader(full_train, batch_size=args.batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              collate_fn=collate_fn)
    val_loader = DataLoader(full_train, batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(val_idx),
                            collate_fn=collate_fn)

    # 特征提取器
    vae = load_sd_vae(device)
    D = load_dcgan_discriminator(device=device) if args.use_gan else None

    # 模型
    vis_dim = 132 if D is not None else 4
    model = FusionCaptioner(len(vocab.itos), 512, vis_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch}] start")
        model.train()
        tot_loss = 0; steps = 0

        for imgs, y_inp, y_tgt in train_loader:

            # ✔ 只有 max_steps > 0 才限制
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"[DEBUG] reached max_steps={args.max_steps}, stop training loop.")
                break

            imgs = imgs.to(device)
            with torch.no_grad():
                z_vae = vae_latents(vae, imgs)
                vis_vec = torch.cat([z_vae, dcgan_mid_features(D, imgs)], dim=1) \
                           if D else z_vae

            y_inp = y_inp.to(device); y_tgt = y_tgt.to(device)
            logits = model(vis_vec, y_inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            tot_loss += loss.item(); steps += 1; global_step += 1
            print(f"[epoch {epoch}] [step {global_step}] loss={loss.item():.4f}")

        print(f"[epoch {epoch}] train loss: {tot_loss/max(1,steps):.4f}")

        # ---------------- validation ----------------
        model.eval(); v_tot=0; v_steps=0
        with torch.no_grad():
            for imgs, y_inp, y_tgt in val_loader:
                if args.max_steps > 0 and v_steps >= min(10, args.max_steps):
                    break
                imgs = imgs.to(device)
                z_vae = vae_latents(vae, imgs)
                vis_vec = torch.cat([z_vae, dcgan_mid_features(D, imgs)], dim=1) \
                           if D else z_vae
                y_inp = y_inp.to(device); y_tgt = y_tgt.to(device)
                logits = model(vis_vec, y_inp)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
                v_tot += loss.item(); v_steps += 1

        print(f"[epoch {epoch}]   val loss: {v_tot/max(1,v_steps):.4f}")

        torch.save({
            "model": model.state_dict(),
            "vocab": vocab.itos,
            "use_gan": args.use_gan,
            "img_size": args.img_size,
        }, os.path.join(args.save_dir, f"fusion_captioner_epoch{epoch}.pt"))

        # ✔ debug 模式下提前停止
        if args.max_steps > 0 and global_step >= args.max_steps:
            print("[INFO] reached max_steps, stop training (debug mode).")
            break


if __name__ == "__main__":
    main()
