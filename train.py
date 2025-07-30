import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# ---------------------------
# Dataset for your format
# ---------------------------
class ConversationDataset(Dataset):
    def __init__(self, json_path, expect_dim=1024, assert_dim=True):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.source, self.target = [], []
        for conv in data:
            embs = [torch.tensor(u["embedding"], dtype=torch.float32) for u in conv]
            if assert_dim:
                for e in embs:
                    assert e.numel() == expect_dim, f"Expected {expect_dim}, got {e.numel()}"
            for i in range(len(embs) - 1):
                self.source.append(embs[i])
                self.target.append(embs[i + 1])

        print(f"✅ Loaded {len(self.source)} pairs from {len(data)} conversations.")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

# ---------------------------
# Liquid Neural Net
# ---------------------------
class LiquidLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * 0.02)
        self.U = nn.Parameter(torch.randn(output_dim, output_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.act = nn.Tanh()

    def forward(self, x, prev_state=None):
        if prev_state is None:
            prev_state = torch.zeros(x.size(0), self.W.size(0), device=x.device)
        return self.act(x @ self.W.T + prev_state @ self.U.T + self.bias)

class LiquidNetwork(nn.Module):
    def __init__(self, in_dim=1024, h_dim=4000, out_dim=1024):
        super().__init__()
        self.l1 = LiquidLayer(in_dim, h_dim)
        self.l2 = LiquidLayer(h_dim, h_dim)
        self.l3 = LiquidLayer(h_dim, h_dim)
        self.l4 = LiquidLayer(h_dim, h_dim)
        self.l5 = nn.Linear(h_dim * 4, out_dim)

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        h4 = self.l4(h3)
        concat = torch.cat([h1, h2, h3, h4], dim=-1)
        return self.l5(concat)

# ---------------------------
# Training with per-step warm-up + cosine decay
# ---------------------------
def train_with_step_warmup_swa(
    json_path="conversations.json",
    checkpoint_dir="checkpoints",
    swa_path="liquid_network_swa.pth",
    batch_size=256,
    epochs=10,
    lr=1e-4,
    warmup_ratio=0.1,  # warm-up steps = warmup_ratio * total_steps
    amp=True,
    expect_dim=1024,
    num_workers=2,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ConversationDataset(json_path, expect_dim=expect_dim)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    model = LiquidNetwork(in_dim=expect_dim, h_dim=4000, out_dim=expect_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=(amp and device == "cuda"))

    total_steps = epochs * len(dataloader)
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"🧮 Total steps: {total_steps}, Warm-up steps: {warmup_steps}")

    swa_model = AveragedModel(model)
    checkpoint_paths = []

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        first_10_time_reported = False
        batch_start = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # ---- LR Warm-up + Cosine Decay ----
            if global_step < warmup_steps:
                lr_scale = global_step / float(max(1, warmup_steps))
                lr_current = lr_scale * lr
            else:
                progress = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr_current = 0.5 * lr * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_current

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(amp and device == "cuda")):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)

            if epoch == 0 and (batch_idx + 1) == 10 and not first_10_time_reported:
                elapsed = time.time() - batch_start
                avg_per_batch = elapsed / 10.0
                est_epoch_time = avg_per_batch * len(dataloader)
                est_total_time = est_epoch_time * epochs
                print(
                    f"⏱️ Estimated per-epoch: {est_epoch_time/60:.2f} min | "
                    f"Total: {est_total_time/3600:.2f} h"
                )
                first_10_time_reported = True

            # SWA update after warm-up
            if global_step >= warmup_steps:
                swa_model.update_parameters(model)

            global_step += 1

        avg_loss = total_loss / len(dataset)
        elapsed_min = (time.time() - epoch_start) / 60.0
        print(f"📘 Epoch [{epoch+1}/{epochs}] | LR: {lr_current:.6e} | Loss: {avg_loss:.6f} | Time: {elapsed_min:.2f} min")

        ckpt_path = os.path.join(checkpoint_dir, f"liquid_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        checkpoint_paths.append(ckpt_path)
        print(f"💾 Saved: {ckpt_path}")

    print("🧮 Updating BN stats for SWA model...")
    update_bn(dataloader, swa_model, device=device)
    torch.save(swa_model.state_dict(), swa_path)
    print(f"✅ Saved SWA model: {swa_path}")

    return model, swa_model


if __name__ == "__main__":
    model, swa_model = train_with_step_warmup_swa(
        json_path="train/dialogues_train.json",
        checkpoint_dir="checkpoints",
        swa_path="liquid_network_swa.pth",
        batch_size=256,
        epochs=10,
        lr=1e-4,
        warmup_ratio=0.2,
        amp=True,
        expect_dim=1024,
    )
