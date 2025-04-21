#!/usr/bin/env python
"""
train_dt.py  –  Literature‑spec Decision‑Transformer (148‑D state)

Matches jbloomAus/DecisionTransformerInterpretability:
• context length 15 (45 tokens)
• d_model 128, 2 layers, 4 heads, FF 512, dropout 0.1
• LR 1e‑4 with cosine decay, 15 epochs, batch 64

Outputs
-------
dt_doorkey.pth    # trained DT weights
"""

import math, random, argparse, torch, torch.nn as nn, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# ---------- constants ----------------------------------------------------
DATA_DIR    = Path("data")
SEQ_LEN     = 15               # 45 tokens after stacking (R,S,A)
BATCHSIZE   = 64
EPOCHS      = 15
LR          = 1e-4
D_MODEL     = 128
N_LAYERS    = 2
N_HEADS     = 4
N_ACTIONS   = 7
PAD_TOKEN   = N_ACTIONS        # Embedding slot 7
# -------------------------------------------------------------------------

class TrajDataset(Dataset):
    """Random contiguous slices of length SEQ_LEN (wrap if traj shorter)."""
    def __init__(self, data_dir, seq_len):
        self.files = sorted(data_dir.glob("traj_*.npz"))
        self.S = seq_len
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        obs, act, rtg = d["obs"], d["act"], d["rtg"]
        T = len(act)
        if T < self.S:
            idxs = (np.arange(self.S) % T)
        else:
            s = random.randint(0, T - self.S)
            idxs = slice(s, s + self.S)
        return (torch.from_numpy(rtg[idxs]).float(),
                torch.from_numpy(obs[idxs]).float(),
                torch.from_numpy(act[idxs]).long())

def collate(b):
    r,o,a = zip(*b); return (torch.stack(r), torch.stack(o), torch.stack(a))

class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_emb = nn.Linear(148, D_MODEL)
        self.rtg_emb   = nn.Linear(1, D_MODEL)
        self.act_emb   = nn.Embedding(N_ACTIONS+1, D_MODEL)  # +PAD/BOS
        self.pos_emb   = nn.Parameter(torch.randn(1, SEQ_LEN*3, D_MODEL)/100)

        enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, 4*D_MODEL, dropout=0.1, batch_first=True)
        self.tfm = nn.TransformerEncoder(enc_layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, N_ACTIONS)

    def forward(self, rtg, state, act_in):
        """
        rtg, state, act_in: [B,S]
        Token sequence: (R,S,A‑1)×S  → predict current A
        """
        tok_r = self.rtg_emb(rtg.unsqueeze(-1))      # [B,S,D]
        tok_s = self.state_emb(state)                # [B,S,D]
        tok_a = self.act_emb(act_in)                 # [B,S,D]
        x = torch.stack((tok_r, tok_s, tok_a), 2).flatten(1,2)  # [B,3S,D]
        x = x + self.pos_emb[:, :x.size(1)]
        h = self.tfm(x)                               # [B,3S,D]
        return self.head(h[:, 2::3])                  # [B,S,N_ACT]

def main():
    pa = argparse.ArgumentParser(); pa.add_argument("--gpu", action="store_true")
    args = pa.parse_args()
    device = ("cuda" if args.gpu and torch.cuda.is_available() else
              "mps"  if args.gpu and torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    dl = DataLoader(TrajDataset(DATA_DIR, SEQ_LEN), batch_size=BATCHSIZE,
                    shuffle=True, num_workers=0, collate_fn=collate)
    model = DecisionTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    for ep in range(EPOCHS):
        model.train(); loss_sum = n = 0
        for rtg, obs, act in dl:
            rtg, obs, act = [x.to(device) for x in (rtg, obs, act)]
            pad = torch.full((act.size(0),1), PAD_TOKEN, device=device)
            act_in = torch.cat([pad, act[:,:-1]], 1)          # BOS shift
            logits = model(rtg, obs, act_in)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, N_ACTIONS), act.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); n += 1
        sched.step()
        print(f"Epoch {ep:02d}  loss {loss_sum/n:.4f}")
    torch.save(model.state_dict(), "dt_doorkey.pth")
    print("✅ Saved DT to dt_doorkey.pth")

if __name__ == "__main__":
    main()
