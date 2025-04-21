#!/usr/bin/env python
"""
train_dti_flat.py
-----------------
Retrain Decision‑Transformer (repo architecture) on 148‑D DoorKey data.

Needs:
  * local clone of DecisionTransformerInterpretability already patched
    to support state_embedding_type="flat".
Outputs:
  dt_dti_flat.pth
"""

import sys, pathlib, random, numpy as np, torch, gymnasium as gym
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm.auto import tqdm
import torch.nn.functional as F

# ── repo path ────────────────────────────────────────────────────────
repo_root = pathlib.Path(
    "/Users/benjaminhawken/Library/CloudStorage/OneDrive-Personal/AI Research/mechinterp-sprint/DecisionTransformerInterpretability"
)
sys.path.extend([str(repo_root), str(repo_root / "src")])

from models.trajectory_transformer import DecisionTransformer
from config import TransformerModelConfig, EnvironmentConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ imports ok | device =", device)

# ── configs ──────────────────────────────────────────────────────────
SEQ_LEN = 15          # timesteps kept from each trajectory
N_CTX   = 44          # 3*SEQ_LEN - 1

model_cfg = TransformerModelConfig(
    d_model = 128,
    n_heads = 4,
    d_mlp   = 512,
    n_layers = 2,
    n_ctx   = N_CTX,              # must satisfy (n_ctx-2)%3==0
    activation_fn = "gelu",
    state_embedding_type = "flat",
    time_embedding_type  = "embedding",
    device = device,
)

obs_space = gym.spaces.Box(0, 255, shape=(148,), dtype=np.float32)
act_space = gym.spaces.Discrete(7)

env_cfg = EnvironmentConfig(
    env_id = "MiniGrid-DoorKey-8x8-v0",
    observation_space = obs_space,
    action_space      = act_space,
    max_steps = 160,
    device     = device,
)

# ── dataset ──────────────────────────────────────────────────────────
class SliceDataset(Dataset):
    def __init__(self, folder="data", S=SEQ_LEN):
        self.files = [f for f in Path(folder).glob("traj_*.npz")
                      if np.load(f)["obs"].shape[1] == 148]
        self.S = S
        print(f"Using {len(self.files)} clean trajectories (obs=148)")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        obs, act, rtg = d["obs"], d["act"], d["rtg"]
        T = len(act)
        if T >= self.S:
            s = random.randint(0, T-self.S)
            sl = slice(s, s+self.S)
            obs, act, rtg = obs[sl], act[sl], rtg[sl]
        else:
            pad = self.S - T
            obs = np.concatenate([obs, np.tile(obs[-1:], (pad,1))], 0)
            act = np.concatenate([act, np.tile(act[-1:], pad)], 0)
            rtg = np.concatenate([rtg, np.tile(rtg[-1:], pad)], 0)
        return (torch.tensor(rtg).float(),
                torch.tensor(obs).float(),
                torch.tensor(act).long())

def collate(b): r,o,a = zip(*b); return torch.stack(r),torch.stack(o),torch.stack(a)
loader = DataLoader(SliceDataset(), batch_size=64, shuffle=True,
                    num_workers=0, collate_fn=collate)

# ── model & optimiser ────────────────────────────────────────────────
model = DecisionTransformer(env_cfg, model_cfg).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# ── training ─────────────────────────────────────────────────────────
for ep in range(15):
    model.train(); tot = cnt = 0
    for rtg, obs, act in tqdm(loader, desc=f"epoch {ep:02d}"):
        rtg, obs, act = [x.to(device) for x in (rtg, obs, act)]
        bs = act.size(0)

        # actions token: length S‑1 (no action for first timestep)
        act_tok = act[:, :-1].unsqueeze(-1)   # (B,14,1)
        rtg_tok = rtg.unsqueeze(-1)           # (B,15,1)
        tsteps  = (torch.arange(SEQ_LEN, device=device)
                   .unsqueeze(0).repeat(bs,1).unsqueeze(-1))  # (B,15,1)

        _, logits, _ = model.forward(
            states    = obs,      # (B,15,148)
            actions   = act_tok,  # (B,14,1)
            rtgs      = rtg_tok,  # (B,15,1)
            timesteps = tsteps,   # (B,15,1)
        )                         # logits (B,15,7)

        loss = F.cross_entropy(logits.reshape(-1,7), act.view(-1))
        optim.zero_grad(); loss.backward(); optim.step()
        tot += loss.item(); cnt += 1
    print(f"epoch {ep:02d} loss {tot/cnt:.4f}")

torch.save(model.state_dict(), "dt_dti_flat.pth")
print("✅ saved dt_dti_flat.pth")
