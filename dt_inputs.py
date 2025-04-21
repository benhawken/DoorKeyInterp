#!/usr/bin/env python
"""
dt_inputs_check.py – verify Decision‑Transformer offline alignment.

Target accuracy ≥ 85 %.
"""

import random
from pathlib import Path
import numpy as np, torch
from train_dt import DecisionTransformer

SEQ_LEN, PAD = 15, 7
DATA_DIR     = Path("data")

def make_slice(rtg, obs, act, t):
    start = max(0, t - SEQ_LEN + 1)
    return (rtg[start:t+1], obs[start:t+1], act[start:t])

def tokens(rtg_seq, state_seq, hist):
    rtg_arr = np.zeros(SEQ_LEN, np.float32)
    rtg_arr[:len(rtg_seq)] = rtg_seq

    state_arr = np.zeros((SEQ_LEN, 148), np.float32)
    state_arr[:len(state_seq)] = state_seq

    act_in = [PAD] + list(hist) + [PAD]*(SEQ_LEN-1-len(hist))

    return (torch.from_numpy(rtg_arr)[None],
            torch.from_numpy(state_arr)[None],
            torch.tensor(act_in, dtype=torch.long)[None])

# load model
model = DecisionTransformer()
model.load_state_dict(torch.load("dt_doorkey.pth", map_location="cpu"))
model.eval()

# evaluate
files = random.sample(list(DATA_DIR.glob("traj_*.npz")), 200)
hits = tot = 0
for f in files:
    d = np.load(f)
    obs, act, rtg = d["obs"], d["act"], d["rtg"]
    for t in range(len(act)):
        rseq, sseq, hseq = make_slice(rtg, obs, act, t)
        r, s, a = tokens(rseq, sseq, hseq)
        with torch.no_grad():
            pred = model(r, s, a)[0, -1].argmax().item()
        hits += pred == act[t]; tot += 1

print(f"offline one‑step accuracy = {hits/tot*100:.1f}%")
