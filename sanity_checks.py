#!/usr/bin/env python
"""
sanity_checks.py  (secure‑random version)

 • Offline next‑action acc on 500 *random* trajectories (≤15 steps)
 • Online success‑rate on 20 episodes
   – temp  ∈ [0.5, 1.0]   (soft‑sampling)
   – RTG   ∈ [1.0, 1.4]
   – env seed randomised

 --vis  →  also saves sanity_hist.png  +  sanity_demo.mp4
"""

import sys, pathlib, numpy as np, torch, argparse, gymnasium as gym, secrets
from pathlib import Path
from tqdm.auto import tqdm

# --------------------------------------------------------------------
# secure RNG (never affected by random.seed / numpy seed)
sys_rng = secrets.SystemRandom()

# repo path & imports
repo = pathlib.Path(
    "/Users/benjaminhawken/Library/CloudStorage/OneDrive-Personal/AI Research/mechinterp-sprint/DecisionTransformerInterpretability"
)
sys.path.extend([str(repo), str(repo / "src")])

from models.trajectory_transformer import DecisionTransformer
from config import TransformerModelConfig, EnvironmentConfig
from minigrid.wrappers import RGBImgPartialObsWrapper
from gymnasium.wrappers import FilterObservation

device = "cpu"

# --------------------------------------------------------------------
# build DT model
cfg = TransformerModelConfig(
    d_model=128, n_heads=4, d_mlp=512, n_layers=2,
    n_ctx=44, activation_fn="gelu",
    state_embedding_type="flat", time_embedding_type="embedding",
    device=device,
)
obs_space = gym.spaces.Box(0, 255, (148,), np.float32)
act_space = gym.spaces.Discrete(7)
env_cfg = EnvironmentConfig("MiniGrid-DoorKey-8x8-v0",
                            observation_space=obs_space,
                            action_space=act_space,
                            max_steps=160,
                            device=device)
model = DecisionTransformer(env_cfg, cfg).to(device)
model.load_state_dict(torch.load("dt_dti_flat.pth", map_location=device))
model.eval()

# --------------------------------------------------------------------
# CLI
pa = argparse.ArgumentParser()
pa.add_argument("--vis", action="store_true",
                help="save histogram & rollout video")
args = pa.parse_args()

# --------------------------------------------------------------------
# offline accuracy helper
def offline_accuracy(model, data_path="data", n_files=500):
    pool = sorted(Path(data_path).glob("traj_*.npz"))
    files = sys_rng.sample(pool, min(n_files, len(pool)))
    print("sampled (offline) files:", [f.name for f in files[:3]])
    ok = tot = 0
    for f in tqdm(files, desc="offline"):
        d = np.load(f)
        obs, act, rtg = d["obs"], d["act"], d["rtg"]
        for t in range(min(15, len(act))):
            s0 = max(0, t - 14)
            S = torch.tensor(obs[s0:t+1], dtype=torch.float32)[None]
            R = torch.tensor(rtg[s0:t+1], dtype=torch.float32)[None, :, None]
            A = torch.tensor(act[s0:t], dtype=torch.long)[None, :, None]
            T = torch.arange(S.shape[1]).long()[None, :, None]
            with torch.no_grad():
                _, logits, _ = model.forward(S, A, R, T)
            ok += logits[0, -1].argmax().item() == int(act[t]); tot += 1
    return ok / tot

# --------------------------------------------------------------------
# env wrappers identical to data pipeline
class FlattenDir(gym.ObservationWrapper):
    def observation(self, obs):
        return np.concatenate([obs["image"].ravel().astype(np.float32),
                               [float(obs["direction"])]])

class KeyDoorShaping(gym.RewardWrapper):
    def reset(self, **kw):
        self.got_key = self.bonus = False
        return super().reset(**kw)
    def reward(self, r):
        if not self.got_key and self.unwrapped.carrying is not None:
            r += .4; self.got_key = True
        if not self.bonus:
            for cell in self.unwrapped.grid.grid:
                if getattr(cell, "is_open", False):
                    r += .2; self.bonus = True; break
        return r

def make_env(render=False):
    mode = "rgb_array" if render else None
    e = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode=mode)
    e = KeyDoorShaping(e)
    e = RGBImgPartialObsWrapper(e, tile_size=1)
    e = FilterObservation(e, ["image", "direction"])
    e = FlattenDir(e)
    return e

# --------------------------------------------------------------------
# online rollout helper
def run_episode(max_steps=160, return_frames=False):
    temp = sys_rng.uniform(0.5, 1.0)
    init_rtg = sys_rng.uniform(1.0, 1.4)
    if return_frames:
        print(f"ep0  temp={temp:.2f}  init_rtg={init_rtg:.2f}")

    env = make_env(render=return_frames)
    obs, _ = env.reset(seed=sys_rng.randrange(10_000))

    S, R, A, ret = [], [], [], init_rtg
    frames = []
    if return_frames:
        f = env.render();  frames.append(f) if f is not None else None

    for _ in range(max_steps):
        S.append(torch.tensor(obs, dtype=torch.float32))
        R.append(torch.tensor(ret, dtype=torch.float32))
        if len(S) > 15:  S, R = S[-15:], R[-15:]
        if len(A) > len(S) - 1: A = A[-(len(S) - 1):]

        states = torch.stack(S)[None].to(device)
        rtgs   = torch.stack(R)[None,:,None].to(device)
        actions= (torch.tensor(A, dtype=torch.long)[None,:,None]
                  .to(device))
        tsteps = (torch.arange(states.size(1)).long()[None,:,None]
                  .to(device))

        with torch.no_grad():
            _, logits, _ = model.forward(states, actions, rtgs, tsteps)
            probs = torch.softmax(logits[0, -1] / temp, -1).cpu().tolist()
            action = sys_rng.choices(range(7), weights=probs, k=1)[0]

        obs, r, term, trunc, _ = env.step(action)
        if return_frames:
            f = env.render(); frames.append(f) if f is not None else None
        A.append(action); ret += r
        if term or trunc: break
    env.close()
    return (ret, frames) if return_frames else (ret, None)

# --------------------------------------------------------------------
# run checks
acc = offline_accuracy(model)
print(f"offline one‑step accuracy: {acc:.1%}")

scores, frames = [], None
for ep in tqdm(range(20), desc="online"):
    ret, fr = run_episode(return_frames=args.vis and ep == 0)
    scores.append(ret)
    if fr is not None:
        frames = fr

succ_rate = np.mean(np.array(scores) >= 1.0)
print(f"online success‑rate (20 eps): {succ_rate:.1%}")

if acc < 0.90 or succ_rate < 0.70:
    sys.exit("❌ sanity checks failed")

# --------------------------------------------------------------------
# optional visualisation
if args.vis:
    import matplotlib.pyplot as plt, imageio.v2 as imageio

    plt.figure(figsize=(4,3))
    plt.hist(scores, bins=10, edgecolor="k")
    plt.axvline(1.0, color="r", linestyle="--", label="solve threshold")
    plt.xlabel("return"); plt.ylabel("count")
    plt.title("DT returns (20 eps)"); plt.legend()
    plt.tight_layout(); plt.savefig("sanity_hist.png")
    print("📊  saved histogram  → sanity_hist.png")

    if frames:
        imageio.mimsave("sanity_demo.mp4", frames, fps=8)
        print("🎬 saved rollout    → sanity_demo.mp4")

print("✅ sanity checks passed")
