#!/usr/bin/env python
"""
collect_data.py  (v12 – entropy‑annealed PPO teacher, 148‑D obs)

Outputs
-------
ppo_doorkey_keyshaped.zip   # near‑deterministic PPO
data/traj_*.npz             # 10k episodes, obs (T, 148)
"""

import argparse, gymnasium as gym, numpy as np
from pathlib import Path
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import FilterObservation
from minigrid.wrappers import RGBImgPartialObsWrapper

# ---------- hyper‑params ---------------------------------------------
BOOT_ENV, BOOT_STEPS = "MiniGrid-DoorKey-6x6-v0", 1_500_000
MAIN_ENV, MAIN_STEPS = "MiniGrid-DoorKey-8x8-v0", 5_000_000
NUM_ENVS             = 6
LOG_EVERY            = 100_000
TRAJ_EPISODES        = 10_000
NET_ARCH             = [256, 256]
SAVE_PATH            = "ppo_doorkey_keyshaped.zip"
ENT_START, ENT_END   = 0.01, 1e-5          # entropy anneal range
DATA_DIR             = Path("data"); DATA_DIR.mkdir(exist_ok=True)
TOTAL_STEPS          = BOOT_STEPS + MAIN_STEPS
# ---------------------------------------------------------------------

# ----------------------- custom wrappers -----------------------------
class FlattenDir(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img = env.observation_space["image"]
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(img.shape[0]*img.shape[1]*3 + 1,), dtype=np.float32
        )
    def observation(self, obs):
        return np.concatenate([obs["image"].ravel().astype(np.float32),
                               [float(obs["direction"])]])

class KeyDoorShaping(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env); self.reset_flags()
    def reset_flags(self): self.got_key=False; self.door_bonus=False
    def reset(self, **kw): self.reset_flags(); return super().reset(**kw)
    def reward(self, r):
        if not self.got_key and self.unwrapped.carrying is not None:
            r += .4; self.got_key=True
        if not self.door_bonus:
            for cell in self.unwrapped.grid.grid:
                if getattr(cell,"is_open",False):
                    r += .2; self.door_bonus=True; break
        return r
# ---------------------------------------------------------------------

def make_env(env_id):
    def _t():
        e = gym.make(env_id, render_mode=None)
        e = KeyDoorShaping(e)
        e = RGBImgPartialObsWrapper(e, tile_size=1)
        e = FilterObservation(e, ["image","direction"])
        e = FlattenDir(e)
        return Monitor(e)
    return _t

def success_rate(model, env_id, n=50):
    env=make_env(env_id)(); wins=0
    for _ in range(n):
        obs,_=env.reset(); done=ret=0
        while not done:
            a,_=model.predict(obs,deterministic=True)
            obs,r,term,trunc,_=env.step(a)
            ret+=r; done=term or trunc
        wins+=ret>=1.0
    env.close(); return wins/n

# -------------------- training loop ----------------------------------
def train(env_id, steps, model=None, start_step=0):
    vec = SubprocVecEnv([make_env(env_id) for _ in range(NUM_ENVS)])

    if model is None:
        model = PPO(
            "MlpPolicy", vec,
            policy_kwargs=dict(net_arch=NET_ARCH),
            n_steps=2048, batch_size=NUM_ENVS*2048,
            learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95,
            ent_coef=ENT_START,          # initial entropy
            verbose=0
        )
    else:
        model.set_env(vec)

    for st in range(0, steps, LOG_EVERY):
        global_step = start_step + st
        progress = global_step / TOTAL_STEPS
        model.ent_coef = ENT_START + (ENT_END - ENT_START) * progress

        model.learn(LOG_EVERY, reset_num_timesteps=False)
        print(f"[{env_id}] @{global_step+LOG_EVERY:,}  "
              f"ent {model.ent_coef:.5f}  "
              f"succ {success_rate(model,env_id):.2%}")
    vec.close(); return model
# ---------------------------------------------------------------------

def dump_trajs(model):
    env=make_env(MAIN_ENV)(); print("\n=== Dumping trajectories ===")
    for ep in trange(TRAJ_EPISODES):
        obs,_=env.reset(seed=ep)
        obs_l,act_l,rw_l,done=[],[],[],False
        while not done:
            a,_=model.predict(obs,deterministic=True)
            obs_l.append(obs); act_l.append(a)
            obs,r,term,trunc,_=env.step(a)
            rw_l.append(r); done=term or trunc
        rtg=np.cumsum(rw_l[::-1])[::-1]
        np.savez_compressed(DATA_DIR/f"traj_{ep:05d}.npz",
                            obs=np.stack(obs_l),act=np.array(act_l,np.int8),
                            rtg=rtg.astype(np.float32))
    env.close(); print("✅ Dump done.")

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--dump_only", action="store_true")
    args=pa.parse_args()

    if args.dump_only:
        model = PPO.load(SAVE_PATH)
    else:
        print("\n=== Phase 1: 6×6 bootstrap ===")
        model = train(BOOT_ENV, BOOT_STEPS)

        print("\n=== Phase 2: 8×8 fine‑tune ===")
        model = train(MAIN_ENV, MAIN_STEPS, model, BOOT_STEPS)

        model.save(SAVE_PATH)
        print(f"✅ saved teacher → {SAVE_PATH}")
        print("Final deterministic success (200 eps):",
              f"{success_rate(model, MAIN_ENV,200):.2%}")

    dump_trajs(model)

if __name__=="__main__":
    main()
