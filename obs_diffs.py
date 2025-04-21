import numpy as np, gymnasium as gym, random, re
from pathlib import Path
from collect_data import FlattenDir, KeyDoorShaping
from minigrid.wrappers import RGBImgPartialObsWrapper
from gymnasium.wrappers import FilterObservation

def make_env():
    e = gym.make("MiniGrid-DoorKey-8x8-v0")
    e = KeyDoorShaping(e)
    e = RGBImgPartialObsWrapper(e, tile_size=1)
    e = FilterObservation(e, ["image", "direction"])
    e = FlattenDir(e)
    return e

file = random.choice(list(Path("data").glob("traj_*.npz")))
idx  = int(re.search(r"(\d+)", file.stem).group(1))  # episode index as seed
d    = np.load(file); ob_rec, ac_rec = d["obs"], d["act"]

env = make_env(); obs_env,_ = env.reset(seed=idx)
for t, act in enumerate(ac_rec[:25]):
    diff = np.linalg.norm(obs_env - ob_rec[t])
    print(f"step {t:02d}  L2 diff {diff:.1f}")
    obs_env, _, term, trunc,_ = env.step(int(act))
    if term or trunc: break
env.close()
