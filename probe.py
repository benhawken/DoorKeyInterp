"""
probe.py
========
Fit a balanced logistic‑regression probe that predicts whether the
Decision‑Transformer is carrying the key in MiniGrid‑DoorKey trajectories.

Main call
---------
    v_unit, info = train_has_key_probe(model, Path("data"), seq_len=15)

`v_unit`  : unit‑length direction (torch.Tensor, d_model)  
`info`    : dict with accuracy, |w|, n_samples, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Callable, Optional, Tuple, Dict, Any

import numpy as np
import torch
from tqdm.auto import tqdm as tqdm_auto
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score


# -------------------------------------------------------------------------
# Helper: build (X, y) dataset
# -------------------------------------------------------------------------
def _collect_dataset(
    model,
    data_dir: Path,
    seq_len: int,
    layer: int,
    hook_type: str,
    max_traj: int,
    win_before: int,
    win_after: int,
    device: str,
    *,
    iterator_wrapper: Optional[Callable[[Iterable], Iterable]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns balanced arrays:
        X : (N, d_model)  residual‑stream vectors at the chosen hook
        y : (N,)         labels (1 = has key)
    """
    X_pos, X_neg = [], []

    traj_paths = sorted(data_dir.glob("traj_*.npz"))[:max_traj]
    if iterator_wrapper is not None:
        traj_paths = iterator_wrapper(traj_paths)

    for tp in traj_paths:
        arr = np.load(tp)
        seq_here = min(seq_len, arr["obs"].shape[0])

        # --- locate key pickup via +0.4 reward spike ---------------------
        rtg_np = arr["rtg"][:seq_here]
        inst = np.empty_like(rtg_np)
        inst[:-1] = rtg_np[:-1] - rtg_np[1:]
        inst[-1] = rtg_np[-1]
        hits = np.where(inst > 0.3)[0]
        pickup_t = int(hits[0]) if len(hits) else None

        # --- first OPEN_DOOR attempt -------------------------------------
        door_hits = np.where(arr["act"][: seq_here - 1] == 2)[0]
        door_t = int(door_hits[0]) if len(door_hits) else seq_here - 1

        keep_min = max(0, (pickup_t or 0) - win_before)
        keep_max = min(seq_here - 1, door_t + win_after)
        if keep_max < keep_min:
            continue  # empty window

        # --- forward to get activations ----------------------------------
        obs = torch.tensor(arr["obs"][keep_min : keep_max + 1],
                           dtype=torch.float32, device=device)[None]
        acts = torch.tensor(arr["act"][keep_min : keep_max],
                            dtype=torch.long, device=device)[None][..., None]
        rtg = torch.tensor(arr["rtg"][keep_min : keep_max + 1],
                           dtype=torch.float32, device=device)[None][..., None]
        tt = torch.arange(keep_min, keep_max + 1, device=device)[None, :, None]

        toks = model.to_tokens(obs, acts, rtg, tt)
        with torch.no_grad():
            _, cache = model.transformer.run_with_cache(
                toks,
                names_filter=f"blocks.{layer}.{hook_type}",
            )
        resid = cache[f"blocks.{layer}.{hook_type}"][0].cpu().numpy()

        # add every state token in window
        for i_env in range(keep_max - keep_min + 1):
            tok_state = 3 * i_env + 1
            if tok_state >= resid.shape[0]:
                break
            has_key = (pickup_t is not None) and (keep_min + i_env) >= pickup_t
            (X_pos if has_key else X_neg).append(resid[tok_state])

    # balance dataset
    n_keep = min(len(X_pos), len(X_neg))
    rng = np.random.default_rng(0)
    X_pos = rng.permutation(np.asarray(X_pos))[:n_keep]
    X_neg = rng.permutation(np.asarray(X_neg))[:n_keep]
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate(
        [np.ones(n_keep, dtype=np.int64), np.zeros(n_keep, dtype=np.int64)], axis=0
    )
    return X, y


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def train_has_key_probe(
    model,
    data_dir: Path,
    seq_len: int,
    *,
    layer_probe: int = 1,
    hook_type: str = "hook_resid_pre",
    max_traj: int = 400,
    window_before: int = 2,
    window_after: int = 6,
    device: str = "cpu",
    progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Fit the probe and return (v_unit, info).
    """
    iterator = (
        (lambda it: tqdm_auto(it, desc="collecting probe data"))
        if progress
        else None
    )

    X, y = _collect_dataset(
        model,
        data_dir,
        seq_len,
        layer_probe,
        hook_type,
        max_traj,
        window_before,
        window_after,
        device,
        iterator_wrapper=iterator,
    )

    # ------------- Standardise + 5‑fold CV over C ------------------------
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-2, 2, 10),
            cv=5,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            n_jobs=-1,
            scoring="accuracy",
            refit=True,
            class_weight="balanced",
        ),
    ).fit(X, y)

    logreg = clf[-1]                       # final estimator
    w_np = logreg.coef_.squeeze()
    b = float(logreg.intercept_[0])
    w = torch.tensor(w_np, dtype=torch.float32)
    v_unit = w / w.norm()

    preds = logreg.predict(clf[:-1].transform(X))
    accuracy = accuracy_score(y, preds)
    logits = torch.tensor(logreg.decision_function(clf[:-1].transform(X)),
                          dtype=torch.float32)

    info: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "w": w,
        "b": b,
        "w_norm": float(w.norm()),
        "mean_x_norm": float(np.linalg.norm(X, axis=1).mean()),
        "mean_abs_logit": float(logits.abs().mean()),
        "n_samples": int(X.shape[0]),
        "layer": layer_probe,
        "hook_type": hook_type,
    }
    return v_unit, info
