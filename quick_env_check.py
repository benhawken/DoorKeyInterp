import importlib.metadata as md, torch, transformer_lens as tl
print("torch           :", torch.__version__)
print("transformer‑lens:", md.version("transformer-lens"))
try:
    import torchtyping, typeguard
    print("torchtyping     :", torchtyping.__version__)
    print("typeguard       :", typeguard.__version__)
except Exception as e:
    print("version check error:", e)
# simple head‑ablation sanity
assert hasattr(tl.HookedTransformer, "hooks"), "TL ≥2.0 expected"


