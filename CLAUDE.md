# Mini-GRP

Minimalist reimplementation of the Octo/GRP robotics policy from the MILA world models course. Fork of `milarobotlearningcourse/mini-grp`.

## Commands

```bash
# Training (local)
python mini-grp.py

# Training (SLURM cluster)
python mini-grp.py --multirun hydra/launcher=submitit_slurm

# Hydra config overrides
python mini-grp.py lr=3e-4 batch_size=128

# Named configs
python mini-grp.py --config-name=bridge-64-light

# Sync with upstream fork
./sync_upstream.sh

# Sim environment install (requires CUDA >=11.8, NVIDIA GPU)
bash sim-install

# Dataset creation
python create_mini_oxe_dataset.py

# Environment test
python test_environment.py
```

## Architecture

Three model approaches:

| Script | Model | Approach |
|---|---|---|
| `mini-grp.py` | GRP | Image patches + char-level text + goal image → transformer encoder → MLP → 7-DOF actions |
| `vla_model.py` | VLA | PaliGemma-3B frozen backbone + MLP action head → 7-DOF actions |
| `vit-64.py` | ViT baseline | ViT encoder for 64x64 images |

**GRP data flow:** Bridge dataset (HuggingFace) → 64x64 images encoded to [-1,1] → 8x8 patches → char-level text tokenization → transformer → continuous 7-DOF actions (3D pos, 3D rot, gripper), normalized by precomputed mean/std.

**Masked training:** GRP randomly masks goal text OR goal image during training for robustness (controlled by `policy.random_masking_enabled`).

## Configuration

- Hydra configs in `conf/`
- Default config: `conf/config.yaml` (transformer model, 64x64 images, batch_size=64, lr=3e-4)
- `bridge-64-light.yaml` — lighter training run
- `bridge-64-submitit.yaml` — SLURM submission
- `grp-mini.yaml` — minimal GRP config
- SLURM launcher: `conf/hydra/launcher/MY_2HRS.yaml` (A100, 32GB, 2hr timeout)

## Conventions

- Images: 64x64, normalized to [-1, 1]
- Text: character-level encoding (no tokenizer), vocab auto-detected
- Action space: 7-DOF continuous (3D position, 3D rotation, gripper)
- Experiment tracking: Weights & Biases
- `mini-grp/` subdirectory is a nested clone of upstream (reference only — do not edit)
- Notebooks (`mini-grp.ipynb`, `mini-vla.ipynb`) are interactive walkthroughs of the training pipelines
