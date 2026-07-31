# DATA LOCATIONS — where everything actually lives (updated 2026-07-20)

Home (`/users/joeunyook`, VAST) has a **50 GB quota**. On 2026-07-19 it filled up
mid-campaign and killed 15 of the 18 P1_v4 jobs. To prevent a repeat, the two
big data directories were moved off home onto Lustre, with **symlinks left in
place** so every script path (`residuals/`, `archive/`) keeps working unchanged.

## The map

| path you use in the repo | is a | real location | filesystem / quota |
|---|---|---|---|
| `~/logit-probe/residuals/` | **symlink** | `/capstor/scratch/cscs/joeunyook/logit-probe/residuals/` | capstor scratch, 150 TB — moved 2026-07-20, 25 GB / 10,795 files at move time |
| `~/logit-probe/archive/` | **symlink** | `/capstor/store/cscs/swissai/a0133/joeunyook/logit-probe-archive/` | project store, 10 TB, **no purge** — moved 2026-07-20, 4.6 GB / 3,960 files (trust_v1) |
| `~/logit-probe/results/`, `responses/`, `results/P1_v4/`, `steer/P1_v4/`, `plots/P1_v4/` | real dirs | still in home (small: tens of MB–low GB) | VAST home, 50 GB |
| `~/logit-probe/P3_postcot/` | real dir | in home — **post-CoT campaign** (results/ + responses/ JSONs only, no residuals, ~200 MB max; seeds 0–39) | VAST home, 50 GB |
| `~/logit-probe/P3_smoke/` | real dir | smoke-test output only (2 sessions, 3 rounds) — never mix into P3_postcot | VAST home, 50 GB |
| HF model weights | env var | `HF_HOME=/capstor/store/cscs/swissai/a0107/models/joeunyook/hf_cache` (set inside the sbatch scripts) | a0107 store, 1 TB (89.7% full — watch this one) |
| micromamba env `ftdiag` | real dir | `/capstor/scratch/cscs/joeunyook/micromamba/envs/ftdiag` | capstor scratch |

Other big projects (GovSim, MoralGym_Storage, models, checkpoints, containers)
already live on `/capstor/scratch/cscs/joeunyook/` — they were never in home.

## Rules of thumb

- **Never `mv` the symlinked dirs back into home** — home cannot hold them.
- **capstor scratch has a purge policy** (untouched files can be cleaned).
  `residuals/` is safe while campaigns actively read/write it; when the paper
  freezes, copy the final residuals to the a0133 store next to the archive.
- The a0133 project store (`/capstor/store/cscs/swissai/a0133/joeunyook/`) is
  the long-term, no-purge destination — 10 TB, essentially empty.
- Caches: `~/.cache/uv` and `~/.cache/pip` were deleted (20 GB, pure caches,
  they regrow on demand). If home fills again, delete them again first.
- Stale mamba locks: if a job dies with `LockFile can't be set at
  ~/.cache/mamba/proc`, delete the stale entries:
  `find ~/.cache/mamba/proc -name '*.json' -mmin +30 -delete`
  (caused by processes killed while holding the registry, e.g. the quota crash).

## Quick health check

```bash
quota                      # VAST home line must stay well under 50 GB
du -h --max-depth=1 ~ | sort -rh | head   # find any new home hog
ls -ld ~/logit-probe/residuals ~/logit-probe/archive   # both must show -> symlink targets
```
