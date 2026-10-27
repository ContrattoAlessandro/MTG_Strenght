# Training on a free cloud GPU (Kaggle or Colab)

The fine-tunes (`phase2.py train`) and the deck model (`phase3.py train`) took
hours on 4 CPU cores. Alessandro's own GPU stays reserved for his other model,
so the free cloud GPUs are the way to speed them up. Both scripts take
`--device cuda`; the default is still `cpu`, so nothing changes locally.

## Recommendation: Kaggle, driven from the terminal

Kaggle is the simpler option for this project, because the whole loop runs
from the `kaggle` CLI: local Claude Code edits `gpu/kaggle/job.py`, pushes it,
checks the log, and downloads the results, with no browser tab to keep open.
Colab needs a person in the browser (cells, Drive authorisation, keeping the
tab alive), so it is the fallback: `gpu/colab_train.ipynb`.

| | Kaggle (free) | Colab (free) |
|---|---|---|
| GPU | T4 ×2, 15 GB each (P100 is retired; the job uses one T4) | T4, 15 GB, when available (not guaranteed at peak hours) |
| Quota | ~30 GPU h per week, resets Saturday; `kaggle quota` shows it | unpublished, varies; you can be refused a GPU |
| Session | up to 12 h, runs in the background after `kernels push` | up to ~12 h, stops after ~90 min idle or if the tab closes |
| Drive from local Claude Code | yes: `kaggle kernels push / status / logs / output` | no CLI; a person runs the notebook (VS Code extension is interactive too) |
| Repo | cloned by the job (the repo is public) | cloned by the first cell |
| Secrets | Kaggle secret `HF_TOKEN`, attached once in the web UI | Colab secret `HF_TOKEN` |
| Results out | `/kaggle/working/results` → `kaggle kernels output` | Google Drive folder, or `results.zip` download |
| Extra data in | private Kaggle dataset → `/kaggle/input/<slug>` | Drive |

Quota and session numbers change over time; check `kaggle quota` and the
Colab FAQ rather than trusting this table.

## What a job needs

- **Phase 2 fine-tunes**: only the repo. `out/phase2/cards.parquet` and
  `struct.parquet` are in git, and the encoder comes from Hugging Face
  (EmbeddingGemma is gated: the token's account must have accepted the
  licence). The top-k hidden-state cache is built on the GPU in minutes.
  Memory: top-4 fine-tuning of EmbeddingGemma-2 or Harrier-0.6B in float32 at
  batch 32 fits a 15 GB T4 comfortably.
- **Phase 3 deck model**: also needs the `<SET>.npz` arrays from
  `phase3_data.py` (built from the 17lands dumps, 0.5–2 GB of downloads per
  set). Build them locally on CPU (it is not GPU work), upload once as a private
  Kaggle dataset, and mount it:
  ```
  # work/p3/dataset-metadata.json: {"title": "mtg-p3", "id": "<user>/mtg-p3", "licenses": [{"name": "CC0-1.0"}]}
  kaggle datasets create -p work/p3          # later updates: kaggle datasets version -p work/p3 -m "..."
  ```
  then add `"<user>/mtg-p3"` to `dataset_sources` in
  `gpu/kaggle/kernel-metadata.json` and use the Phase 3 command shown in
  `job.py`. A larger `--batch-size` (e.g. 4096) uses the GPU better; scale
  `--lr` with care.
- **Scryfall bulk data** is only needed for `prep`, `embed.py` and Phase 0;
  `mtgstrength.scryfall.download_oracle_cards` fetches it inside the job if a
  command needs it.

## Kaggle: one-time setup

1. Kaggle account, phone-verified (needed for GPU and internet in notebooks).
2. Settings > API > *Create New Token*; put `kaggle.json` in `~/.kaggle/`
   (or `kaggle auth login`). `pip install kaggle`.
3. In `gpu/kaggle/kernel-metadata.json`, replace `KAGGLE_USERNAME` with your
   username. Push once: `kaggle kernels push -p gpu/kaggle`.
4. Open the notebook on kaggle.com > Add-ons > Secrets > add `HF_TOKEN`
   (Hugging Face read token) and tick it for this notebook, then push again.
   The attachment belongs to the notebook, so later pushes should keep it; if
   a log says `no HF_TOKEN secret`, re-tick it in the notebook's editor.

## Kaggle: each job (what local Claude Code does)

```
# 1. edit BRANCH (the job clones that branch, so push your code first) and COMMANDS
$EDITOR gpu/kaggle/job.py
# 2. run it on a T4
kaggle kernels push -p gpu/kaggle --accelerator NvidiaTeslaT4
# 3. follow it (status: queued / running / complete / error)
kaggle kernels status <user>/mtg-strength-gpu
kaggle kernels logs <user>/mtg-strength-gpu | tail -50
# 4. fetch the results and put them in the repo
kaggle kernels output <user>/mtg-strength-gpu -p kaggle_out
cp -r kaggle_out/results/. .
```

The job copies every file under `out/` and `models/` that the commands
created or changed (git-ignored ones too, e.g. `--save-dir models/phase2`)
into `results/`, with repo-relative paths, so step 4 drops them in place.
Several commands in `COMMANDS` run in order within one 12 h session; if one
fails, the job still saves what the earlier ones produced and then fails.
`kaggle_out/` is scratch; add it to `.git/info/exclude` or delete it.

The job's per-epoch checkpoints live in the session's scratch disk and are
lost when it ends, so keep each job well under 12 h (a three-seed Phase 2 run
is far below that on a T4).

## Colab

Open `gpu/colab_train.ipynb` in Colab
(<https://colab.research.google.com/github/ContrattoAlessandro/mtg_strenght/blob/main/gpu/colab_train.ipynb>),
choose a T4 runtime, add the `HF_TOKEN` secret, and run the cells. With
`SAVE_TO_DRIVE` the cache and checkpoints go to Drive, so a disconnect only
costs a restart of the cells; the run resumes from the last epoch. Results end
up in `MyDrive/mtg_strength_gpu/results.zip`; unzip it at the repo root.

## Notes

- GPU runs are not bit-identical to the CPU runs (different kernels), so
  compare GPU results with GPU results, or rerun the baseline on GPU too.
- Saved weights (`--save-dir`, `deck_model.pt`) are written so they load on a
  CPU-only machine, e.g. the web app.
- Contamination rules and the temporal split are unchanged: the GPU only
  makes the same runs faster.
