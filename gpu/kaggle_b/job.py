"""Second-session copy of gpu/kaggle/job.py (kernel mtg-strength-gpu-b).

Kaggle GPU job for the MTG strength repo (see docs/GPU.md).

Pushed with `kaggle kernels push -p gpu/kaggle`; it runs on a Kaggle T4 with
no browser open. It clones the repo, installs the requirements, runs
LANES from the repo root (one lane per GPU, in parallel), and copies every
file under out/ and models/ that the commands created or changed into
/kaggle/working/results/, keeping repo-relative paths. Fetch them with
`kaggle kernels output <user>/mtg-strength-gpu -p kaggle_out` and copy
kaggle_out/results/* over the repo.

Edit BRANCH and LANES before each push. The Hugging Face token comes from
the Kaggle secret HF_TOKEN (attach it once to this notebook in the web UI,
Add-ons > Secrets), needed for gated encoders such as EmbeddingGemma.
"""

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/ContrattoAlessandro/mtg_strenght"
BRANCH = "research-queue"
WORK = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
CACHE = os.path.join(WORK, "cache")   # hidden-state caches: big, not worth downloading
# Second Kaggle session (runs alongside gpu/kaggle/job.py).
# b1 (done): lr screen, fold V3, seed 0: Qwen3-Emb-0.6B best 3e-5, Harrier-0.6B 1e-4
# (full FT fp16, ~320 s/epoch). b2 (done): Qwen3-Emb-4B LoRA r16 best 1e-4
# (~1450 s/epoch, 4.5x the 0.6B models).
# b4 (done): E1 stage 1, 1 seed x 3 folds: Qwen3-Emb-0.6B 0.649, Qwen3-4B LoRA
# 0.647, Harrier-0.6B 0.622 (dropped, > 0.01 behind), Gemma-2 top-12 0.635.
# b5 (done): Qwen3-4B seeds 1-2; E1 kept Gemma-2.
# Job b6 (seed 0 now, with the last ~1.5 h of the week; seeds 1-2 at the reset): F1 aux deck-value head and F2
# Commander head on the final recipe (Gemma-2 top-12, 10 epochs), 3 seeds.
GEMMA = ("python scripts/phase2.py train --encoder google/embeddinggemma-2 --val-fold V1 V2 V3"
         " --device cuda --cache-dir {cache} --top-k 12 --max-epochs 10 --seeds 0")
LANES = [[f"{GEMMA} --deck-values out/phase3/deck_values.csv --aux-w 0.5 --name f1_aux"],
         [f"{GEMMA} --target label --label-csv out/phase4/commander_labels_first.csv"
          " --name f2_commander"]]
# Data from a Kaggle dataset (dataset_sources in kernel-metadata.json) is
# mounted read-only under /kaggle/input/<slug>/, e.g. for Phase 3:
#   "python scripts/phase3.py train --device cuda --work-dir /kaggle/input/mtg-p3"
#   " --model-dir models --batch-size 4096"
RESULTS = "/kaggle/working/results"
USE_LANES_JSON = False   # True: take the lanes from gpu/kaggle/lanes.json on BRANCH


def sh(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main() -> None:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:   # no secret attached: public encoders still work
        print(f"no HF_TOKEN secret ({e}); gated models will fail", flush=True)
    sh("nvidia-smi")
    repo = os.path.join(WORK, "repo")
    sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {repo}")
    # torch with CUDA is preinstalled; requirements.txt leaves it as is.
    sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=repo)
    sh(f"{sys.executable} -m pip install -q -U transformers sentence-transformers peft bitsandbytes", cwd=repo)
    # Kaggle's preinstalled torchao (0.10) is too old for peft's LoRA import; we don't use it.
    sh(f"{sys.executable} -m pip uninstall -y -q torchao", cwd=repo)
    # gpu/kaggle/lanes.json on the cloned branch (e.g. from sweep_d3.py) replaces LANES;
    # only this file is uploaded by `kernels push`, so the JSON comes from git.
    lanes_json = os.path.join(repo, "gpu", "kaggle", "lanes.json")
    all_lanes = LANES
    if USE_LANES_JSON and os.path.exists(lanes_json):
        import json
        all_lanes = json.load(open(lanes_json))
    import torch
    n_gpu = max(torch.cuda.device_count(), 1)
    if len(all_lanes) > n_gpu:   # fewer GPUs than lanes: run the lanes one after another
        lanes = [[c for lane in all_lanes for c in lane]]
    else:
        lanes = all_lanes
    os.makedirs(RESULTS, exist_ok=True)
    procs = []
    for i, lane in enumerate(lanes):
        cache = f"{CACHE}{i}"
        script = "fail=0; " + " ".join(f"echo '$ {c.format(cache=cache)}'; {c.format(cache=cache)}"
                                       f" || {{ echo 'FAILED: {c.format(cache=cache)}'; fail=1; }};"
                                       for c in lane) + " exit $fail"
        log = open(os.path.join(RESULTS, f"lane{i}.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        print(f"lane {i} on GPU {i}: {len(lane)} commands", flush=True)
        procs.append((subprocess.Popen(script, shell=True, cwd=repo, env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log))
    failed = None
    for i, (p, log) in enumerate(procs):
        rc = p.wait()
        log.close()
        print(f"--- lane {i} exit {rc} ---", flush=True)
        print(open(log.name).read()[-20000:], flush=True)
        if rc and failed is None:
            failed = subprocess.CalledProcessError(rc, f"lane {i}")
    # Copy what changed even after a failure (logs, finished seeds).
    changed = subprocess.run("git ls-files --modified --others --exclude-standard out models;"
                             " git ls-files --others --ignored --exclude-standard out models",
                             shell=True, cwd=repo, capture_output=True, text=True).stdout.split()
    for rel in sorted(set(changed)):
        dst = os.path.join(RESULTS, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(repo, rel), dst)
        print(f"saved {rel}", flush=True)
    if failed:
        raise failed


if __name__ == "__main__":
    main()
