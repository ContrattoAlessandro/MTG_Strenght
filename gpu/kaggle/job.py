"""Kaggle GPU job for the MTG strength repo (see docs/GPU.md).

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
# Step C + D1/D2 of docs/RESEARCH_PLAN.md (job v8). Validation folds only.
FOLD = ("python scripts/phase2.py train --encoder google/embeddinggemma-2 --val-fold V1 V2 V3"
        " --max-epochs 8 --device cuda --cache-dir {cache}")
PROBE = ("python scripts/contamination_probe.py names --device cuda --emb-dir {cache}/names --encoders"
         " sentence-transformers/all-mpnet-base-v2 BAAI/bge-base-en-v1.5 google/embeddinggemma-2"
         " Qwen/Qwen3-Embedding-0.6B Qwen/Qwen3-Embedding-4B microsoft/harrier-oss-v1-0.6b"
         " nvidia/Nemotron-3.5-Embed-1B-BF16")
# One lane per GPU: lanes run in parallel (lane i sees only GPU i and gets its
# own cache dir, {cache}); the commands within a lane run in order, and a
# failed command does not stop the ones after it.
LANES = [
    [PROBE,
     f"{FOLD} --top-k 4 --seeds 0 1 2 --text-variant no_reminder --name d1_gemma_k4_norem",
     f"{FOLD} --top-k 8 --seeds 0 1 2 --name d2_gemma_k8"],
    [f"{FOLD} --top-k 4 --seeds 0 1 2 --name d2_gemma_k4 --save-dir models/phase2",
     *[f"python scripts/contamination_probe.py paraphrase --device cuda --cache-dir {{cache}}"
       f" --model-path models/phase2/d2_gemma_k4/{fold}_seed0.pt models/phase2/d2_gemma_k4/{fold}_seed1.pt"
       f" models/phase2/d2_gemma_k4/{fold}_seed2.pt --encoder-date 2025-01-31"
       f" --sets {' '.join(sets)} --tag d2_gemma_k4_{fold}"
       for fold, sets in [("V1", ["MKM", "OTJ", "MH3"]), ("V2", ["BLB", "DSK", "FDN"]),
                          ("V3", ["TDM", "FIN", "EOE"])]],
     f"{FOLD} --top-k -1 --llrd 0.9 --enc-lr 2e-5 --seeds 0 --name d2_gemma_full_lr2e-5",
     f"{FOLD} --top-k -1 --llrd 0.9 --enc-lr 5e-5 --seeds 0 --name d2_gemma_full_lr5e-5"],
]
# Data from a Kaggle dataset (dataset_sources in kernel-metadata.json) is
# mounted read-only under /kaggle/input/<slug>/, e.g. for Phase 3:
#   "python scripts/phase3.py train --device cuda --work-dir /kaggle/input/mtg-p3"
#   " --model-dir models --batch-size 4096"
RESULTS = "/kaggle/working/results"


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
    import torch
    n_gpu = max(torch.cuda.device_count(), 1)
    if len(LANES) > n_gpu:   # fewer GPUs than lanes: run the lanes one after another
        lanes = [[c for lane in LANES for c in lane]]
    else:
        lanes = LANES
    os.makedirs(RESULTS, exist_ok=True)
    procs = []
    for i, lane in enumerate(lanes):
        cache = f"{CACHE}{i}"
        script = "fail=0; " + " ".join(f"echo '$ {c.format(cache=cache)}'; {c.format(cache=cache)}"
                                       f" || {{ echo 'FAILED: {c.format(cache=cache)}'; fail=1; }};"
                                       for c in lane) + " exit $fail"
        log = open(os.path.join(RESULTS, f"lane{i}.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
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
