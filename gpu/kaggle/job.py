"""Kaggle GPU job for the MTG strength repo (see docs/GPU.md).

Pushed with `kaggle kernels push -p gpu/kaggle`; it runs on a Kaggle T4 with
no browser open. It clones the repo, installs the requirements, runs
COMMANDS from the repo root, and copies every file under out/ and models/
that the commands created or changed into /kaggle/working/results/, keeping
repo-relative paths. Fetch them with
`kaggle kernels output <user>/mtg-strength-gpu -p kaggle_out` and copy
kaggle_out/results/* over the repo.

Edit BRANCH and COMMANDS before each push. The Hugging Face token comes from
the Kaggle secret HF_TOKEN (attach it once to this notebook in the web UI,
Add-ons > Secrets), needed for gated encoders such as EmbeddingGemma.
"""

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/ContrattoAlessandro/mtg_strenght"
BRANCH = "main"
WORK = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"
CACHE = os.path.join(WORK, "cache")   # hidden-state caches: big, not worth downloading
COMMANDS = [
    "python scripts/phase2.py train --encoder google/embeddinggemma-2 --top-k 4"
    " --name ft_embeddinggemma2_k4_gpu --max-epochs 6 --seeds 0 1 2"
    f" --device cuda --cache-dir {CACHE} --save-dir models/phase2",
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
    sh(f"{sys.executable} -m pip install -q -U transformers sentence-transformers", cwd=repo)
    failed = None
    for cmd in COMMANDS:
        try:
            sh(cmd, cwd=repo)
        except subprocess.CalledProcessError as e:
            failed = e
            break
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
