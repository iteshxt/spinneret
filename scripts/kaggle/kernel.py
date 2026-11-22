"""Kaggle kernel: train one Spinneret config on a free Kaggle GPU (specs §5.10).

push.sh fills in RUN and ARGS and pushes this file as a script kernel. The private
dataset made by dataset.sh holds the code and the prepared data; Spider itself is
downloaded on the VM, as on Colab. Work happens in /tmp so the kernel output
(/kaggle/working) holds only the log and the best checkpoint.

The Hugging Face token would come from the Kaggle secret HF_TOKEN (then checkpoints go to
the Hub each epoch and the run resumes from there), but secrets are not readable in
CLI-pushed script kernels, so in practice the run trains with --no-hub and the best
checkpoint is returned as output, to be scored and uploaded from the laptop.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

RUN = "__RUN__"
ARGS = json.loads("__ARGS__")  # spinneret train arguments, e.g. ["--config", "configs/..."]

WORK, OUT = "/tmp/spinneret", "/kaggle/working"
LOG = open(f"{OUT}/{RUN}.log", "a", buffering=1)  # noqa: SIM115 — open for the whole run


def sh(cmd: str, env: dict | None = None) -> None:
    print("$", cmd, flush=True)
    LOG.write(f"$ {cmd}\n")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=WORK if os.path.isdir(WORK) else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, **(env or {})},
    )
    for line in proc.stdout:
        LOG.write(line)
        if '"event"' in line or "Error" in line:
            print(line, end="", flush=True)
    if proc.wait():
        raise SystemExit(f"command failed: {cmd}")


def find(name: str, marker: str) -> tuple[str, str] | None:
    """The dataset item ``name``: ("tar", path) if kept as an archive, ("dir", path) if
    Kaggle unpacked it (identified by ``marker`` inside it)."""
    for path in glob.glob(f"/kaggle/input/**/{name}.tar.gz", recursive=True):
        return "tar", path
    for path in glob.glob(f"/kaggle/input/**/{marker}", recursive=True):
        return "dir", path[: -len(marker)].rstrip("/")
    return None


code = find("spinneret-code", "spinneret/cli.py")
data = find("spinneret-data", "processed/stats.json")
if not code or not data:
    subprocess.run("find /kaggle/input -maxdepth 5 | head -50", shell=True)
    sys.exit(f"dataset not found (code={code}, data={data})")
print("code", code, "data", data, flush=True)
os.makedirs(f"{WORK}/data", exist_ok=True)
for (kind, path), dest in ((code, WORK), (data, f"{WORK}/data")):
    if kind == "tar":
        sh(f"tar -xzf {path} -C {dest}")
    else:
        shutil.copytree(path, dest, dirs_exist_ok=True)

hub = False
try:
    from kaggle_secrets import UserSecretsClient

    token = UserSecretsClient().get_secret("HF_TOKEN").strip()
    os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)
    path = os.path.expanduser("~/.cache/huggingface/token")
    with open(path, "w") as fh:
        fh.write(token)
    os.chmod(path, 0o600)
    from huggingface_hub import whoami

    me = whoami(token=token)  # verify before training; prints the account, never the token
    role = (me.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"HF_LOGIN_OK user={me['name']} role={role}", flush=True)
    hub = True
except Exception as exc:  # noqa: BLE001 — no usable secret: train without the Hub
    print(f"HF_LOGIN_FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)

if ARGS == ["--check-hf"]:  # verification run: report the login and stop
    print("SPINNERET_KAGGLE_DONE", RUN, "hub" if hub else "no-hub", flush=True)
    sys.exit(0 if hub else 1)

env = {
    "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"],
    "PYTHONUNBUFFERED": "1",
    "UV_LINK_MODE": "copy",
}
sh("python3 -m pip install --quiet uv", env)
sh("uv python install 3.12 && uv sync --extra train --quiet", env)
sh("uv run spinneret download", env)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", env)
train = ["uv", "run", "spinneret", "train", *ARGS, *(["--resume", "hub"] if hub else ["--no-hub"])]
sh(" ".join(train), env)

for best in glob.glob(f"{WORK}/checkpoints/*/best"):
    run_name = best.split("/")[-2]
    shutil.copytree(best, f"{OUT}/{run_name}/best", dirs_exist_ok=True)
    for f in glob.glob(f"{WORK}/checkpoints/{run_name}/*.json"):
        shutil.copy(f, f"{OUT}/{run_name}/")
print("SPINNERET_KAGGLE_DONE", RUN, flush=True)
