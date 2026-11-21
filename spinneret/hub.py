"""Hugging Face Hub: checkpoint push/pull and ``hub:`` references (specs §5.8).

Authentication always comes from the stored ``huggingface_hub`` login (laptop) or
the copied login file (Colab VM) — never from arguments, configs or logs.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

HUB_PREFIX = "hub:"
LAST = "last-checkpoint"
# Files that only matter for resuming; kept out of the repo root (the "best" model).
RESUME_ONLY = ["optimizer.pt", "scheduler.pt", "rng.pt"]


def is_hub_ref(ref: str) -> bool:
    return ref.startswith(HUB_PREFIX)


def repo_id(ref: str) -> str:
    return ref.removeprefix(HUB_PREFIX).strip("/")


def resolve_checkpoint(ref: str | Path) -> Path:
    """Local folder for a checkpoint given as a path or ``hub:<user>/<repo>``."""
    ref = str(ref)
    if not is_hub_ref(ref):
        return Path(ref)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id(ref), ignore_patterns=[f"{LAST}/*"]))


class HubSync:
    """Push best / last checkpoints of one training run to one model repo."""

    def __init__(self, repo: str, *, private: bool = True) -> None:
        from huggingface_hub import HfApi

        self.repo = repo_id(repo)
        self.private = private
        self.api = HfApi()
        self._ready = False

    def _ensure_repo(self) -> None:
        if not self._ready:
            self.api.create_repo(self.repo, private=self.private, exist_ok=True)
            self._ready = True

    def _upload(self, folder: Path, message: str, path_in_repo: str | None = None) -> bool:
        try:
            self._ensure_repo()
            self.api.upload_folder(
                repo_id=self.repo,
                folder_path=str(folder),
                path_in_repo=path_in_repo,
                commit_message=message,
                ignore_patterns=None if path_in_repo else RESUME_ONLY,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — never kill a training run over an upload
            log.warning("hub upload to %s failed: %s", self.repo, exc)
            return False

    def push_best(self, folder: Path, message: str) -> bool:
        return self._upload(folder, message)

    def push_last(self, folder: Path, message: str) -> bool:
        return self._upload(folder, message, path_in_repo=LAST)

    def download_last(self, dest: Path) -> Path | None:
        """Fetch ``last-checkpoint/`` into ``dest``; None if the repo has none."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            snapshot_download(self.repo, allow_patterns=[f"{LAST}/*"], local_dir=str(dest))
        except RepositoryNotFoundError:
            return None
        folder = dest / LAST
        return folder if (folder / "state.json").is_file() else None
