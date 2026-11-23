"""Model registry for the API (design §13): load once, in the background, behind one GPU lock.

The server answers schema and metrics requests straight away; ``/api/query`` returns 503
``model_loading`` until every configured model is ready. Generation runs one request at a
time (``gpu_lock``), so concurrent requests queue instead of competing for GPU memory; at
most ``queue_limit`` may wait, and the rest get 503 ``busy`` straight away.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from spinneret.config import ServeConfig, ServeModel
from spinneret.data.schema import Schema
from spinneret.model.translator import Translation

log = logging.getLogger("spinneret.api")


class TranslatorLike(Protocol):
    params_m: float
    uses_skeleton: bool

    def translate(self, question: str, db_id: str, k: int | None = None) -> Translation: ...


Loader = Callable[[ServeConfig, dict[str, Schema]], dict[str, TranslatorLike]]


def load_translators(cfg: ServeConfig, schemas: dict[str, Schema]) -> dict[str, TranslatorLike]:
    """Build every configured model; rankers are loaded once and shared."""
    import torch

    from spinneret.model.translator import Translator
    from spinneret.ranker.infer import SchemaRanker

    auto = "cuda" if torch.cuda.is_available() else "cpu"
    device = auto if cfg.device == "auto" else cfg.device
    rankers: dict[str, Any] = {}
    out: dict[str, TranslatorLike] = {}
    for m in cfg.models:
        ranker = None
        if m.ranker:
            if m.ranker not in rankers:
                log.info("loading ranker %s", cfg.rankers[m.ranker])
                rankers[m.ranker] = SchemaRanker(cfg.rankers[m.ranker], device=device)
            ranker = rankers[m.ranker]
        log.info("loading %s (%s)", m.id, m.generator)
        out[m.id] = Translator(m.generator, schemas, ranker=ranker, device=device,
                               beam_size=m.beam_size)  # fmt: skip
    if schemas:  # the first generate on a GPU is several times slower; pay it at startup
        db_id = "concert_singer" if "concert_singer" in schemas else next(iter(schemas))
        for t in out.values():
            t.translate("How many singers do we have?", db_id)
    return out


class Busy(Exception):
    """Too many questions are already waiting for the GPU."""


class Registry:
    def __init__(self, cfg: ServeConfig, schemas: dict[str, Schema], loader: Loader) -> None:
        self.cfg = cfg
        self.models: dict[str, ServeModel] = {m.id: m for m in cfg.models}
        self.translators: dict[str, TranslatorLike] = {}
        self.error: str | None = None
        self.gpu_lock = threading.Lock()
        self._waiting = 0
        self._waiting_lock = threading.Lock()
        self._schemas = schemas
        self._loader = loader
        self._ready = threading.Event()
        if cfg.default_model and cfg.default_model not in self.models:
            raise ValueError(f"default_model {cfg.default_model!r} is not in models")
        for m in cfg.models:
            if m.ranker and m.ranker not in cfg.rankers:
                raise ValueError(f"model {m.id!r} uses unknown ranker {m.ranker!r}")

    @contextmanager
    def gpu(self) -> Iterator[None]:
        """Hold the GPU lock, or raise ``Busy`` if ``queue_limit`` requests already hold or
        wait for it."""
        with self._waiting_lock:
            if self._waiting >= self.cfg.queue_limit:
                raise Busy
            self._waiting += 1
        try:
            with self.gpu_lock:
                yield
        finally:
            with self._waiting_lock:
                self._waiting -= 1

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def device(self) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if self.cfg.device == "cpu" or not torch.cuda.is_available():
            return "cpu"
        return torch.cuda.get_device_name(0)

    def start(self, background: bool = True) -> None:
        if not self.cfg.models:
            self._ready.set()
            return
        if background:
            threading.Thread(target=self._load, name="model-loader", daemon=True).start()
        else:
            self._load()

    def _load(self) -> None:
        try:
            self.translators = self._loader(self.cfg, self._schemas)
            log.info("models ready: %s", ", ".join(self.translators))
        except Exception as exc:  # noqa: BLE001 — reported by /api/health, server stays up
            log.exception("model loading failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()
