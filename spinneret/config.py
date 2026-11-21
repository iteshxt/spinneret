"""Experiment configuration (one YAML file per experiment, specs §3)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL = "google-t5/t5-small"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    name: str = DEFAULT_MODEL
    added_tokens: Literal["auto", "none"] | list[str] = "auto"


class DataConfig(_Strict):
    spider_dir: Path = Path("data/raw/spider")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    include_train_others: bool = True
    include_test: bool = False
    validation_fraction: float = Field(0.10, gt=0, lt=1)
    db_content: bool = True
    max_values_per_column: int = 2
    schema_input: Literal["full", "ranked"] = Field("full", alias="schema")
    ranker_checkpoint: str | None = None
    top_k_tables: int = 4
    top_k_columns: int = 5
    include_foreign_keys: bool = True
    target_format: Literal["sql", "skeleton_sql"] = "sql"
    max_source_len: int = 512
    max_target_len: int = 256

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def inputs_dir(self) -> Path:
        """Folder with {train,valid,dev}.jsonl for this config's schema input."""
        if self.schema_input == "full":
            return self.processed_dir
        if not self.ranker_checkpoint:
            raise ValueError("schema: ranked needs data.ranker_checkpoint")
        name = self.ranker_checkpoint.removeprefix("hub:").rstrip("/").split("/")[-1]
        return self.processed_dir / "ranked" / name


class TrainConfig(_Strict):
    optimizer: Literal["adafactor", "adamw"] = "adafactor"
    learning_rate: float = 1e-3
    lr_schedule: Literal["constant", "linear"] = "constant"
    warmup_ratio: float = 0.0
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    max_epochs: int = 50
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_checkpointing: bool = False
    eval_every_epochs: int = 1
    eval_decoding: Literal["greedy", "beam"] = "greedy"
    early_stopping_patience: int = 5
    select_best_by: Literal["valid_ex", "valid_em"] = "valid_ex"
    eval_batch_size: int = 64
    log_every_steps: int = 50
    max_valid_examples: int | None = None  # validate on a subset to save time


class HubConfig(_Strict):
    repo: str | None = None
    push_during_training: bool = True
    private_during_training: bool = True
    push_last_every_epochs: int = 1


class DecodingConfig(_Strict):
    beam_size: int = 4
    mode: Literal["greedy", "beam", "eg"] = "eg"


class ExperimentConfig(_Strict):
    run_name: str = "default"
    seed: int = 42
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    hub: HubConfig = HubConfig()
    decoding: DecodingConfig = DecodingConfig()


class ServeConfig(_Strict):
    """``configs/serve.yaml`` (specs §9). Models are wired in from week 3."""

    spider_dir: Path = Path("data/raw/spider")
    databases: Literal["dev", "all"] = "dev"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    default_model: str | None = None


def load_serve_config(path: Path | None) -> ServeConfig:
    if path is None:
        return ServeConfig()
    return ServeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_config(path: Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    data = cfg.model_dump(mode="json", by_alias=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
