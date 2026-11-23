---
license: cc-by-sa-4.0
language: en
base_model: FacebookAI/roberta-base
datasets:
  - xlangai/spider
tags:
  - text-to-sql
  - spider
  - schema-linking
---

# Spinneret — schema ranker (RoBERTa-base)

A cross-encoder that reads a question together with a database schema and scores every table and column for whether the SQL answer needs it. Spinneret's Lite and Spinneret models only see the top 4 tables × 5 columns it keeps, which shortens their input and removes distracting schema items.

Part of [Spinneret](https://github.com/iteshxt/spinneret). Use it through the Spinneret code (`SchemaRanker` in `spinneret/ranker/infer.py`): the classification heads (`heads.pt`) are not a standard `transformers` head.

## Results

- Validation: table AUC 0.984, column AUC 0.978; coverage (every needed item kept) at top (4, 5): 98.1%.
- Spider dev coverage at (4, 5): 98.5%.

Dev coverage % by top-k (tables × columns):

| tables \ columns | 4 | 5 | 6 |
|---|---|---|---|
| 3 | 95.9 | 97.6 | 98.4 |
| 4 | 96.8 | 98.5 | 99.2 |
| 5 | 96.8 | 98.5 | 99.2 |

## Training

- Data: Spider train (incl. `train_others`), labels = the tables and columns used by the gold SQL; validation split by database.
- Best epoch: 2; selection by validation coverage.

## Licence and attribution

- Weights: fine-tuned from `FacebookAI/roberta-base` (MIT).
- Training data: [Spider](https://yale-lily.github.io/spider) (Yu et al., EMNLP 2018),
  CC BY-SA 4.0 — this model is a derivative and is shared under the same licence.
- Code, training configs and full evaluation: [https://github.com/iteshxt/spinneret](https://github.com/iteshxt/spinneret).
