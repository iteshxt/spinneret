---
license: cc-by-sa-4.0
language: en
library_name: transformers
base_model: google-t5/t5-small
datasets:
  - xlangai/spider
tags:
  - text-to-sql
  - spider
  - t5
  - sql
---

# Spinneret ablation E3 · T5-small without value matching

`google-t5/t5-small` fine-tuned on Spider to turn an English question and a database schema into SQLite SQL. It reads the whole schema.

Part of [Spinneret](https://github.com/iteshxt/spinneret): Baseline (T5-small), Lite (T5-small + ranker) and Spinneret (T5-base + ranker).

> An ablation checkpoint (E3): the Baseline trained and evaluated without database values in its input, to measure what value matching adds. Not used by the demo; for real use take [spinneret-t5-small](https://huggingface.co/iteshxt/spinneret-t5-small).

## Results on Spider dev

1,034 questions on 20 databases not seen in training; official test-suite evaluation; execution-guided beam search with 4 beams (the first candidate that runs is kept).

| | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Exact match % | 73.0 | 40.8 | 36.8 | 15.1 | 43.7 |
| Execution accuracy % | 72.2 | 44.8 | 42.0 | 18.7 | 46.7 |

Valid (executable) SQL: 66.4%.

| Question type | n | EX % |
|---|---|---|
| simple lookup | 252 | 65.5 |
| aggregation | 292 | 65.1 |
| two-table join | 288 | 25.0 |
| multi-join | 43 | 2.3 |
| nested / set-op | 159 | 34.6 |

## Input and output

Input: `question | database | table : column , column | …`. Names are lower-cased; no database values are shown (this checkpoint was trained without them).

```text
What is the average, minimum, and maximum age of all singers from France? | concert_singer | stadium : stadium_id , location , name , capacity , highest , lowest , average | singer : singer_id , name , country , song_name , song_release_year , age , is_male | concert : concert_id , concert_name , theme , stadium_id , year | singer_in_concert : concert_id , singer_id
```

Output:

```text
select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
```

## How to use

The easiest way is the Spinneret code, which builds the input, decodes and runs the SQL:

```bash
git clone https://github.com/iteshxt/spinneret && cd spinneret
uv sync --extra train --extra api
uv run spinneret download
(cd web && npm ci && npm run build)
uv run spinneret serve   # demo on :8000
```

With `transformers` alone, given an input string built as above:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("iteshxt/spinneret-ablation-t1-no-values")
model = AutoModelForSeq2SeqLM.from_pretrained("iteshxt/spinneret-ablation-t1-no-values")
ids = tok(source, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, num_beams=4, num_return_sequences=4, max_new_tokens=256)
candidates = tok.batch_decode(out, skip_special_tokens=True)
```

## Training

- Data: Spider `train_spider` (7,000 questions) including `train_others` (1,659), minus a validation set of 10% of the training databases (split by database).
- Optimiser: adafactor, learning rate 0.001, effective batch 32, early stopping on validation EX.
- Best epoch: 12 (validation EM 42.3%, EX 46.8%).
- Hardware: Kaggle T4 (scored on the laptop); the full config is `config.yaml` in this repo.

## Limitations

- English questions in the style of Spider; other phrasing and domains are less reliable.
- Errors on unseen schemas are mostly wrong joins and wrong columns; queries joining three or
  more tables, and nested or set queries, are much weaker than simple lookups (see the
  per-type table above).
- Values must match the database's spelling; the model copies values it is shown, and numbers
  written in words ("100 million") are often wrong.
- Execution-guided decoding needs the database: without it, take the top beam (lower EX).
- Intended for read-only querying and research; check generated SQL before relying on it.

## Licence and attribution

- Weights: fine-tuned from `google-t5/t5-small` (Apache-2.0).
- Training data: [Spider](https://yale-lily.github.io/spider) (Yu et al., EMNLP 2018),
  CC BY-SA 4.0 — this model is a derivative and is shared under the same licence.
- Code, training configs and full evaluation: [https://github.com/iteshxt/spinneret](https://github.com/iteshxt/spinneret).
