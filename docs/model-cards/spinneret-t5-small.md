---
license: cc-by-sa-4.0
language: en
library_name: transformers
pipeline_tag: text2text-generation
base_model: google-t5/t5-small
datasets:
  - xlangai/spider
tags:
  - text-to-sql
  - spider
  - t5
  - sql
---

# Spinneret Baseline · T5-small

`google-t5/t5-small` fine-tuned on Spider to turn an English question and a database schema into SQLite SQL. It reads the whole schema.

Part of [Spinneret](https://github.com/iteshxt/spinneret): Baseline (T5-small), Lite (T5-small + ranker) and Spinneret (T5-base + ranker).

## Results on Spider dev

1,034 questions on 20 databases not seen in training; official test-suite evaluation; execution-guided beam search with 4 beams (the first candidate that runs is kept).

| | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Exact match % | 79.4 | 43.7 | 41.4 | 22.3 | 48.5 |
| Execution accuracy % | 76.6 | 45.1 | 44.8 | 24.1 | 49.2 |

Valid (executable) SQL: 67.5%.

| Question type | n | EX % |
|---|---|---|
| simple lookup | 252 | 65.5 |
| aggregation | 292 | 70.9 |
| two-table join | 288 | 25.3 |
| multi-join | 43 | 14.0 |
| nested / set-op | 159 | 36.5 |

## Size and speed

- Parameters: 60.5M
- Latency per question, full pipeline at batch 1 on a laptop NVIDIA GeForce GTX 1650: median 0.37 s, p95 0.74 s
- Peak GPU memory: 0.41 GB

## Input and output

Input: `question | database | table : column , column ( matched value ) | …`. Names are lower-cased; database values that match words in the question are shown in brackets next to their column.

```text
What is the average, minimum, and maximum age of all singers from France? | concert_singer | stadium : stadium_id , location , name , capacity , highest , lowest , average | singer : singer_id , name , country ( France ) , song_name , song_release_year , age , is_male | concert : concert_id , concert_name , theme , stadium_id , year | singer_in_concert : concert_id , singer_id
```

Output:

```text
select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
```

## How to use

The easiest way is the Spinneret code, which builds the input (value matching), decodes and runs the SQL:

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
tok = AutoTokenizer.from_pretrained("iteshxt/spinneret-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("iteshxt/spinneret-t5-small")
ids = tok(source, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, num_beams=4, num_return_sequences=4, max_new_tokens=256)
candidates = tok.batch_decode(out, skip_special_tokens=True)
```

## Training

- Data: Spider `train_spider` (7,000 questions) including `train_others` (1,659), minus a validation set of 10% of the training databases (split by database).
- Optimiser: adafactor, learning rate 0.001, effective batch 32, early stopping on validation EX.
- Best epoch: 18 (validation EM 44.7%, EX 49.3%).
- Hardware: free Colab T4 GPUs and a laptop GTX 1650, resuming through Hub checkpoints; the full config is `config.yaml` in this repo.

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
