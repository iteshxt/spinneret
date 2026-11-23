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

# Spinneret Lite · T5-small + ranker

`google-t5/t5-small` fine-tuned on Spider to turn an English question and a database schema into SQLite SQL. It reads only the tables and columns kept by the schema ranker ([iteshxt/spinneret-ranker-roberta-base](https://huggingface.co/iteshxt/spinneret-ranker-roberta-base)), top 4 tables × 5 columns.

Part of [Spinneret](https://github.com/iteshxt/spinneret): Baseline (T5-small), Lite (T5-small + ranker) and Spinneret (T5-base + ranker).

## Results on Spider dev

1,034 questions on 20 databases not seen in training; official test-suite evaluation; execution-guided beam search with 4 beams (the first candidate that runs is kept).

| | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Exact match % | 85.9 | 51.6 | 39.1 | 19.3 | 52.5 |
| Execution accuracy % | 82.7 | 56.3 | 45.4 | 25.3 | 55.8 |

Valid (executable) SQL: 82.2%.

| Question type | n | EX % |
|---|---|---|
| simple lookup | 252 | 73.8 |
| aggregation | 292 | 76.4 |
| two-table join | 288 | 37.5 |
| multi-join | 43 | 0.0 |
| nested / set-op | 159 | 37.7 |

## Size and speed

- Parameters: 60.5M (+ 125.8M ranker)
- Latency per question, full pipeline at batch 1 on a laptop NVIDIA GeForce GTX 1650: median 0.39 s, p95 0.91 s
- Peak GPU memory: 0.62 GB

## Input and output

Input: `question | database | table : column , column ( matched value ) | … | fk : a.x = b.y , …`. Names are lower-cased; database values that match words in the question are shown in brackets next to their column.

```text
What is the average, minimum, and maximum age of all singers from France? | concert_singer | singer : age , country ( France ) , singer_id , name , is_male | singer_in_concert : singer_id , concert_id | stadium : average , stadium_id , highest , lowest , name | concert : year , stadium_id , concert_id , concert_name , theme | fk : concert.stadium_id = stadium.stadium_id , singer_in_concert.singer_id = singer.singer_id , singer_in_concert.concert_id = concert.concert_id
```

Output:

```text
select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
```

## How to use

The easiest way is the Spinneret code, which builds the input (value matching, schema ranking), decodes and runs the SQL:

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
tok = AutoTokenizer.from_pretrained("iteshxt/spinneret-ablation-t2-no-skeleton")
model = AutoModelForSeq2SeqLM.from_pretrained("iteshxt/spinneret-ablation-t2-no-skeleton")
ids = tok(source, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, num_beams=4, num_return_sequences=4, max_new_tokens=256)
candidates = tok.batch_decode(out, skip_special_tokens=True)
```

## Training

- Data: Spider `train_spider` (7,000 questions) including `train_others` (1,659), minus a validation set of 10% of the training databases (split by database).
- Optimiser: adafactor, learning rate 0.001, effective batch 32, early stopping on validation EX.
- Best epoch: 17 (validation EM 50.1%, EX 54.5%).
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
