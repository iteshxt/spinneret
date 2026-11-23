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

# Spinneret Lite · T5-small + ranker + skeleton

`google-t5/t5-small` fine-tuned on Spider to turn an English question and a database schema into SQLite SQL. It reads only the tables and columns kept by the schema ranker ([iteshxt/spinneret-ranker-roberta-base](https://huggingface.co/iteshxt/spinneret-ranker-roberta-base)), top 4 tables × 5 columns. It writes the query's skeleton first, then the full query.

Part of [Spinneret](https://github.com/iteshxt/spinneret): Baseline (T5-small), Lite (T5-small + ranker) and Spinneret (T5-base + ranker).

> This is the Lite model as first planned, with skeleton-first decoding. Ablation E4 showed the skeleton does not help a model this small, so the demo serves [spinneret-ablation-t2-no-skeleton](https://huggingface.co/iteshxt/spinneret-ablation-t2-no-skeleton) instead; this checkpoint is kept for the evaluation.

## Results on Spider dev

1,034 questions on 20 databases not seen in training; official test-suite evaluation; execution-guided beam search with 4 beams (the first candidate that runs is kept).

| | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Exact match % | 85.1 | 51.3 | 37.9 | 21.7 | 52.4 |
| Execution accuracy % | 82.3 | 54.7 | 39.7 | 27.1 | 54.4 |

Valid (executable) SQL: 78.3%.

| Question type | n | EX % |
|---|---|---|
| simple lookup | 252 | 71.8 |
| aggregation | 292 | 75.7 |
| two-table join | 288 | 36.5 |
| multi-join | 43 | 0.0 |
| nested / set-op | 159 | 34.6 |

## Size and speed

- Parameters: 60.5M (+ 125.8M ranker)
- Latency per question, full pipeline at batch 1 on a laptop NVIDIA GeForce GTX 1650: median 0.62 s, p95 1.18 s
- Peak GPU memory: 0.65 GB

## Input and output

Input: `question | database | table : column , column ( matched value ) | … | fk : a.x = b.y , …`. Names are lower-cased; database values that match words in the question are shown in brackets next to their column.

```text
What is the average, minimum, and maximum age of all singers from France? | concert_singer | singer : age , country ( France ) , singer_id , name , is_male | singer_in_concert : singer_id , concert_id | stadium : average , stadium_id , highest , lowest , name | concert : year , stadium_id , concert_id , concert_name , theme | fk : concert.stadium_id = stadium.stadium_id , singer_in_concert.singer_id = singer.singer_id , singer_in_concert.concert_id = concert.concert_id
```

Output:

```text
select avg ( _ ) , min ( _ ) , max ( _ ) from _ where _ = _ | select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'
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
tok = AutoTokenizer.from_pretrained("iteshxt/spinneret-lite-t5-small-ranked")
model = AutoModelForSeq2SeqLM.from_pretrained("iteshxt/spinneret-lite-t5-small-ranked")
ids = tok(source, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, num_beams=4, num_return_sequences=4, max_new_tokens=256)
candidates = tok.batch_decode(out, skip_special_tokens=True)
sql = candidates[0].split(" | ", 1)[-1]  # drop the skeleton
```

## Training

- Data: Spider `train_spider` (7,000 questions) including `train_others` (1,659), minus a validation set of 10% of the training databases (split by database).
- Optimiser: adafactor, learning rate 0.001, effective batch 32, early stopping on validation EX.
- Best epoch: 12 (validation EM 50.8%, EX 52.6%).
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
