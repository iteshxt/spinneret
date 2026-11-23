---
license: cc-by-sa-4.0
language: en
library_name: transformers
pipeline_tag: text2text-generation
base_model: google-t5/t5-base
datasets:
  - xlangai/spider
tags:
  - text-to-sql
  - spider
  - t5
  - sql
---

# Spinneret · T5-base + ranker

`google-t5/t5-base` fine-tuned on Spider to turn an English question and a database schema into SQLite SQL. It reads only the tables and columns kept by the schema ranker ([iteshxt/spinneret-ranker-roberta-base](https://huggingface.co/iteshxt/spinneret-ranker-roberta-base)), top 4 tables × 5 columns. It writes the query's skeleton first, then the full query.

Part of [Spinneret](https://github.com/iteshxt/spinneret): Baseline (T5-small), Lite (T5-small + ranker) and Spinneret (T5-base + ranker).

## Results on Spider dev

1,034 questions on 20 databases not seen in training; official test-suite evaluation; execution-guided beam search with 8 beams (the first candidate that runs is kept).

| | easy | medium | hard | extra | all |
|---|---|---|---|---|---|
| Exact match % | 86.3 | 67.7 | 46.0 | 30.7 | 62.6 |
| Execution accuracy % | 89.1 | 73.3 | 58.0 | 37.3 | 68.8 |

Valid (executable) SQL: 91.7%.

| Question type | n | EX % |
|---|---|---|
| simple lookup | 252 | 83.7 |
| aggregation | 292 | 85.3 |
| two-table join | 288 | 55.2 |
| multi-join | 43 | 27.9 |
| nested / set-op | 159 | 50.3 |

## Size and speed

- Parameters: 222.9M (+ 125.8M ranker)
- Latency per question, full pipeline at batch 1 on a laptop NVIDIA GeForce GTX 1650: median 1.29 s, p95 2.50 s
- Peak GPU memory: 1.66 GB

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
tok = AutoTokenizer.from_pretrained("iteshxt/spinneret-t5-base-ranked")
model = AutoModelForSeq2SeqLM.from_pretrained("iteshxt/spinneret-t5-base-ranked")
ids = tok(source, return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, num_beams=8, num_return_sequences=8, max_new_tokens=256)
candidates = tok.batch_decode(out, skip_special_tokens=True)
sql = candidates[0].split(" | ", 1)[-1]  # drop the skeleton
```

## Training

- Data: Spider `train_spider` (7,000 questions) including `train_others` (1,659), minus a validation set of 10% of the training databases (split by database).
- Optimiser: adafactor, learning rate 0.0001, effective batch 32, early stopping on validation EX.
- Best epoch: 18 (validation EM 59.7%, EX 67.0%).
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

- Weights: fine-tuned from `google-t5/t5-base` (Apache-2.0).
- Training data: [Spider](https://yale-lily.github.io/spider) (Yu et al., EMNLP 2018),
  CC BY-SA 4.0 — this model is a derivative and is shared under the same licence.
- Code, training configs and full evaluation: [https://github.com/iteshxt/spinneret](https://github.com/iteshxt/spinneret).
