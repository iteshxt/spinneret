from __future__ import annotations

import sqlite3

import pytest

from spinneret.data.schema import Schema
from spinneret.data.serialize import RankedSchema, ranked_view, render, serialize_full
from spinneret.data.values import ValueMatcher, build_value_index, question_ngrams

TINY = {
    "db_id": "tiny",
    "table_names_original": ["singer"],
    "table_names": ["singer"],
    "column_names_original": [[-1, "*"], [0, "Name"], [0, "Country"], [0, "Debut"], [0, "Age"]],
    "column_names": [],
    "column_types": ["text", "text", "text", "text", "number"],
    "primary_keys": [],
    "foreign_keys": [],
}


@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (Name text, Country text, Debut text, Age int)")
    conn.executemany(
        "INSERT INTO singer VALUES (?, ?, ?, ?)",
        [
            ("Joe Sharp", "France", "1998", 52),
            ("Timbaland", "United States", "2001", 32),
            ("Rose White", "The Netherlands", "1999", 41),
        ],
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def matcher(tiny_db, tmp_path):
    schema = Schema.from_spider(TINY)
    return ValueMatcher(lambda _db: tiny_db, {"tiny": schema}, tmp_path / "cache")


def test_value_index_skips_numbers(tiny_db):
    index = build_value_index(tiny_db, Schema.from_spider(TINY))
    assert index["singer.country"] == ["France", "United States", "The Netherlands"]
    assert "singer.debut" not in index  # all values numeric
    assert "singer.age" not in index  # not a text column


def test_exact_and_fuzzy_matches(matcher):
    assert matcher.match("Which singers are from france?", "tiny") == {"singer.country": ["France"]}
    assert matcher.match("Show singers from the united states", "tiny") == {
        "singer.country": ["United States"]
    }
    # fuzzy: "frances" ≈ "france"
    assert matcher.match("singers from Frances", "tiny") == {"singer.country": ["France"]}


def test_numbers_and_stopwords_never_match(matcher):
    assert matcher.match("Who debuted in 1998?", "tiny") == {}
    assert question_ngrams("what is the") == []


def test_value_cache_written(matcher, tmp_path):
    matcher.match("france", "tiny")
    assert (tmp_path / "cache" / "tiny.json").is_file()


def test_full_serialization(concert_schema):
    text, truncated = serialize_full(
        "Which singers are from France?", concert_schema, {"singer.country": ["France"]}
    )
    assert not truncated
    assert text == (
        "Which singers are from France? | concert_singer | "
        "stadium : stadium_id , location , name , capacity , highest , lowest , average | "
        "singer : singer_id , name , country ( France ) , song_name , song_release_year , age , "
        "is_male | concert : concert_id , concert_name , theme , stadium_id , year | "
        "singer_in_concert : concert_id , singer_id"
    )


def test_full_serialization_respects_budget(concert_schema):
    count = lambda s: len(s.split())  # noqa: E731
    text, truncated = serialize_full(
        "How many singers are there?", concert_schema, {}, count_tokens=count, max_tokens=30
    )
    assert truncated and count(text) <= 30
    assert "singer :" in text  # the relevant table survives truncation


def test_ranked_view_top_k_with_foreign_keys(concert_schema):
    ranked = RankedSchema(
        tables={"singer": 0.99, "singer_in_concert": 0.8, "concert": 0.3, "stadium": 0.01},
        columns={"singer.name": 0.95, "singer.age": 0.9, "singer.country": 0.2},
    )
    view = ranked_view(concert_schema, ranked, top_k_tables=2, top_k_columns=2)
    assert [t for t, _ in view.tables] == ["singer", "singer_in_concert"]
    # top-2 columns plus the key linking the two kept tables, in score order
    assert view.tables[0] == ("singer", ["Name", "Age", "Singer_ID"])
    assert "Singer_ID" in view.tables[1][1]
    assert view.foreign_keys == ["singer_in_concert.singer_id = singer.singer_id"]
    text = render("q?", "concert_singer", view)
    assert text.endswith("| fk : singer_in_concert.singer_id = singer.singer_id")


def test_ranked_view_force_includes_gold(concert_schema):
    ranked = RankedSchema(tables={"singer": 0.9}, columns={"singer.name": 0.9})
    view = ranked_view(
        concert_schema,
        ranked,
        top_k_tables=1,
        top_k_columns=1,
        force_tables=["stadium"],
        force_columns={"singer": ["Is_male"]},
    )
    names = dict(view.tables)
    assert "stadium" in names
    assert "Is_male" in names["singer"]
