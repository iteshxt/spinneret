from __future__ import annotations

import zipfile

import pytest

from spinneret.data.download import extract


def _zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_extract_skips_junk(tmp_path):
    archive = tmp_path / "spider_data.zip"
    _zip(
        archive,
        {
            "spider_data/tables.json": "[]",
            "spider_data/database/a/a.sqlite": "x",
            "spider_data/.DS_Store": "junk",
            "__MACOSX/spider_data/._tables.json": "junk",
        },
    )
    out = tmp_path / "spider"
    assert extract(archive, out) == 2
    assert (out / "tables.json").is_file()
    assert (out / "database" / "a" / "a.sqlite").is_file()
    assert not (out / ".DS_Store").exists()


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    _zip(archive, {"spider_data/../../escape.txt": "x"})
    with pytest.raises(ValueError, match="unsafe path"):
        extract(archive, tmp_path / "spider")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.network
def test_t5_tokenizer_needs_less_than_tokens():
    from spinneret.model.tokenizer import load_tokenizer, missing_characters, roundtrip, tokens_for

    base = load_tokenizer("google-t5/t5-small")
    sqls = [
        "select name from singer where age < 30 and year <= 2000",
        "select count ( * ) from t where x <> 2",
    ]
    assert "<" in missing_characters(base, sqls)
    assert roundtrip(base, sqls[0]) != sqls[0]

    fixed = load_tokenizer("google-t5/t5-small", tokens_for("<"))
    for sql in sqls:
        assert " ".join(roundtrip(fixed, sql).split()) == sql
