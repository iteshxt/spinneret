"""Canonical surface form for SQL targets.

Rules (specs §4.6):

1. keywords, functions, table names, aliases and column names → lower case;
2. string literals keep their case; double-quoted literals become single-quoted;
3. one space between tokens, none around ``.`` in qualified names
   (``select count ( * ) from singer as t1 where t1.name = 'Joe'``);
4. no trailing semicolon.

The result must execute exactly like the original — a slow test checks this on
every Spider gold query.
"""

from __future__ import annotations

import sqlparse
from sqlparse import tokens as T


def _single_quoted(literal: str) -> str:
    """Re-quote a SQL string literal with single quotes."""
    quote, inner = literal[0], literal[1:-1]
    if quote == '"':
        inner = inner.replace('""', '"').replace('\\"', '"')
    else:
        inner = inner.replace("''", "'")
    return "'" + inner.replace("'", "''") + "'"


def sql_tokens(sql: str) -> list[str]:
    """Lexical tokens of ``sql`` in canonical form (no whitespace or comments)."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        return []
    out: list[str] = []
    for statement in sqlparse.parse(text):
        for tok in statement.flatten():
            if tok.is_whitespace or tok.ttype in T.Comment:
                continue
            if tok.ttype in T.Literal.String.Single or tok.ttype in T.Literal.String.Symbol:
                out.append(_single_quoted(tok.value))
            elif tok.ttype is T.Punctuation and tok.value == ";":
                continue
            else:
                out.append(tok.value.lower())
    return out


def join_tokens(tokens: list[str]) -> str:
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i and tok != "." and tokens[i - 1] != ".":
            parts.append(" ")
        parts.append(tok)
    return "".join(parts)


def normalize_sql(sql: str) -> str:
    return join_tokens(sql_tokens(sql))
