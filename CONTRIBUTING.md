# Contributing to litmapr

Thanks for helping. A few things to know first.

## Setup

```
uv sync
uv run pytest
uv run ruff check .
```

Both must pass before a pull request is merged; CI checks them on every push.

## Where things live

| Folder / file | Job |
|---|---|
| `ingest/` | Readers: turn a file into `Page` objects (`CbzReader`, `PdfReader`) |
| `extract/` | Vision: turn a page image into text and a scene description |
| `llm/` | Text models and embeddings (`TextLLM`, `Embedder`) |
| `extraction.py` | Phase 3: pages → entities, mentions, relations |
| `resolve.py`, `callbacks.py` | Phase 4: merging duplicates, finding callbacks |
| `narrator.py` | Phase 6: spoiler-safe explanations |
| `model.py` | The data model (Pydantic) |
| `store.py` | SQLite storage |
| `web/` | FastAPI server and the 3D explorer page |
| `cli.py` | Commands. Kept thin: it calls the modules above |

## Adding a file format

1. Write `ingest/<format>_reader.py` with a class that inherits `Reader` and yields `Page` objects from `pages()`.
2. Add one line to `READERS` in `cli.py`.
3. Add a fixture in `tests/conftest.py` that builds a tiny sample file, and tests for the reader.

## Rules of thumb

- **Never commit comics or books**, not even for tests. Fixtures build tiny fake files in `tmp_path`.
- **Tests never call a real model.** Use `FakeExtractor`, `FakeLLM`, `ScriptedLLM` and `FakeEmbedder` from `conftest.py`.
- **Changing a prompt?** Bump its `PROMPT_VERSION` so old cached answers aren't reused.
- **Changing the database tables?** Bump `SCHEMA_VERSION` in `store.py`.
