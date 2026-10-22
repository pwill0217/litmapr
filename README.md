# litmapr

Turn books and comics into an interactive 3D map of plot points.

litmapr reads the files you own (PDF, CBZ, CBR, EPUB), pulls out characters, devices, concepts and relationships, and links them across a whole series. Import comics or standard books and see where every idea was planted and where it pays off. A narrator explains any plot point you click, without spoiling what you haven't read yet.

> **Status:** early development (Phase 0: project skeleton). Nothing is mapped yet.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/pwill0217/litmapr.git
cd litmapr
uv sync
```

## Usage

```
uv run litmapr --help
uv run litmapr ingest path/to/book.cbz
uv run litmapr serve
```

Commands are placeholders for now.

## Development

```
uv run ruff check .
uv run pytest
```

CI runs both on every push.

## Your files stay yours

litmapr works only with files you provide. It never downloads, stores or shares books or comics, and this repo contains no copyrighted material.

## License

MIT. See [LICENSE](LICENSE).
