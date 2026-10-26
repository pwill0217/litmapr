# litmapr

Turn books and comics into an interactive 3D map of plot points.

litmapr reads the files you own (CBZ comics and PDF books or comics), pulls out characters, devices, concepts, places and relationships, and links them across a whole series. Import comics or standard books and see where every idea was planted and where it pays off. A narrator explains any plot point you click, without spoiling what you haven't read yet.

Everything runs on your own computer with local AI models through [Ollama](https://ollama.com). Nothing is uploaded anywhere.

> **Status:** works end to end and is being tested on Jonathan Hickman's *Fantastic Four*. Expect rough edges; see [Known issues](#known-issues).

## Install

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```
git clone https://github.com/pwill0217/litmapr.git
cd litmapr
uv sync
```

Pull the local models litmapr uses:

```
ollama pull qwen2.5vl:3b            # reads comic pages and finds pictures (vision)
ollama pull qwen3.6:35b-a3b-coding  # finds entities, relations, callbacks, duplicates
ollama pull nomic-embed-text        # spots likely duplicates and callbacks
ollama pull qwen3:4b-instruct       # the narrator: small and fast
```

Any Ollama model can be swapped in with `--model`, `--vision-model` or `--embed-model`. On a smaller computer, a mid-size model such as `qwen3:8b` works as the text model. Use an "instruct" model for the narrator: a "thinking" model writes out its reasoning first, which is slow.

It is developed on a GTX 1060 (6 GB) with 64 GB of RAM. The big text model mostly runs on the CPU there, so extraction is slow but works; the narrator's small model fits on the GPU.

## Usage

```
uv run litmapr add "D:\comics\FF_570.cbz" --year 2009 --month 10
uv run litmapr add "D:\comics\Hickman FF"        # a whole folder, one issue at a time
uv run litmapr merge                             # fold duplicates like "Invisible Woman" into "Susan Storm"
uv run litmapr link                              # find callbacks between issues
uv run litmapr portraits                         # optional: pictures for the dossiers
uv run litmapr serve                             # open the 3D explorer in your browser
```

The year and month put each issue in reading order across series.

### Pictures

Dossiers and map markers can show a picture of each character, device, place, faction and event, cut out of your own comics:

```
uv sync --extra portraits          # optional: background removal (large download, about 1 GB)
uv run litmapr portraits           # the 30 most-seen of each kind
uv run litmapr portraits --kind character --kind device
uv run litmapr portraits --name "Spider-Man" --issue 574 --page 8   # look on a page you pick
uv run litmapr portraits --name "Galactus" --image galactus.png     # use your own picture
uv run litmapr portraits --name "Thing" --remove                    # delete a wrong one
```

The vision model finds the entry on its busiest pages and litmapr crops it. For characters and devices it then checks the crop really shows them (not someone standing next to them) and, with the extra installed, removes the background. Places, factions and events get the whole panel. Concepts keep their icon.

Small vision models still pick the wrong person sometimes. Every dossier has **Replace picture** and **Remove picture** to fix that in the browser. Pictures stay in `data/portraits` on your computer; `export` never includes them, and `serve --no-portraits` hides them.

### Omnibus editions

An omnibus holds many issues in one file. Tell litmapr where each issue starts, so every issue gets its own place in the reading order and its own page numbers:

```
uv run litmapr contents "D:\comics\Hickman FF Omnibus.cbz"   # drafts "Hickman FF Omnibus.contents.txt" from bookmarks
# check the draft against the book, add years and months, then:
uv run litmapr add "D:\comics\Hickman FF Omnibus.cbz"        # picks up the contents file automatically
```

A contents file is one issue per line (`#` lines are ignored):

```
# first page, series, issue, year, month
5,   Fantastic Four, 570, 2009, 10
29,  Fantastic Four, 571, 2009, 11
```

Bookmarks come from a PDF's table of contents or a CBZ's ComicInfo page bookmarks. With no bookmarks, write the file from the book's table of contents.

### All commands

| Command | What it does |
|---|---|
| `add` | Read a file or folder and add it to the map (`ingest` + `extract` in one step) |
| `ingest` | Read every page and save its text to `data/maps/<name>.json` (`--ocr` reads images) |
| `inspect` | Look at one page: its text, a scene description (`--ocr`), or save its image |
| `extract` | Find entities, appearances and relationships in an ingested file |
| `merge` | Find and merge entities that are really the same one, including codenames and real names |
| `merge-entities` | Merge two entities by hand |
| `link` | Find later moments that call back to earlier ones |
| `entities` | List everything in the map, most-mentioned first |
| `show` | Everything about one entity: first appearance, appearances, quotes, relations, callbacks |
| `works` | Every issue in reading order |
| `contents` | Draft an omnibus's contents file from its bookmarks |
| `portraits` | Cut pictures of characters, devices, places, factions and events out of the comics |
| `narrate` | Have the narrator explain an entity, up to a reading point (`--style briefing`) |
| `serve` | Start the 3D explorer at http://127.0.0.1:8000 |
| `export` / `import` | Share a map as JSON (no comic pages included) |

Run any command with `--help` for its options.

### The explorer

- **I've read up to** hides everything after that issue: the map, the side panel and the narrator never show spoilers.
- **Timeline depth** places each node by where it first appears in reading order, so you can fly along the story.
- Click a marker to open its dossier: picture, stats, a presence strip across the issues, plot points with quotes, connections with how warm or hostile they are, callbacks, and the narrator.
- **The narrator** speaks out loud with your browser's built-in voices (in Edge, pick one with "Natural" in its name). Choose **Storyteller** or **Field briefing**. The words stream in as the model writes them, and finished narrations are saved, so opening one again is instant.
- Selecting an entry fades everything it isn't connected to.
- The legend on the left hides or shows each kind of entity, with a live count.

## How it works

```
comic / book → pages (ingest) → text + scene per page (vision model)
             → entities, appearances, relations (text model, checked by Pydantic)
             → duplicates merged (embeddings + a read of the whole cast list + text model)
             → callbacks found (embeddings + text model)
             → SQLite → 3D explorer + narrator (FastAPI + 3d-force-graph)
```

Every model answer is cached, so re-running a step never repeats work it already did.

## Known issues

- Extraction sometimes files an entry under the wrong kind (a team or a universe listed as a character).
- Duplicates the text model doesn't recognise stay separate. Fold them with `merge-entities "Keep" "Drop"`.
- Wrong pictures still happen with the small vision model. Replace or remove them from the dossier.
- On a full map the centre gets crowded and labels overlap. Hide some kinds in the legend, or zoom in.

Bug reports are welcome: open an issue with the command you ran and what you saw.

## Development

```
uv run ruff check .
uv run pytest
```

CI runs both on every push. Tests never call a real model: they use fakes, so they run anywhere in a few seconds. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Your files stay yours

litmapr works only with files you provide. It never downloads, stores or shares books or comics, and this repo contains no copyrighted material. Exported maps contain names, short summaries and brief quotes, not pages.

## License

MIT. See [LICENSE](LICENSE). The bundled [3d-force-graph](https://github.com/vasturiano/3d-force-graph) library is MIT-licensed by Vasco Asturiano, and the bundled Chakra Petch and IBM Plex Mono fonts are under the SIL Open Font License (`src/litmapr/web/static/vendor/`).
