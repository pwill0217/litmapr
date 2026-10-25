from pathlib import Path
from typing import Annotated

import typer

from litmapr.ingest.cbz_reader import CbzReader
from litmapr.ingest.pdf_reader import PdfReader

app = typer.Typer()

# File extension -> the reader class that handles it.
# Supporting a new format = write a reader + add one line here.
READERS = {
    ".cbz": CbzReader,
    ".pdf": PdfReader,
}


@app.command()
def ingest(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]):
    """Import a book or comic file."""
    reader_class = READERS.get(path.suffix.lower())
    if reader_class is None:
        supported = ", ".join(READERS)
        print(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
        raise typer.Exit(code=1)

    reader = reader_class(path)
    pages = list(reader.pages())  # the reader does all the real work here
    with_text = sum(1 for page in pages if page.text)
    print(f"found {len(pages)} pages in {path} ({with_text} with text)")


@app.command()
def serve():
    """Start the 3D explorer."""
    print("Not implemented yet")
