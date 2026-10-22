import typer

app = typer.Typer()


@app.command()
def ingest(path: str):
    """Import a book or comic file."""
    print(f"Not implemented yet: {path}")


@app.command()
def serve():
    """Start the 3D explorer."""
    print("Not implemented yet")
