from typer.testing import CliRunner
from litmapr import app

runner = CliRunner()

def test_app():
    result = runner.invoke(app, ["--help"])