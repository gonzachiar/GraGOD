import typer

app = typer.Typer()


@app.command()
def test_cli():
    typer.secho("GraGOD CLI is working!", fg=typer.colors.GREEN, bold=True)


@app.command()
def hello(name: str):
    typer.echo(f"Hello {name}")


if __name__ == "__main__":
    app()
