import csv
import sys

from rich.progress import track
from rich.console import Console
import translators as ts
import click

console = Console(file=sys.stderr)

@click.command
@click.argument("input", type=click.File('r'), default=sys.stdin)
@click.argument("output", type=click.File('w'), default=sys.stdout)
@click.option("--src", default="de", help="source language")
@click.option("--dst", default="en", help="target language")
@click.option("--src-column", default=0, help="column containing source text")
@click.option("-d", "--delimiter", default="\t", help="table delimiter")
def translate(input, output, src, dst, src_column, delimiter):
    reader = csv.reader(input, delimiter=delimiter)

    for fields in track(reader, description="Translating...", console=console):
        word = fields[src_column]
        translation = ts.translate_text(word, from_language=src, to_language=dst, translator='google')

        fields.append(translation)
        output.write(delimiter.join(fields) + "\n")

if __name__ == "__main__":
    translate()