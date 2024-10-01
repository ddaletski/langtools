import math
import re
import sys
from collections import defaultdict
from typing import List, TextIO

import click
import spacy
from rich.console import Console
from rich.progress import track

console = Console(file=sys.stderr)

def fail(message: str):
    raise Exception(message)

def load_freqs(input: TextIO, delimiter: str, description: str) -> defaultdict:
    freqs = defaultdict(int)

    for line in track(input.readlines(), description=description, console=console):
        fields = line.split(delimiter)
        stripped = line.strip("\n")

        try:
            word = fields[0]
        except:
            fail(f"failed to read a word from column #1 at '{stripped}'")
        try:
            freq = int(fields[1])
        except:
            fail(f"failed to read a frequency from column #2 at '{stripped}'")

        freqs[word] += freq

    return freqs

@click.group()
def cli():
    """
    Command line interface for working with frequencies of words in a corpus.
    """
    pass

@cli.command(help="build a frequency dictionary from text(s)")
@click.argument("inputs", type=click.File('r'), nargs=-1)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
def build(inputs: List[TextIO], output: TextIO | None):
    if len(inputs) == 0:
        inputs = [sys.stdin]

    freqs = defaultdict(int)
    for file_idx, ifile in enumerate(inputs):
        for line in track(ifile.readlines(), description=f"Processing file #{file_idx+1}", console=console):
            line = re.sub(r"\W", " ", line)
            for word in line.split(" "):
                if word == "":
                    continue
                freqs[word] += 1

    for word, freq in freqs.items():
        output.write(f"{word}\t{freq}\n")

@cli.command(help="merge multiple frequency dictionaries")
@click.argument("inputs", type=click.File('r'), nargs=-1)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
@click.option("--in-threshold", type=click.IntRange(min=0), default=0, help="word frequency threshold in an input file. Anything less frequent is skipped")
@click.option("--out-threshold", type=click.IntRange(min=0), default=0, help="word frequency threshold in the output file. Anything less frequent is trimmed from the result")
def merge(inputs: List[TextIO],
          output: TextIO | None,
          delimiter: str,
          in_threshold: int,
          out_threshold: int
    ):
    if len(inputs) == 0:
        inputs = [sys.stdin]

    merged = defaultdict(int)
    for file_idx, ifile in enumerate(inputs):
        file_freqs = load_freqs(ifile, delimiter, f"Loading file #{file_idx+1}")
        for word, freq in track(file_freqs.items(), description=f"Processing file #{file_idx+1}", console=console):
            if freq >= in_threshold:
                merged[word] += freq

    filtered = [(k, v) for (k, v) in merged.items() if v >= out_threshold]

    for word, freq in filtered:
        output.write(f"{word}{delimiter}{freq}\n")


@cli.command(help="clean words, removing numbers and punctuation")
@click.argument("input", type=click.File('r'), default=sys.stdin)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
def clean(input: TextIO, output: TextIO, delimiter: str):
    freqs = load_freqs(input, delimiter, "Loading...")

    for word, freq in track(freqs.items(), description="Cleaning...", console=console):
        if re.search(r"[0-9\W]", word):
            continue

        output.write(f"{word}{delimiter}{freq}\n")

@cli.command(help="normalize words (lemmatize)")
@click.argument("input", type=click.File('r'), default=sys.stdin)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
@click.option("-m", "--model", type=str, default="de_core_news_md", help="spacy model")
def normalize(input: TextIO, output: TextIO, delimiter: str, model: str):
    nlp = spacy.load(model)

    freqs = load_freqs(input, delimiter, "Loading...")

    normalized = defaultdict(int)

    for word, freq in track(freqs.items(), description="Normalizing...", console=console):
        lemma = nlp(word)[0].lemma_
        normalized[lemma] += freq

    for word, freq in normalized.items():
        output.write(f"{word}{delimiter}{freq}\n")


@cli.command(help="select the most common words")
@click.argument("input", type=click.File('r'), default=sys.stdin)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
@click.option("-m", "--metric", type=click.Choice(["words", "chars"]), default="words",
              help="how text volume is measured. 'words' targets understanding P%% of corpus words.\n'chars' targets understanding P%% of corpus text volume")
@click.option("-p", "--percentile", type=click.IntRange(min=1, max=100), default=90, help="how much of the original text data would be kept if only the top N words are kept")
def select_top(input: TextIO, output: TextIO, delimiter: str, metric: str, percentile: int):
    if metric == "words":
        impact_fn = lambda _, freq: freq
    elif metric == "chars":
        impact_fn = lambda word, freq: len(word) * freq
    else:
        fail(f"unknown metric '{metric}'")

    freqs = load_freqs(input, delimiter, "Loading...")
    total = 0
    for word, freq in freqs.items():
        impact = impact_fn(word, freq)
        total += impact
        freqs[word] = (impact, freq)

    target = int(math.ceil(total * percentile / 100.0))
    sorted_by_impact = sorted(freqs.items(), key=lambda x: x[1][0], reverse=True)

    for word, (impact, freq) in sorted_by_impact:
        output.write(f"{word}{delimiter}{freq}\n")
        target -= impact
        if target <= 0:
            break

@cli.command(help="select given words from a frequency table")
@click.option("--words", type=click.File('r'), required=True, help="list of words to select")
@click.option("--table", type=click.File('r'), required=True, help="frequency table to sample from")
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout, help="output frequency table")
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
@click.option("--missing", type=click.Choice(["skip", "zero"]), default="skip", help="how to handle missing words")
def select(words: TextIO, table: TextIO, output: TextIO, delimiter: str, missing: str):
    freqs = load_freqs(table, delimiter, "Loading...")

    for line in track(words.readlines(), description="Selecting...", console=console):
        word = line.strip("\n")

        if word in freqs:
            freq = freqs[word]
            output.write(f"{word}{delimiter}{freq}\n")
        elif missing == "zero":
            output.write(f"{word}{delimiter}0\n")


@cli.command(help="find the difference between two frequency tables")
@click.argument("before", type=click.File('r'), required=True)
@click.argument("after", type=click.File('r'), required=True)
@click.option("-o", "--output", type=click.File('w'), default=sys.stdout)
@click.option("-d", "--delimiter", type=str, default="\t", help="table delimiter")
def diff(before: TextIO, after: TextIO, output: TextIO, delimiter: str):
    freqs_before = load_freqs(before, delimiter, "Loading file #1...")
    freqs_after = load_freqs(after, delimiter, "Loading file #2...")

    for key in track(freqs_before.keys() | freqs_after.keys(), description="Comparing...", console=console):
        diff = freqs_after.get(key, 0) - freqs_before.get(key, 0)
        if diff == 0:
            continue

        output.write(f"{key}{delimiter}{diff}\n")


@cli.command
@click.option("--subset", type=click.File('r'), required=True)
@click.option("--all", type=click.File('r'), required=True)
def subset_stats(subset: TextIO, all: TextIO):
    subset_freqs = load_freqs(subset, "\t", "Loading...")
    all_freqs = load_freqs(all, "\t", "Loading...")

    all_occurences = sum(all_freqs.values())
    subset_occurrences = sum(subset_freqs.values())

    percentage = subset_occurrences / all_occurences * 100
    print(f"the subset spans {percentage:.2f}% of the corpus")


if __name__ == "__main__":
    cli()