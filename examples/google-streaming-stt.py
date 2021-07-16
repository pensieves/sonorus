import sys
from pathlib import Path

path = str(Path(__file__).parents[1].resolve())
sys.path.append(path)

import argparse
from sonorus.speech import GoogleSTT


parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--lang",
    type=str,
    help="Specify the language as a BCP-47 language tag",
    default="en-IN",
)


args = parser.parse_args()


while True:
    try:
        GoogleSTT(lang=args.lang).streaming_transcribe()

    except Exception as exception:
        print(exception)
