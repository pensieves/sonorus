import sys
from pathlib import Path

path = str(Path(__file__).parents[1].resolve())
sys.path.append(path)

import argparse
from pathlib import Path
import numpy as np
import pyaudio
import soundfile as sf
from sonorus.audio import VADAudioInputStreamer

TYPE2FMT = {
    "PCM_U8": pyaudio.paUInt8,
    "PCM_S8": pyaudio.paInt8,
    "PCM_16": pyaudio.paInt16,
    "PCM_32": pyaudio.paInt32,
    "FLOAT": pyaudio.paFloat32,
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "-r",
    "--sample_rate",
    type=int,
    help="Sample rate of the audio being recorded",
    default=16000,
)

parser.add_argument(
    "-f",
    "--format",
    choices=list(TYPE2FMT.keys()),
    help="Encoding format for recording",
    default="PCM_16",
)

parser.add_argument(
    "-o",
    "--out_dir",
    help="Output directory where recorded files should be saved",
    default=".",
)

parser.add_argument(
    "-p",
    "--prompt",
    action="store_true",
    help="Specify to provide prompt for continue recording",
)

args = parser.parse_args()

audio_streamer = VADAudioInputStreamer(
    sample_rate=args.sample_rate,
    pa_format=TYPE2FMT[args.format],
)

out_dir = Path(args.out_dir)
file_count = 1

with audio_streamer as streamer:

    if args.prompt:
        input("Press any key to continue recording:")

    for i, stream in enumerate(streamer.stream()):

        if stream is not None:

            out_file = f"{out_dir}/{file_count}.wav"
            sf.write(
                out_file, 
                np.frombuffer(
                    stream, 
                    streamer.FMT2TYPE[streamer.pa_format],
                ), 
                samplerate=args.sample_rate,
                subtype=args.format
            )

            print(f"Written to file {out_file}")
            file_count += 1

            if args.prompt:
                input("Press any key to continue recording:")