import argparse

import sys
from pathlib import Path

path = str(Path(__file__).parents[1].resolve())
sys.path.append(path)

from sonorus.speech import Wav2Vec2STT
from sonorus.speech.lm import (
    FairseqTokenDictionary,
    W2lKenLMDecoder,
    W2lViterbiDecoder,
    W2lFairseqLMDecoder,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--lang",
    type=str,
    help="Specify the language as a BCP-47 language tag",
    default="en-US",
)

parser.add_argument(
    "-g",
    "--gpu_idx",
    type=int,
    help="index of GPU to be used for accelerated computation",
)

parser.add_argument(
    "--decoder",
    choices=["kenlm", "fairseqlm", "viterbi"],
    help="type of W2L decoder to be used",
)

parser.add_argument(
    "--lm_path", help="path to the .arpa or .bin kenlm model",
)

parser.add_argument(
    "--lexicon_path", help="path to the lexicon file",
)

parser.add_argument(
    "--lm_weight",
    type=float,
    default=1,
    help="weight to be used with language score in W2L decoder",
)

parser.add_argument(
    "--word_weight",
    type=float,
    default=2,
    help="weight to be used for word insertion in W2L decoder",
)

parser.add_argument(
    "--unk_weight",
    type=float,
    default=float("-inf"),
    help="weight to be used for unkown token insertion in W2L decoder",
)

parser.add_argument(
    "--sil_weight",
    type=float,
    default=4,
    help="weight to be used for silence/word break insertion in W2L decoder",
)

parser.add_argument(
    "--beam", type=int, default=5, help="No. of beams to be used for W2L decoding",
)

parser.add_argument(
    "--beam_size_token", type=int, default=100, help="beam size token for W2L decoding",
)

parser.add_argument(
    "--beam_threshold", type=float, default=25, help="beam threshold for W2L decoding",
)

args = parser.parse_args()

stt = Wav2Vec2STT(gpu_idx=args.gpu_idx)

decoder = None
if args.decoder:

    token_dict = FairseqTokenDictionary(
        symbols_int_map=stt.model_processor.tokenizer.get_vocab()
    )

    if args.decoder == "viterbi":
        decoder = W2lViterbiDecoder(token_dict)

    else:
        init_kwargs = dict(
            token_dict=token_dict,
            lexicon=args.lexicon_path,
            lang_model=args.lm_path,
            beam=args.beam,
            beam_size_token=args.beam_size_token,
            beam_threshold=args.beam_threshold,
            lm_weight=args.lm_weight,
            word_weight=args.word_weight,
            unk_weight=args.unk_weight,
            sil_weight=args.sil_weight,
        )

        if args.decoder == "kenlm":
            decoder = W2lKenLMDecoder(**init_kwargs)

        else:
            # too slow on CPU, run W2lFairseqLMDecoder on GPU
            init_kwargs["gpu_idx"] = 0
            decoder = W2lFairseqLMDecoder(**init_kwargs)

stt.set_decoder(decoder).streaming_transcribe()
