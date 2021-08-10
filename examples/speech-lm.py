import sys
from pathlib import Path

path = str(Path(__file__).parents[1].resolve())
sys.path.append(path)

from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer
import numpy as np

from sonorus.speech.lm import (
    FairseqTokenDictionary,
    W2lKenLMDecoder,
    W2lViterbiDecoder,
    W2lFairseqLMDecoder,
)

import optuna
from optuna.integration import BoTorchSampler
import joblib


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def map_to_pred(batch):
    input_values = processor(
        batch["speech"], return_tensors="pt", padding="longest"
    ).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch


def map_to_pred_lm(batch, decoder):
    input_values = processor(
        batch["speech"], return_tensors="pt", padding="longest"
    ).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    logits = logits.float().cpu().contiguous()
    decoded = decoder.decode(logits)
    # 1st sample, 1st best transcription
    transcription = decoder.post_process(decoded)
    batch["transcription"] = transcription
    return batch


def get_wer(result, batch_size=-1, lm=False):
    def transcripts():
        return (
            [x[0] for x in result["transcription"]] if lm else result["transcription"]
        )

    errors = []

    if batch_size > 0:
        for i in range(0, len(result), batch_size):
            errors.append(
                wer(
                    result["text"][i : i + batch_size],
                    transcripts()[i : i + batch_size],
                )
            )
    else:
        errors.append(wer(result["text"], transcripts()))

    return np.mean(errors)


librispeech_eval = load_dataset(
    "librispeech_asr",
    "clean",
    split="validation",
    # split="test",
    ignore_verifications=True,
)  # ,
# download_mode="force_redownload")


model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

librispeech_eval = librispeech_eval.map(map_to_array)

result = librispeech_eval.map(
    map_to_pred, batched=True, batch_size=1, remove_columns=["speech"]
)
print("Acoustic WER:", get_wer(result, batch_size=1000, lm=False))

token_dict = FairseqTokenDictionary(indexed_symbols=processor.get_vocab())

lexicon_path = "/home/harold/Documents/IISc-work/imperio/data/speech/fairseq/librispeech_lexicon.lst"
lm_path = "/home/harold/Documents/IISc-work/imperio/data/speech/fairseq/lm_librispeech_kenlm_word_4g_200kvocab.bin"

# decoder = W2lKenLMDecoder(
#     token_dict=token_dict,
#     lexicon=lexicon_path,
#     lang_model=lm_path,
#     beam=1500,
#     beam_size_token=100,
#     beam_threshold=25,
#     lm_weight=1.5,
#     word_weight=-1,
#     unk_weight=float("-inf"),
#     sil_weight=0,
# )

# result = librispeech_eval.map(lambda batch: map_to_pred_lm(batch, decoder), batched=True, batch_size=1, remove_columns=["speech"])
# print("KenLM WER:", get_wer(result, batch_size=1000, lm=True))

n_startup_trials = 10
bayes_opt_sampler = BoTorchSampler(n_startup_trials=n_startup_trials)
study = optuna.create_study(sampler=bayes_opt_sampler)


def objective(trial):

    lm_weight = trial.suggest_float("lm_weight", 0, 5)
    word_weight = trial.suggest_float("word_weight", -5, 5)
    sil_weight = trial.suggest_float("sil_weight", -5, 5)

    decoder = W2lKenLMDecoder(
        token_dict=token_dict,
        lexicon=lexicon_path,
        lang_model=lm_path,
        beam=500,
        beam_size_token=100,
        beam_threshold=25,
        lm_weight=lm_weight,
        word_weight=word_weight,
        unk_weight=float("-inf"),
        sil_weight=sil_weight,
    )

    result = librispeech_eval.map(
        lambda batch: map_to_pred_lm(batch, decoder),
        batched=True,
        batch_size=1,
        remove_columns=["speech"],
    )

    return get_wer(result, batch_size=1000, lm=True)


n_trials = 150
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
print("Best KenLM WER: ", study.best_value)
print("Best params: ", study.best_params)
joblib.dump(study, "speech-lm-hyperparams-opt-study.jb")
