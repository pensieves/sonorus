from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleComputationOptions
from kaldi.asr import NnetLatticeFasterRecognizer
from kaldi.alignment import NnetAligner
from kaldi.fstext import SymbolTable
from kaldi.util.table import SequentialMatrixReader

import soundfile as sf
from tempfile import gettempdir
from pathlib import Path
import shutil
import numpy as np
import re

from ...utilities.utils import create_random_dir
from ...audio.utils import audio_float2int


def simplify_phoneme(phoneme):
    r"""remove digit as count id of phonemes and further sub-categories followed 
    by _ e.g. AE1_I will be simplified to AE"""
    return re.sub(r"(\d+|_.$)", "", phoneme)


class PhonemeSegmenter(object):
    def __init__(
        self,
        model_rxfilename,
        graph_rxfilename,
        symbols_filename,
        tree_rxfilename,
        lexicon_rxfilename,
        disambig_rxfilename,
        phoneme_file,
        mfcc_conf,
        ivec_conf,
        decoder_opts=dict(beam=13, max_active=7000),
        decodable_opts=dict(
            acoustic_scale=1.0, frame_subsampling_factor=3, frames_per_chunk=150
        ),
        simplify_phoneme=simplify_phoneme,  # set to None if not desired
        as_dict=True,  # return phonemes info as dict or tuple
        pos2key=(
            "name",
            "start",
            "duration",
        ),  # keys to be used for dict while repacking
        work_dir=None,  # temp directories will be used for intermediate steps
    ):

        decoder_options = LatticeFasterDecoderOptions()
        decoder_options.beam = decoder_opts["beam"]
        decoder_options.max_active = decoder_opts["max_active"]

        decodable_options = NnetSimpleComputationOptions()
        decodable_options.acoustic_scale = decodable_opts["acoustic_scale"]
        decodable_options.frame_subsampling_factor = decodable_opts[
            "frame_subsampling_factor"
        ]
        decodable_options.frames_per_chunk = decodable_opts["frames_per_chunk"]

        self.asr = NnetLatticeFasterRecognizer.from_files(
            model_rxfilename=model_rxfilename,
            graph_rxfilename=graph_rxfilename,
            symbols_filename=symbols_filename,
            decoder_opts=decoder_options,
            decodable_opts=decodable_options,
        )

        self.aligner = NnetAligner.from_files(
            model_rxfilename=model_rxfilename,
            tree_rxfilename=tree_rxfilename,
            lexicon_rxfilename=lexicon_rxfilename,
            symbols_filename=symbols_filename,
            disambig_rxfilename=disambig_rxfilename,
            decodable_opts=decodable_options,
        )

        self.phoneme_table = SymbolTable.read_text(phoneme_file)
        self.mfcc_conf = mfcc_conf
        self.ivec_conf = ivec_conf

        self.simplify_phoneme = simplify_phoneme

        self.as_dict = as_dict
        self.pos2key = pos2key

        self.work_dir = work_dir if work_dir else gettempdir()

    def segment(self, audio, sample_rate=22050, clean_up=True):

        temp_dir = create_random_dir(work_dir=self.work_dir)
        audio_file = temp_dir / "audio.wav"
        wav_scp = temp_dir / "wav.scp"
        spk2utt = temp_dir / "spk2utt"

        if len(audio) == 1:  # single row but given as 2D matrix
            audio = audio[0]
        if np.issubdtype(audio.dtype, np.floating):
            audio = audio_float2int(audio)

        sf.write(file=audio_file, data=audio, samplerate=sample_rate)
        with open(wav_scp, "w") as f:
            f.write(f"utt1 {audio_file}")
        with open(spk2utt, "w") as f:
            f.write("utt1 utt1")

        segmented_phoneme = self.segment_from_file(wav_scp, spk2utt)

        if clean_up:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return segmented_phoneme

    def segment_from_file(self, wav_scp, spk2utt):
        return self.phoneme_from_rspecs(*self.get_rspecs(wav_scp, spk2utt))

    def phoneme_from_rspecs(self, feats_rspec, ivectors_rspec):

        aligned_phonemes = dict()

        with SequentialMatrixReader(feats_rspec) as f, SequentialMatrixReader(
            ivectors_rspec
        ) as i:

            for (key, feats), (_, ivectors) in zip(f, i):

                decoded = self.asr.decode((feats, ivectors))

                phoneme_alignment = self.aligner.to_phone_alignment(
                    decoded["alignment"], self.phoneme_table
                )

                phoneme_alignment = self.repack(self.simplify(phoneme_alignment))

                aligned_phonemes[key] = dict(
                    text=decoded["text"], phonemes=phoneme_alignment
                )

        return aligned_phonemes

    def simplify(self, phoneme_alignment):
        if self.simplify_phoneme:
            phoneme_alignment = [
                (self.simplify_phoneme(i[0]), i[1], i[2]) for i in phoneme_alignment
            ]
        return phoneme_alignment

    def repack(self, phoneme_alignment):
        if self.as_dict:
            phoneme_alignment = [dict(zip(self.pos2key, i)) for i in phoneme_alignment]
        return phoneme_alignment

    def get_rspecs(self, wav_scp, spk2utt):

        feats_rspec = (
            f"ark:compute-mfcc-feats --config={self.mfcc_conf} scp:{wav_scp} ark:- |"
        )

        ivectors_rspec = (
            f"ark:compute-mfcc-feats --config={self.mfcc_conf} scp:{wav_scp} ark:- |"
            f"ivector-extract-online2 --config={self.ivec_conf} ark:{spk2utt} ark:- ark:- |"
        )

        return feats_rspec, ivectors_rspec
