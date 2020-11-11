# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import argparse
import sys
import time
from pathlib import Path

import dllogger as DLLogger
import numpy as np
import torch
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity
from torch.nn.utils.rnn import pad_sequence

import models
from common.tb_dllogger import (init_inference_metadata, stdout_metric_format,
                                unique_log_fpath)
from common.text import text_to_sequence, basic_cleaners
from fastpitch.model import FastPitch
from pitch_transform import pitch_transform_custom
from waveglow import model as glow
import matplotlib.pyplot as plt
sys.modules['glow'] = glow
import soundfile as sf
import librosa

def reconstruct_waveform(mel, n_iter=32):
    """Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform."""
    denormalized = np.exp(mel)
    S = librosa.feature.inverse.mel_to_stft(
        denormalized, power=1, sr=22050,
        n_fft=1024, fmin=0, fmax=8000)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=256, win_length=1024)
    return wav



def plot_mel(mel: np.array):
    mel = np.flip(mel, axis=0)
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.imshow(mel, interpolation='nearest', aspect='auto')
    return fig



if __name__ == '__main__':


    model_path = '/Users/cschaefe/fast_pitch_models/model.pyt'
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    config = checkpoint['config']

    print(config['n_symbols'])
    model = FastPitch(**config)
    model.load_state_dict(checkpoint['state_dict'])
    input = torch.tensor(text_to_sequence(basic_cleaners('Hallo Welt.'))).unsqueeze(0)

    mel_out, _, _, _ = model.infer(input, None)
    wav = reconstruct_waveform(mel_out.detach().squeeze().numpy())
    sf.write(f'/tmp/fast_pitch/{0}_griffinlim.wav', wav.astype(np.float32), samplerate=22050)
    torch.save(mel_out, f'/tmp/fast_pitch/{0}_melgan.mel')
