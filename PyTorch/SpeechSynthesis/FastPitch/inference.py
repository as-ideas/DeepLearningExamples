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


def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, default='Dies ist ein Test.',
                        help='Full path to the input text (phareses separated by newlines)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--cuda', action='store_true',
                        help='Run inference on a GPU using CUDA')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Enable cudnn benchmark mode')
    parser.add_argument('--fastpitch', type=str,
                        help='Full path to the generator checkpoint file (skip to use ground truth mels)')
    parser.add_argument('--waveglow', type=str,
                        help='Full path to the WaveGlow model checkpoint file (skip to only generate mels)')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float,
                        help='WaveGlow sigma')
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float,
                        help='WaveGlow denoising')
    parser.add_argument('-sr', '--sampling-rate', default=22050, type=int,
                        help='Sampling rate')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')
    parser.add_argument('--amp', action='store_true',
                        help='Inference with AMP')
    parser.add_argument('-bs', '--batch-size', type=int, default=64)
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Repeat inference for benchmarking')
    parser.add_argument('--torchscript', action='store_true',
                        help='Apply TorchScript')
    parser.add_argument('--ema', action='store_true',
                        help='Use EMA averaged model (if saved in checkpoints)')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset (for loading extra data fields)')

    transform = parser.add_argument_group('transform')
    transform.add_argument('--fade-out', type=int, default=5,
                           help='Number of fadeout frames at the end')
    transform.add_argument('--pace', type=float, default=1.0,
                           help='Adjust the pace of speech')
    transform.add_argument('--pitch-transform-flatten', action='store_true',
                           help='Flatten the pitch')
    transform.add_argument('--pitch-transform-invert', action='store_true',
                           help='Invert the pitch wrt mean value')
    transform.add_argument('--pitch-transform-amplify', type=float, default=1.0,
                           help='Amplify pitch variability, typical values are in the range (1.0, 3.0).')
    transform.add_argument('--pitch-transform-shift', type=float, default=0.0,
                           help='Raise/lower the pitch by <hz>')
    transform.add_argument('--pitch-transform-custom', action='store_true',
                           help='Apply the transform from pitch_transform.py')
    return parser


def load_and_setup_model(model_name, parser, checkpoint, amp, device,
                         unk_args=[], forward_is_infer=False, ema=True,
                         jitable=False):
    model_parser = models.parse_model_args(model_name, parser, add_help=False)
    model_args, model_unk_args = model_parser.parse_known_args()
    unk_args[:] = list(set(unk_args) & set(model_unk_args))

    model_config = models.get_model_config(model_name, model_args)

    model = models.get_model(model_name, model_config, device,
                             forward_is_infer=forward_is_infer,
                             jitable=jitable)

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location=torch.device('cpu'))
        status = ''

        if 'state_dict' in checkpoint_data:
            sd = checkpoint_data['state_dict']
            if ema and 'ema_state_dict' in checkpoint_data:
                sd = checkpoint_data['ema_state_dict']
                status += ' (EMA)'
            elif ema and not 'ema_state_dict' in checkpoint_data:
                print(f'WARNING: EMA weights missing for {model_name}')

            if any(key.startswith('module.') for key in sd):
                sd = {k.replace('module.', ''): v for k,v in sd.items()}
            status += ' ' + str(model.load_state_dict(sd, strict=False))
        else:
            model = checkpoint_data['model']
        print(f'Loaded {model_name}{status}')

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)
    if amp:
        model.half()
    model.eval()
    return model.to(device)


def load_fields(fpath):
    lines = [l.strip() for l in open(fpath, encoding='utf-8')]
    if fpath.endswith('.tsv'):
        columns = lines[0].split('\t')
        fields = list(zip(*[t.split('\t') for t in lines[1:]]))
    else:
        columns = ['text']
        fields = [lines]
    return {c:f for c, f in zip(columns, fields)}


def prepare_input_sequence(fields, device, batch_size=128, dataset=None,
                           load_mels=False, load_pitch=False):
    fields['text'] = [torch.LongTensor(text_to_sequence(t))
                      for t in fields['text']]
    order = np.argsort([-t.size(0) for t in fields['text']])

    fields['text'] = [fields['text'][i] for i in order]
    fields['text_lens'] = torch.LongTensor([t.size(0) for t in fields['text']])

    if load_mels:
        assert 'mel' in fields
        fields['mel'] = [
            torch.load(Path(dataset, fields['mel'][i])).t() for i in order]
        fields['mel_lens'] = torch.LongTensor([t.size(0) for t in fields['mel']])

    if load_pitch:
        assert 'pitch' in fields
        fields['pitch'] = [
            torch.load(Path(dataset, fields['pitch'][i])) for i in order]
        fields['pitch_lens'] = torch.LongTensor([t.size(0) for t in fields['pitch']])

    if 'output' in fields:
        fields['output'] = [fields['output'][i] for i in order]

    # cut into batches & pad
    batches = []
    for b in range(0, len(order), batch_size):
        batch = {f: values[b:b+batch_size] for f, values in fields.items()}
        for f in batch:
            if f == 'text':
                batch[f] = pad_sequence(batch[f], batch_first=True)
            elif f == 'mel' and load_mels:
                batch[f] = pad_sequence(batch[f], batch_first=True).permute(0, 2, 1)
            elif f == 'pitch' and load_pitch:
                batch[f] = pad_sequence(batch[f], batch_first=True)

            if type(batch[f]) is torch.Tensor:
                batch[f] = batch[f].to(device)
        batches.append(batch)

    return batches


def build_pitch_transformation(args):

    if args.pitch_transform_custom:
        def custom_(pitch, pitch_lens, mean, std):
            return (pitch_transform_custom(pitch * std + mean, pitch_lens)
                    - mean) / std
        return custom_

    fun = 'pitch'
    if args.pitch_transform_flatten:
        fun = f'({fun}) * 0.0'
    if args.pitch_transform_invert:
        fun = f'({fun}) * -1.0'
    if args.pitch_transform_amplify:
        ampl = args.pitch_transform_amplify
        fun = f'({fun}) * {ampl}'
    if args.pitch_transform_shift != 0.0:
        hz = args.pitch_transform_shift
        fun = f'({fun}) + {hz} / std'
    return eval(f'lambda pitch, pitch_lens, mean, std: {fun}')


class MeasureTime(list):
    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        self.append(time.perf_counter() - self.t0)

    def __add__(self, other):
        assert len(self) == len(other)
        return MeasureTime(sum(ab) for ab in zip(self, other))


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU.
    """
    parser = argparse.ArgumentParser(description='PyTorch FastPitch Inference',
                                     allow_abbrev=False)
    parser = parse_args(parser)
    args, unk_args = parser.parse_known_args()

    if args.output is not None:
        Path(args.output).mkdir(parents=False, exist_ok=True)

    device = torch.device('cuda' if args.cuda else 'cpu')

    model_path = '/Users/cschaefe/fast_pitch_models/FastPitch_checkpoint_600.pt'

    if args.fastpitch != 'SKIP':
        generator = load_and_setup_model(
            'FastPitch', parser, model_path, args.amp, device,
            unk_args=unk_args, forward_is_infer=True, ema=args.ema,
            jitable=args.torchscript)

        if args.torchscript:
            generator = torch.jit.script(generator)
    else:
        generator = None

    with open('sentences.txt', 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    def pitch_transform(x, *args):
        return x * 2.

    for i, text in enumerate(sentences, 1):
        print(f'synthing {i} / {len(sentences)}')
        text = basic_cleaners(text)
        inputs = text_to_sequence(text)
        inputs = torch.tensor(inputs).unsqueeze(0)
        with torch.no_grad():
            mel_out, dec_lens, dur_pred, pitch_pred = generator.infer(inputs, None, pitch_transform=pitch_transform)

        wav = reconstruct_waveform(mel_out.squeeze().numpy())
        sf.write(f'/tmp/fast_pitch/{i}_griffinlim.wav', wav.astype(np.float32), samplerate=22050)
        torch.save(mel_out, f'/tmp/fast_pitch/{i}_melgan.mel')


if __name__ == '__main__':
    main()
