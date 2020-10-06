
import torch
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import numpy as np
from inference import plot_mel, reconstruct_waveform

mel = torch.load('/Users/cschaefe/datasets/audio_data/Cutted_merged_fastpitch/mels/02792.pt')


wav = reconstruct_waveform(mel.squeeze().numpy())

sf.write('/tmp/test_fastpitch.wav', wav.astype(np.float32), samplerate=22050)
print(mel)