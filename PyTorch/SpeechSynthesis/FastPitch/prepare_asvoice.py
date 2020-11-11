import pickle
from pathlib import Path
from random import Random
from typing import Union
import torch
import numpy as np

from common.text import basic_cleaners

def unpickle_binary(file: Union[str, Path]):
    with open(str(file), 'rb') as f:
        return pickle.load(f)

def get_files(path: Union[str, Path], extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def ljspeech(csv_file):
    text_dict = {}
    with open(csv_file, encoding='utf-8') as f:
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = basic_cleaners(split[1])
    return text_dict


data_path = Path('/Users/cschaefe/datasets/asvoice2_fastpitch')

csv_path = data_path / 'metadata.csv'
dur_path = data_path / 'alg'
dur_path_torch = data_path / 'durations'
dur_path_torch.mkdir(parents=True, exist_ok=True)
(data_path / 'mels').mkdir(parents=True, exist_ok=True)

train_meta_file = data_path / 'metadata_train.txt'
val_meta_file = data_path / 'metadata_val.txt'
train_meta_file_taco = data_path / 'metadata_train_taco.txt'
val_meta_file_taco = data_path / 'metadata_val_taco.txt'

text_dict = unpickle_binary(data_path / 'text_dict.pkl')

durs = get_files(dur_path, extension='npy')

file_ids = [d.stem for d in durs]
file_ids.sort()
random = Random(42)
random.shuffle(file_ids)

val_ids = file_ids[:100]
train_ids = file_ids[100:]

print('preparing fast pitch text')
train_strings = [f'mels/{item_id}.pt|durations/{item_id}.pt|pitch_char/{item_id}.pt|{text_dict[item_id]}\n' for item_id in train_ids]
val_strings = [f'mels/{item_id}.pt|durations/{item_id}.pt|pitch_char/{item_id}.pt|{text_dict[item_id]}\n' for item_id in val_ids]
with open(train_meta_file, 'w+', encoding='utf-8') as f:
    f.write(''.join(train_strings))
with open(val_meta_file, 'w+', encoding='utf-8') as f:
    f.write(''.join(val_strings))

print('taco text')
train_strings = [f'mel/{item_id}.npy|{text_dict[item_id]}\n' for item_id in train_ids]
val_strings = [f'mel/{item_id}.npy|{text_dict[item_id]}\n' for item_id in val_ids]
with open(train_meta_file_taco, 'w+', encoding='utf-8') as f:
    f.write(''.join(train_strings))
with open(val_meta_file_taco, 'w+', encoding='utf-8') as f:
    f.write(''.join(val_strings))


for item_id in file_ids:
    dur = np.load(dur_path / f'{item_id}.npy')
    dur_torch = torch.tensor(dur).int()
    torch.save(dur_torch, dur_path_torch / f'{item_id}.pt')