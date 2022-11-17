import os
import warnings

import numpy as np
import torch
from torch.serialization import SourceChangeWarning
from tqdm.auto import tqdm

import audio
import text
import utils
import waveglow
from src.utils import ROOT_PATH

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
TEXT_CLEANERS = ["english_cleaners"]

def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(DEVICE)
    src_pos = torch.from_numpy(src_pos).long().to(DEVICE)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)["mel_output"]
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(extra_data=None):
    # 3 test utterances, 1 from train
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    data_list = list(text.text_to_sequence(test, TEXT_CLEANERS) for test in tests)

    if extra_data is not None: #  utterance from train "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
        data_list.append(extra_data)

    return data_list

def run_synthesis(model, extra_data=None):
    warnings.filterwarnings("ignore", category=SourceChangeWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    assert DEVICE == 'cuda:0', "CPU is not supported, run on the machine with GPU"

    WaveGlow = utils.get_WaveGlow()
    WaveGlow.to(DEVICE)

    # clean from warning files
    patch_files = ['Conv1d', 'ConvTranspose1d', 'Invertible1x1Conv', "ModuleList", "WaveGlow", "WN"]
    for patch in patch_files:
        os.remove(str(ROOT_PATH / f'{patch}.patch'))

    data_list = get_data(extra_data=extra_data)
    for speed in [1]:
        for i, phn in tqdm(enumerate(data_list), desc=f"eval_speed_{speed}", total=len(data_list)):
            mel, mel_cuda = synthesis(model, phn, speed)
            
            os.makedirs("results", exist_ok=True)
            
            audio.tools.inv_mel_spec(
                mel, f"results/s={speed}_{i}.wav"
            )
            
            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                f"results/s={speed}_{i}_waveglow.wav"
            )
