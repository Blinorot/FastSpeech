import argparse
import json
import os
import warnings

import numpy as np
import torch
from pyexpat import model
from torch.serialization import SourceChangeWarning
from tqdm.auto import tqdm
from typing_extensions import assert_type

import audio
import src.model as module_model
import text
import utils
import waveglow
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
TEXT_CLEANERS = ["english_cleaners"]

def synthesis(model, text, alpha=1.0, gamma=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(DEVICE)
    src_pos = torch.from_numpy(src_pos).long().to(DEVICE)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, gamma=gamma)["mel_output"]
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(extra_data=None):
    # 3 test utterances, 1 from train
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]

    phoneme_tests = [
        "AH0 D IY0 F IH1 B R IH0 L EY2 T ER0 IH1 Z AH0 D IH0 V AY1 S DH AE1 T G IH1 V Z AH0 HH AY1 EH1 N ER0 JH IY0 IH0 L EH1 K T R IH0 K SH AA1 K T UW1 DH AH0 HH AA1 R T AH1 V S AH1 M W AH2 N HH UW1 IH1 Z IH0 N K AA1 R D IY0 AE2 K ER0 EH1 S T",
        "M AE2 S AH0 CH UW1 S AH0 T S IH1 N S T AH0 T UW2 T AH1 V T EH0 K N AA1 L AH0 JH IY0 M EY1 B IY1 B EH1 S T N OW1 N F AO1 R IH1 T S M AE1 TH S AY1 AH0 N S AH0 N D EH1 N JH AH0 N IH1 R IH0 NG EH2 JH AH0 K EY1 SH AH0 N",
        "W AA1 S ER0 S T IY2 N D IH1 S T AH0 N S AO1 R K AE N T AO R AH V IH1 CH R UW1 B IH0 N S T IY2 N M EH1 T R IH0 K IH1 Z AH0 D IH1 S T AH0 N S F AH1 NG K SH AH0 N D IH0 F AY1 N D B IH0 T W IY1 N P R AA2 B AH0 B IH1 L AH0 T IY0 D IH2 S T R AH0 B Y UW1 SH AH0 N Z AA1 N AH0 G IH1 V AH0 N M EH1 T R IH0 K S P EY1 S"
    ]

    true_phoneme_tests = []
    for test in phoneme_tests:
        true_test = []
        for elem in test.split():
            if elem == '_':
                true_test.append(' ')
            else:
                true_test.append(elem)
        true_phoneme_tests.append('_'.join(true_test))

    #data_list = list(text.text_to_sequence(test, TEXT_CLEANERS) for test in tests)
    data_list = list(text._arpabet_to_sequence(test) for test in true_phoneme_tests)

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
        for energy in [0.8, 1]:
            for i, phn in tqdm(enumerate(data_list), desc=f"eval_speed_{speed}_{energy}", total=len(data_list)):
                mel, mel_cuda = synthesis(model, phn, alpha=speed, gamma=energy)
                
                os.makedirs("results", exist_ok=True)
                
                # audio.tools.inv_mel_spec(
                #     mel, f"results/s={speed}_{i}.wav"
                # )
                
                waveglow.inference.inference(
                    mel_cuda, WaveGlow,
                    f"results/s={speed}_{energy}_{i}_waveglow.wav"
                )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Synthesize")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="path to checkpoint config file (default: None)",
    )
    args.add_argument(
        "-p",
        "--pretrained",
        default=None,
        type=str,
        help="path to latest checkpoint to init model weights with it (default: None)",
    )
    
    args = args.parse_args()
    
    assert args.config is not None
    assert args.pretrained is not None, 'Provide model checkpoint to use in script mode'

    with open(args.config, 'r') as f:
        config = ConfigParser(json.load(f))

    logger = config.get_logger("test")

    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    state_dict = torch.load(args.pretrained, map_location=DEVICE)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    run_synthesis(model)
