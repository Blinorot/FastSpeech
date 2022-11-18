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

def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(DEVICE)
    src_pos = torch.from_numpy(src_pos).long().to(DEVICE)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha)["mel_output"]
        print(mel.shape)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(extra_data=None):
    # 3 test utterances, 1 from train
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]

    phoneme_tests = [
        "EY _ D IY F IH B R IH L EY T ER _ IH Z _ EY _ D IH V AY S _ DH AE T _ G IH V Z _ EY _ HH AY _ EH N ER JH IY _ IH L EH K T R IH K _ SH AA K _ T UW _ DH AH _ HH AA R T _ AH V _ S AH M W AH N _ HH UW _ IH Z _ IH N _ K AA R D IY AE K _ ER EH S T",
        "M AE S AH CH UW S AH T S _ IH N S T IH T UW T _ AH V _ T EH K N AA L AH JH IY _ M EY _ B IY _ B EH S T _ N OW N _ F AO R _ IH T S _ M AE TH _ S AY AH N S _ AH N D _ EH N JH AH N IH R IH NG _ EH JH AH K EY SH AH N",
        "W AA S ER S T IY N _ D IH S T AH N S _ AO R _ K AE N T AO R AH V AH CH _ R UW B IH N S T IY N _ M EH T R IH K _ IH Z _ EY _ D IH S T AH N S _ F AH NG K SH AH N _ D IH F AY N D _ B IH T W IY N _ P R AA B AH B IH L AH T IY _ D IH S T R AH B Y UW SH AH N Z _ AA N _ EY _ G IH V AH N _ M EH T R IH K _ S P EY S"
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

    print(data_list)

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
