import json
import os
import re
import time
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
import textgrid
import torch
from text import _arpabet_to_sequence, _clean_text, text_to_sequence
from tqdm.auto import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(data_path, mel_ground_truth, alignment_path, text_cleaners, batch_expand_size):    
    data_path = str(ROOT_PATH / data_path)
    mel_ground_truth = str(ROOT_PATH / mel_ground_truth)
    alignment_path = str(ROOT_PATH / alignment_path)

    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)

        dur_gt_name = os.path.join(
            alignment_path, "npy/ljspeech-mfa-%05d.npy" % (i+1))
        duration = np.load(dur_gt_name)

        character_gt_name = os.path.join(
            alignment_path, "text/ljspeech-mfa-%05d.txt" % (i+1))
        with open(character_gt_name, 'r') as f:
            character = f.readline()

        character = np.array(_arpabet_to_sequence(character))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
            
        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target,
                       "batch_expand_size": batch_expand_size})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


def get_character_duration(intervals):
    sr = 22050 # for ljspeech
    hop_length = 256
    win_length = 1024
    min_times = []
    max_times = []
    letters = []
    bad_tokens = ['sil', 'sp', '_', '~', '', 'spn']
    for i in range(len(intervals)):
        min_times.append(int(intervals[i].minTime * sr))
        max_times.append(int(intervals[i].maxTime * sr))
        letters.append(intervals[i].mark if intervals[i].mark not in bad_tokens else ' ')
    alignments = np.zeros(len(letters), dtype=int)
    for i in range(len(letters)):
        start = (min_times[i] - win_length) // hop_length + 1
        end = (max_times[i] - win_length) // hop_length + 1
        alignments[i] = end - start
    alignments[-1] += 1
    return '_'.join(letters), alignments
