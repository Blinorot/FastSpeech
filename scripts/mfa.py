import os
from ctypes import alignment
from pathlib import Path

import numpy as np
import textgrid
from tqdm.auto import tqdm


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

def process():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    alignment_path = data_dir / 'MFA'
    alignment_path_text = data_dir / 'MFA_2' / 'text'
    alignment_path_npy = data_dir / 'MFA_2' / 'npy'
    alignment_path_text.mkdir(exist_ok=True, parents=True)
    alignment_path_npy.mkdir(exist_ok=True, parents=True)

    for i, fpath in tqdm(enumerate(alignment_path.iterdir())):
        duration = textgrid.TextGrid.fromFile(str(fpath))[1]
        character, duration = get_character_duration(duration)
        np.save(alignment_path_npy / f'{fpath.name[:-9]}.npy', duration)
        with open(str(alignment_path_text / f'{fpath.name[:-9]}.txt'), 'w') as f:
            f.write(character)


if __name__ == '__main__':
    process()
